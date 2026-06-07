import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .common import filter_manifest_rows, parse_root_mapping, parse_tasks, read_jsonl, write_json
from .data import RlbenchHeuristicWaypointDataset, build_text_vocab, collate_waypoint_batch, text_vocab_list
from .diffusion import DDPMNoiseScheduler, EMAModel
from .model import MultiViewDiffusionPolicy
from .text import dummy_text_features, encode_clip_texts


def bool_env(value) -> bool:
    return str(value).lower() in ("1", "true", "yes", "y", "on")


def parse_rgb_roots(args):
    return parse_root_mapping(args.rgb_root_200, args.rgb_root_400, args.rgb_root, root_name="RGB root", required=True)


def parse_lowdim_roots(args, fallback_roots=None):
    roots = parse_root_mapping(
        args.lowdim_root_200,
        args.lowdim_root_400,
        args.lowdim_root,
        root_name="low-dim root",
        required=False,
    )
    return roots or dict(fallback_roots or {})


def make_lr_scheduler(optimizer, max_steps, warmup_steps, min_lr_ratio):
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def raw_model(model):
    return model.module if isinstance(model, DDP) else model


def make_optimizer(model, lr, visual_lr, weight_decay):
    base_model = raw_model(model)
    visual_params = list(base_model.obs_encoder.visual_parameters())
    visual_param_ids = {id(param) for param in visual_params}
    other_params = [param for param in model.parameters() if id(param) not in visual_param_ids]
    return torch.optim.AdamW(
        [
            {"params": other_params, "lr": float(lr)},
            {"params": visual_params, "lr": float(visual_lr)},
        ],
        weight_decay=float(weight_decay),
    )


def save_checkpoint(path, model, ema, config, action_mean, action_std, proprio_mean, proprio_std, text_features, texts, step, val_loss, optimizer=None, scheduler=None):
    payload = {
        "model": raw_model(model).state_dict(),
        "ema_model": ema.state_dict() if ema is not None else None,
        "config": config,
        "action_mean": np.asarray(action_mean, dtype=np.float32),
        "action_std": np.asarray(action_std, dtype=np.float32),
        "proprio_mean": np.asarray(proprio_mean, dtype=np.float32),
        "proprio_std": np.asarray(proprio_std, dtype=np.float32),
        "text_features": np.asarray(text_features, dtype=np.float32),
        "texts": list(texts),
        "step": int(step),
        "val_loss": None if val_loss is None else float(val_loss),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def build_datasets(args, rgb_roots, lowdim_roots, text_to_idx):
    common = dict(
        manifest_path=args.manifest,
        rgb_roots=rgb_roots,
        lowdim_roots=lowdim_roots,
        tasks=args.task,
        obs_horizon=args.obs_horizon,
        sample_every_n=args.sample_every_n,
        view_names=args.view_names.split(","),
        image_size=args.image_size,
        crop_size=args.crop_size,
        proprio_mode=args.proprio_mode,
        text_to_idx=text_to_idx,
        validate_image_paths=args.validate_image_paths,
        loose_pickle=True,
    )
    train_set = RlbenchHeuristicWaypointDataset(
        split=args.train_split,
        random_crop=args.random_crop,
        max_episodes=args.max_train_episodes,
        max_episodes_per_task=args.max_train_episodes_per_task,
        **common,
    )
    val_set = None
    if args.eval_every > 0:
        val_set = RlbenchHeuristicWaypointDataset(
            split=args.val_split,
            random_crop=False,
            max_episodes=args.max_val_episodes,
            max_episodes_per_task=args.max_val_episodes_per_task,
            **common,
        )
    return train_set, val_set


def build_text_table(args, text_to_idx, device, is_main):
    texts = text_vocab_list(text_to_idx)
    if args.dummy_text_dim > 0:
        if is_main:
            print(f"Using deterministic dummy text features dim={args.dummy_text_dim}", flush=True)
        return dummy_text_features(texts, dim=args.dummy_text_dim, seed=args.seed), texts
    if is_main:
        print(f"Encoding {len(texts)} CLIP texts with {args.clip_model}", flush=True)
    features = encode_clip_texts(
        texts,
        model_name=args.clip_model,
        device=device.type,
        batch_size=args.clip_batch_size,
        local_files_only=args.clip_local_files_only,
    )
    return features, texts


def move_batch(batch, device, text_features):
    image = batch["image"].to(device, non_blocking=True)
    proprio = batch["proprio"].to(device, non_blocking=True)
    action = batch["action"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    text_token = text_features[batch["text_idx"].to(device, non_blocking=True)]
    return image, proprio, text_token, action, mask


def loss_on_batch(model, batch, device, text_features, scheduler, action_mean, action_std, amp_enabled, amp_dtype, gripper_loss_weight):
    image, proprio, text_token, action, mask = move_batch(batch, device, text_features)
    mean = action_mean.view(1, 1, -1)
    std = action_std.view(1, 1, -1)
    action_norm = (action - mean) / std
    noise = torch.randn_like(action_norm)
    timesteps = torch.randint(0, scheduler.train_steps, (action.shape[0],), device=device, dtype=torch.long)
    noisy = scheduler.add_noise(action_norm, noise, timesteps)
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
        pred = model(noisy, timesteps, image, proprio, text_token)
    weight = torch.ones((1, 1, action.shape[-1]), dtype=torch.float32, device=device)
    weight[..., -1] = float(gripper_loss_weight)
    mse = (pred.float() - noise.float()).square() * weight
    denom = mask.sum().clamp_min(1.0) * action.shape[-1]
    return (mse * mask[..., None]).sum() / denom


@torch.no_grad()
def evaluate(model, loader, device, text_features, scheduler, action_mean, action_std, amp_enabled, amp_dtype, gripper_loss_weight, max_batches=None):
    model.eval()
    losses = []
    for batch_i, batch in enumerate(loader):
        if max_batches is not None and batch_i >= int(max_batches):
            break
        loss = loss_on_batch(
            model,
            batch,
            device,
            text_features,
            scheduler,
            action_mean,
            action_std,
            amp_enabled,
            amp_dtype,
            gripper_loss_weight,
        )
        losses.append(float(loss.detach().cpu()))
    model.train()
    return float(np.mean(losses)) if losses else None


def train(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank() if distributed else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    if args.device == "cuda" and torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    out_dir = Path(args.out_dir).resolve()
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    rgb_roots = parse_rgb_roots(args)
    lowdim_roots = parse_lowdim_roots(args, fallback_roots=rgb_roots)
    all_rows = read_jsonl(args.manifest)
    task_filter = parse_tasks(args.task)
    vocab_rows = [r for r in all_rows if task_filter is None or r.get("task") in task_filter]
    text_to_idx = build_text_vocab(vocab_rows)
    text_features_np, texts = build_text_table(args, text_to_idx, device, is_main)
    train_set, val_set = build_datasets(args, rgb_roots, lowdim_roots, text_to_idx)
    action_mean_np, action_std_np = train_set.compute_action_stats()
    proprio_mean_np, proprio_std_np = train_set.compute_proprio_stats()
    train_set.set_proprio_stats(proprio_mean_np, proprio_std_np)
    if val_set is not None:
        val_set.set_proprio_stats(proprio_mean_np, proprio_std_np)

    if is_main:
        print(
            f"Built datasets: train_samples={len(train_set)} val_samples={0 if val_set is None else len(val_set)} "
            f"texts={len(texts)} action_mean={action_mean_np.tolist()}",
            flush=True,
        )

    train_sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed) if distributed else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0 and args.persistent_workers,
        collate_fn=collate_waypoint_batch,
        drop_last=True,
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0 and args.persistent_workers,
            collate_fn=collate_waypoint_batch,
            drop_last=False,
        )

    unet_dims = tuple(int(x) for x in args.unet_dims.split(",") if x.strip())
    text_dim = int(text_features_np.shape[-1])
    model = MultiViewDiffusionPolicy(
        obs_horizon=args.obs_horizon,
        num_views=len(args.view_names.split(",")),
        proprio_dim=7,
        text_dim=text_dim,
        action_dim=7,
        action_horizon=1,
        visual_backbone=args.visual_backbone,
        visual_feature_dim=args.visual_feature_dim,
        imagenet_pretrained=args.imagenet_pretrained,
        group_norm=not args.no_group_norm,
        share_visual_encoder=args.share_visual_encoder,
        global_cond_dim=args.global_cond_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        unet_dims=unet_dims,
        dropout=args.dropout,
    ).to(device)
    ema = EMAModel(model, decay=args.ema_decay).to(device) if args.use_ema else None
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = make_optimizer(model, args.lr, args.visual_lr, args.weight_decay)
    lr_scheduler = make_lr_scheduler(optimizer, args.max_steps, args.warmup_steps, args.min_lr_ratio)
    noise_scheduler = DDPMNoiseScheduler(args.diffusion_steps, device=device)
    text_features = torch.as_tensor(text_features_np, dtype=torch.float32, device=device)
    action_mean = torch.as_tensor(action_mean_np, dtype=torch.float32, device=device)
    action_std = torch.as_tensor(action_std_np, dtype=torch.float32, device=device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and args.amp_dtype == "float16")

    config = vars(args).copy()
    config.update(
        {
            "rgb_roots": {k: str(v) for k, v in rgb_roots.items()},
            "lowdim_roots": {k: str(v) for k, v in lowdim_roots.items()},
            "texts": texts,
            "text_dim": text_dim,
            "action_dim": 7,
            "action_horizon": 1,
            "execute_horizon": 1,
            "proprio_dim": 7,
            "share_visual_encoder": bool(args.share_visual_encoder),
            "policy": "robomimic_style_multiview_resnet_clip_waypoint_dp",
            "target": "next_full_task_heuristic_waypoint_absolute_rpy7",
        }
    )
    if is_main:
        write_json(out_dir / "config.json", config)

    best_val = None
    step = 0
    model.train()
    progress = tqdm(total=args.max_steps, disable=not is_main)
    while step < args.max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(step)
        for batch in train_loader:
            if step >= args.max_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = loss_on_batch(
                model,
                batch,
                device,
                text_features,
                noise_scheduler,
                action_mean,
                action_std,
                amp_enabled,
                amp_dtype,
                args.gripper_loss_weight,
            )
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            if ema is not None:
                ema.update(raw_model(model))
            step += 1
            if is_main and step % args.log_every == 0:
                progress.set_postfix(loss=f"{float(loss.detach().cpu()):.5f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")
            if is_main:
                progress.update(1)
            do_eval = val_loader is not None and args.eval_every > 0 and step % args.eval_every == 0
            do_save = is_main and args.save_every > 0 and step % args.save_every == 0
            if do_eval:
                eval_model = ema.ema_model if ema is not None else raw_model(model)
                val_loss = evaluate(
                    eval_model,
                    val_loader,
                    device,
                    text_features,
                    noise_scheduler,
                    action_mean,
                    action_std,
                    amp_enabled,
                    amp_dtype,
                    args.gripper_loss_weight,
                    max_batches=args.max_eval_batches,
                )
                if is_main:
                    print(f"step={step} val_loss={val_loss}", flush=True)
                    if best_val is None or (val_loss is not None and val_loss < best_val):
                        best_val = val_loss
                        save_checkpoint(
                            out_dir / "best.pt",
                            model,
                            ema,
                            config,
                            action_mean_np,
                            action_std_np,
                            proprio_mean_np,
                            proprio_std_np,
                            text_features_np,
                            texts,
                            step,
                            val_loss,
                        )
            if do_save:
                save_checkpoint(
                    out_dir / f"step_{step}.pt",
                    model,
                    ema,
                    config,
                    action_mean_np,
                    action_std_np,
                    proprio_mean_np,
                    proprio_std_np,
                    text_features_np,
                    texts,
                    step,
                    best_val,
                )
                save_checkpoint(
                    out_dir / "latest.pt",
                    model,
                    ema,
                    config,
                    action_mean_np,
                    action_std_np,
                    proprio_mean_np,
                    proprio_std_np,
                    text_features_np,
                    texts,
                    step,
                    best_val,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                )
    if is_main:
        save_checkpoint(
            out_dir / "last.pt",
            model,
            ema,
            config,
            action_mean_np,
            action_std_np,
            proprio_mean_np,
            proprio_std_np,
            text_features_np,
            texts,
            step,
            best_val,
        )
        if not (out_dir / "latest.pt").exists():
            save_checkpoint(
                out_dir / "latest.pt",
                model,
                ema,
                config,
                action_mean_np,
                action_std_np,
                proprio_mean_np,
                proprio_std_np,
                text_features_np,
                texts,
                step,
                best_val,
            )
        progress.close()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rgb-root-200", default=None)
    parser.add_argument("--rgb-root-400", default=None)
    parser.add_argument("--rgb-root", action="append", default=None, help="Extra mapping source_bundle=/path")
    parser.add_argument("--lowdim-root-200", default=None)
    parser.add_argument("--lowdim-root-400", default=None)
    parser.add_argument("--lowdim-root", action="append", default=None, help="Extra mapping source_bundle=/path")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument("--max-train-episodes-per-task", type=int, default=None)
    parser.add_argument("--max-val-episodes", type=int, default=None)
    parser.add_argument("--max-val-episodes-per-task", type=int, default=None)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--sample-every-n", type=int, default=0)
    parser.add_argument("--view-names", default="front,left_shoulder,right_shoulder")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--random-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-image-paths", action="store_true")
    parser.add_argument("--proprio-mode", choices=("ee_rotvec", "ee_rpy"), default="ee_rotvec")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--clip-batch-size", type=int, default=64)
    parser.add_argument("--clip-local-files-only", action="store_true")
    parser.add_argument("--dummy-text-dim", type=int, default=0)
    parser.add_argument("--visual-backbone", choices=("resnet18", "resnet34"), default="resnet18")
    parser.add_argument("--visual-feature-dim", type=int, default=64)
    parser.add_argument("--imagenet-pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-group-norm", action="store_true")
    parser.add_argument("--share-visual-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--global-cond-dim", type=int, default=512)
    parser.add_argument("--fusion-hidden-dim", type=int, default=512)
    parser.add_argument("--unet-dims", default="256,512,1024")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--gripper-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--visual-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
