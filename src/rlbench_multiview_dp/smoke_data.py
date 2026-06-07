import argparse
from pathlib import Path

from .data import RlbenchHeuristicWaypointDataset, build_text_vocab
from .common import parse_tasks, read_jsonl
from .train import parse_lowdim_roots, parse_rgb_roots


def main():
    parser = argparse.ArgumentParser(description="Lightweight data-path smoke test for the RLBench multi-view DP repo.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rgb-root-200", default=None)
    parser.add_argument("--rgb-root-400", default=None)
    parser.add_argument("--rgb-root", action="append", default=None, help="Extra mapping source_bundle=/path")
    parser.add_argument("--lowdim-root-200", default=None)
    parser.add_argument("--lowdim-root-400", default=None)
    parser.add_argument("--lowdim-root", action="append", default=None, help="Extra mapping source_bundle=/path")
    parser.add_argument("--split", default="train")
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--sample-every-n", type=int, default=0)
    parser.add_argument("--view-names", default="front,left_shoulder,right_shoulder")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--validate-image-paths", action="store_true")
    parser.add_argument("--proprio-mode", choices=("ee_rotvec", "ee_rpy"), default="ee_rotvec")
    args = parser.parse_args()

    rgb_roots = parse_rgb_roots(args)
    lowdim_roots = parse_lowdim_roots(args, fallback_roots=rgb_roots)
    rows = read_jsonl(Path(args.manifest))
    task_filter = parse_tasks(args.task)
    vocab_rows = [r for r in rows if task_filter is None or r.get("task") in task_filter]
    text_to_idx = build_text_vocab(vocab_rows)
    dataset = RlbenchHeuristicWaypointDataset(
        manifest_path=args.manifest,
        rgb_roots=rgb_roots,
        lowdim_roots=lowdim_roots,
        split=args.split,
        tasks=args.task,
        obs_horizon=args.obs_horizon,
        sample_every_n=args.sample_every_n,
        view_names=args.view_names.split(","),
        image_size=args.image_size,
        crop_size=args.crop_size,
        random_crop=False,
        max_episodes=args.max_episodes,
        max_episodes_per_task=args.max_episodes_per_task,
        proprio_mode=args.proprio_mode,
        text_to_idx=text_to_idx,
        validate_image_paths=args.validate_image_paths,
        loose_pickle=True,
    )
    proprio_mean, proprio_std = dataset.compute_proprio_stats()
    action_mean, action_std = dataset.compute_action_stats()
    dataset.set_proprio_stats(proprio_mean, proprio_std)

    print(f"manifest={Path(args.manifest).resolve()}")
    print(f"rgb_roots={{{', '.join(f'{k}: {v}' for k, v in sorted(rgb_roots.items()))}}}")
    print(f"lowdim_roots={{{', '.join(f'{k}: {v}' for k, v in sorted(lowdim_roots.items()))}}}")
    print(f"episodes={len(dataset.episode_rows)} samples={len(dataset)} texts={len(text_to_idx)}")
    print(f"action_mean={action_mean.tolist()}")
    print(f"action_std={action_std.tolist()}")
    print(f"proprio_mean={proprio_mean.tolist()}")
    print(f"proprio_std={proprio_std.tolist()}")
    for i in range(min(int(args.num_samples), len(dataset))):
        item = dataset[i]
        sample = dataset.samples[i]
        print(
            "sample "
            f"{i}: task={sample.task} episode={sample.episode} "
            f"current={sample.current_frame_idx} target={sample.target_frame_idx} "
            f"rgb_dir={sample.rgb_episode_dir} lowdim_dir={sample.lowdim_episode_dir} "
            f"obs_frames={sample.obs_frame_indices} image_shape={tuple(item['image'].shape)} "
            f"proprio_shape={tuple(item['proprio'].shape)} action={item['action'][0].numpy().round(4).tolist()} "
            f"text={sample.task_instruction!r}"
        )


if __name__ == "__main__":
    main()
