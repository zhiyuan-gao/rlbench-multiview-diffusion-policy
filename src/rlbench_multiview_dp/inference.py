from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .common import VIEW_NAMES, VIEW_TO_ATTR, choose_checkpoint, obs_to_proprio, torch_load
from .data import IMAGENET_MEAN, IMAGENET_STD
from .diffusion import sample_ddim
from .model import MultiViewDiffusionPolicy


def build_model_from_config(cfg, text_dim):
    unet_dims = cfg.get("unet_dims", "256,512,1024")
    if isinstance(unet_dims, str):
        unet_dims = tuple(int(x) for x in unet_dims.split(",") if x.strip())
    return MultiViewDiffusionPolicy(
        obs_horizon=int(cfg.get("obs_horizon", 2)),
        num_views=len(str(cfg.get("view_names", "front,left_shoulder,right_shoulder")).split(",")),
        proprio_dim=int(cfg.get("proprio_dim", 7)),
        text_dim=int(text_dim),
        action_dim=int(cfg.get("action_dim", 7)),
        action_horizon=int(cfg.get("action_horizon", 1)),
        visual_backbone=cfg.get("visual_backbone", "resnet18"),
        visual_feature_dim=int(cfg.get("visual_feature_dim", 64)),
        imagenet_pretrained=bool(cfg.get("imagenet_pretrained", False)),
        group_norm=not bool(cfg.get("no_group_norm", False)),
        share_visual_encoder=bool(cfg.get("share_visual_encoder", False)),
        global_cond_dim=int(cfg.get("global_cond_dim", 512)),
        fusion_hidden_dim=int(cfg.get("fusion_hidden_dim", 512)),
        unet_dims=unet_dims,
        dropout=float(cfg.get("dropout", 0.1)),
    )


def _obs_image(obs, view):
    frame = getattr(obs, VIEW_TO_ATTR[view])
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        if np.nanmax(frame) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def preprocess_image_array(frame, image_size: int, crop_size: int, normalize: bool = True):
    img = Image.fromarray(frame).convert("RGB")
    if image_size > 0:
        img = img.resize((image_size, image_size), Image.BILINEAR)
    if crop_size and crop_size > 0 and crop_size < img.width:
        left = (img.width - crop_size) // 2
        top = (img.height - crop_size) // 2
        img = img.crop((left, top, left + crop_size, top + crop_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if normalize:
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1)).astype(np.float32)


class MultiViewDpPolicyRunner:
    def __init__(
        self,
        policy_dir,
        checkpoint="latest",
        device="cuda",
        sample_steps=100,
        use_ema=True,
        amp=True,
        amp_dtype="bfloat16",
    ):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        ckpt_path = choose_checkpoint(Path(policy_dir), checkpoint)
        self.checkpoint = torch_load(ckpt_path, map_location="cpu")
        self.cfg = self.checkpoint["config"]
        self.texts = list(self.checkpoint["texts"])
        self.text_to_idx = {text: i for i, text in enumerate(self.texts)}
        self.text_features = torch.as_tensor(self.checkpoint["text_features"], dtype=torch.float32, device=self.device)
        self.action_mean = np.asarray(self.checkpoint["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(self.checkpoint["action_std"], dtype=np.float32)
        self.proprio_mean = np.asarray(self.checkpoint["proprio_mean"], dtype=np.float32)
        self.proprio_std = np.maximum(np.asarray(self.checkpoint["proprio_std"], dtype=np.float32), 1e-4)
        self.model = build_model_from_config(self.cfg, self.text_features.shape[-1]).to(self.device)
        state = self.checkpoint.get("ema_model") if use_ema and self.checkpoint.get("ema_model") is not None else self.checkpoint["model"]
        self.model.load_state_dict(state)
        self.model.eval()
        self.obs_horizon = int(self.cfg.get("obs_horizon", 2))
        self.action_horizon = int(self.cfg.get("action_horizon", 1))
        self.action_dim = int(self.cfg.get("action_dim", 7))
        self.view_names = tuple(str(self.cfg.get("view_names", "front,left_shoulder,right_shoulder")).split(","))
        self.image_size = int(self.cfg.get("image_size", 224))
        self.crop_size = int(self.cfg.get("crop_size", self.image_size))
        self.proprio_mode = self.cfg.get("proprio_mode", "ee_rotvec")
        self.diffusion_steps = int(self.cfg.get("diffusion_steps", 100))
        self.sample_steps = int(sample_steps)
        self.amp = bool(amp)
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
        self.history = deque(maxlen=self.obs_horizon)

    def reset(self, obs):
        self.history.clear()
        for _ in range(self.obs_horizon):
            self.history.append(obs)

    def observe(self, obs):
        if not self.history:
            self.reset(obs)
        else:
            self.history.append(obs)

    def _text_token(self, task_instruction):
        text = str(task_instruction).strip()
        if text not in self.text_to_idx:
            known = ", ".join(self.texts[:8])
            raise KeyError(f"Task instruction {text!r} was not in checkpoint text vocab. Known examples: {known}")
        idx = self.text_to_idx[text]
        return self.text_features[idx : idx + 1]

    def _make_batch(self, task_instruction):
        if len(self.history) != self.obs_horizon:
            raise RuntimeError("Policy history is not initialized")
        images = []
        proprio = []
        for obs in self.history:
            view_imgs = [
                preprocess_image_array(_obs_image(obs, view), self.image_size, self.crop_size, normalize=True)
                for view in self.view_names
            ]
            images.append(np.stack(view_imgs, axis=0))
            state = obs_to_proprio(obs, self.proprio_mode)
            proprio.append((state - self.proprio_mean) / self.proprio_std)
        images = torch.from_numpy(np.stack(images, axis=0)[None]).to(self.device, non_blocking=True)
        proprio = torch.from_numpy(np.stack(proprio, axis=0).astype(np.float32)[None]).to(self.device, non_blocking=True)
        text_token = self._text_token(task_instruction)
        return images, proprio, text_token

    @torch.inference_mode()
    def sample_action(self, task_instruction):
        images, proprio, text_token = self._make_batch(task_instruction)
        pred = sample_ddim(
            self.model,
            images,
            proprio,
            text_token,
            action_shape=(1, self.action_horizon, self.action_dim),
            action_mean=self.action_mean,
            action_std=self.action_std,
            train_steps=self.diffusion_steps,
            sample_steps=self.sample_steps,
            amp=self.amp,
            amp_dtype=self.amp_dtype,
        )
        return pred[0, 0].detach().cpu().numpy().astype(np.float32)
