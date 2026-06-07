from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .common import (
    VIEW_NAMES,
    absolute_rpy7_from_obs,
    clean_waypoints,
    filter_manifest_rows,
    image_path_for_frame,
    load_observations,
    obs_to_proprio,
    read_jsonl,
    resolve_episode_dir,
)


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class WaypointSample:
    rgb_episode_dir: Path
    lowdim_episode_dir: Path
    task: str
    variation: str
    episode: str
    split: str
    source_bundle: str
    task_instruction: str
    current_frame_idx: int
    target_frame_idx: int
    segment_idx: int
    obs_frame_indices: List[int]
    action: np.ndarray
    proprio: np.ndarray
    text_idx: int


def build_text_vocab(rows: Sequence[dict]) -> Dict[str, int]:
    text_to_idx = {}
    for row in rows:
        text = str(row.get("task_instruction") or row.get("task") or "").strip()
        if not text:
            text = str(row.get("task", "")).replace("_", " ")
        if text not in text_to_idx:
            text_to_idx[text] = len(text_to_idx)
    return text_to_idx


def text_vocab_list(text_to_idx: Dict[str, int]) -> List[str]:
    out = [None] * len(text_to_idx)
    for text, idx in text_to_idx.items():
        out[int(idx)] = text
    return out


def waypoint_history_indices(points: Sequence[int], segment_idx: int, obs_horizon: int) -> List[int]:
    out = []
    for idx in range(int(segment_idx) - int(obs_horizon) + 1, int(segment_idx) + 1):
        out.append(int(points[max(idx, 0)]))
    return out


class RlbenchHeuristicWaypointDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        rgb_roots: Dict[str, Path],
        split: str,
        lowdim_roots: Optional[Dict[str, Path]] = None,
        tasks: Optional[Sequence[str]] = None,
        obs_horizon: int = 2,
        sample_every_n: int = 10,
        view_names: Sequence[str] = VIEW_NAMES,
        image_size: int = 224,
        crop_size: Optional[int] = None,
        random_crop: bool = False,
        max_episodes: Optional[int] = None,
        max_episodes_per_task: Optional[int] = None,
        proprio_mode: str = "ee_rotvec",
        text_to_idx: Optional[Dict[str, int]] = None,
        validate_image_paths: bool = False,
        loose_pickle: bool = True,
        normalize_images: bool = True,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        self.rgb_roots = {str(k): Path(v).resolve() for k, v in rgb_roots.items()}
        self.lowdim_roots = {str(k): Path(v).resolve() for k, v in (lowdim_roots or rgb_roots).items()}
        self.split = split
        self.obs_horizon = int(obs_horizon)
        self.sample_every_n = int(sample_every_n)
        self.view_names = tuple(view_names)
        self.image_size = int(image_size)
        self.crop_size = int(crop_size) if crop_size is not None else int(image_size)
        self.random_crop = bool(random_crop)
        self.proprio_mode = proprio_mode
        self.normalize_images = bool(normalize_images)
        self.proprio_mean = None
        self.proprio_std = None

        all_rows = read_jsonl(self.manifest_path)
        rows = filter_manifest_rows(
            all_rows,
            split=split,
            tasks=tasks,
            max_episodes=max_episodes,
            max_episodes_per_task=max_episodes_per_task,
        )
        if text_to_idx is None:
            text_to_idx = build_text_vocab(all_rows)
        self.text_to_idx = dict(text_to_idx)
        self.texts = text_vocab_list(self.text_to_idx)
        self.samples = []
        self.episode_rows = rows
        self._build_samples(rows, validate_image_paths=validate_image_paths, loose_pickle=loose_pickle)
        if not self.samples:
            raise RuntimeError(f"No samples built from {self.manifest_path} split={split}")
        self.actions = np.stack([s.action for s in self.samples]).astype(np.float32)
        self.action_mask = np.ones((len(self.samples), 1), dtype=np.float32)

    def _build_samples(self, rows, validate_image_paths: bool, loose_pickle: bool) -> None:
        for row in rows:
            rgb_episode_dir = resolve_episode_dir(row, self.rgb_roots, root_name="RGB root")
            lowdim_episode_dir = resolve_episode_dir(row, self.lowdim_roots, root_name="low-dim root")
            observations = load_observations(lowdim_episode_dir, loose=loose_pickle)
            num_frames = min(int(row.get("num_frames", len(observations))), len(observations))
            waypoints = clean_waypoints(row, num_frames=num_frames)
            points = [0] + [p for p in waypoints if 0 < int(p) < num_frames]
            points = sorted(set(int(p) for p in points))
            if len(points) < 2:
                continue
            text = str(row.get("task_instruction") or row.get("task") or "").strip()
            if not text:
                text = str(row.get("task", "")).replace("_", " ")
            if text not in self.text_to_idx:
                raise KeyError(f"Text {text!r} is missing from text vocabulary")
            text_idx = int(self.text_to_idx[text])

            for segment_idx in range(len(points) - 1):
                start = int(points[segment_idx])
                target = int(points[segment_idx + 1])
                if target <= start:
                    continue
                if self.sample_every_n <= 0:
                    current_frames = [start]
                else:
                    current_frames = list(range(start, target, self.sample_every_n))
                    if start not in current_frames:
                        current_frames.insert(0, start)
                current_frames = sorted(set(int(x) for x in current_frames if start <= int(x) < target))
                for current in current_frames:
                    if self.sample_every_n <= 0:
                        obs_frames = waypoint_history_indices(points, segment_idx, self.obs_horizon)
                    else:
                        obs_frames = [max(0, i) for i in range(current - self.obs_horizon + 1, current + 1)]
                    proprio = np.stack(
                        [obs_to_proprio(observations[min(f, num_frames - 1)], self.proprio_mode) for f in obs_frames],
                        axis=0,
                    ).astype(np.float32)
                    action = absolute_rpy7_from_obs(observations[target])
                    if validate_image_paths:
                        for frame_idx in obs_frames:
                            for view in self.view_names:
                                path = image_path_for_frame(rgb_episode_dir, view, frame_idx)
                                if not path.exists():
                                    raise FileNotFoundError(path)
                    self.samples.append(
                        WaypointSample(
                            rgb_episode_dir=rgb_episode_dir,
                            lowdim_episode_dir=lowdim_episode_dir,
                            task=str(row["task"]),
                            variation=str(row["variation"]),
                            episode=str(row["episode"]),
                            split=str(row["split"]),
                            source_bundle=str(row.get("source_bundle", "")),
                            task_instruction=text,
                            current_frame_idx=int(current),
                            target_frame_idx=int(target),
                            segment_idx=int(segment_idx + 1),
                            obs_frame_indices=obs_frames,
                            action=action,
                            proprio=proprio,
                            text_idx=text_idx,
                        )
                    )

    def set_proprio_stats(self, mean, std):
        self.proprio_mean = np.asarray(mean, dtype=np.float32)
        self.proprio_std = np.maximum(np.asarray(std, dtype=np.float32), 1e-4)

    def compute_proprio_stats(self):
        proprio = np.concatenate([s.proprio for s in self.samples], axis=0)
        mean = proprio.mean(axis=0).astype(np.float32)
        std = np.maximum(proprio.std(axis=0).astype(np.float32), 1e-4)
        return mean, std

    def compute_action_stats(self):
        actions = np.stack([s.action for s in self.samples], axis=0)
        mean = actions.mean(axis=0).astype(np.float32)
        std = np.maximum(actions.std(axis=0).astype(np.float32), 1e-4)
        return mean, std

    def __len__(self):
        return len(self.samples)

    def _crop_params(self, width: int, height: int):
        crop = int(self.crop_size)
        if crop <= 0 or (crop >= width and crop >= height):
            return 0, 0, width, height
        if self.random_crop:
            left = np.random.randint(0, max(width - crop + 1, 1))
            top = np.random.randint(0, max(height - crop + 1, 1))
        else:
            left = max((width - crop) // 2, 0)
            top = max((height - crop) // 2, 0)
        return int(left), int(top), int(min(crop, width)), int(min(crop, height))

    def _load_image(self, path: Path, crop_params):
        img = Image.open(path).convert("RGB")
        if self.image_size > 0:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        left, top, width, height = crop_params
        if width > 0 and height > 0 and (width != img.width or height != img.height):
            img = img.crop((left, top, left + width, top + height))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.normalize_images:
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = np.transpose(arr, (2, 0, 1))
        return arr.astype(np.float32)

    def __getitem__(self, index):
        sample = self.samples[int(index)]
        effective_size = self.image_size if self.image_size > 0 else 256
        crop_params = self._crop_params(effective_size, effective_size)
        image_seq = []
        for frame_idx in sample.obs_frame_indices:
            view_imgs = []
            for view in self.view_names:
                view_imgs.append(
                    self._load_image(image_path_for_frame(sample.rgb_episode_dir, view, frame_idx), crop_params)
                )
            image_seq.append(np.stack(view_imgs, axis=0))
        images = np.stack(image_seq, axis=0)

        proprio = sample.proprio
        if self.proprio_mean is not None:
            proprio = (proprio - self.proprio_mean[None, :]) / self.proprio_std[None, :]

        return {
            "image": torch.from_numpy(images),
            "proprio": torch.from_numpy(proprio.astype(np.float32)),
            "text_idx": torch.tensor(sample.text_idx, dtype=torch.long),
            "action": torch.from_numpy(sample.action[None, :].astype(np.float32)),
            "mask": torch.ones((1,), dtype=torch.float32),
            "task": sample.task,
            "episode": sample.episode,
            "current_frame_idx": torch.tensor(sample.current_frame_idx, dtype=torch.long),
            "target_frame_idx": torch.tensor(sample.target_frame_idx, dtype=torch.long),
        }


def collate_waypoint_batch(batch):
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "proprio": torch.stack([b["proprio"] for b in batch], dim=0),
        "text_idx": torch.stack([b["text_idx"] for b in batch], dim=0),
        "action": torch.stack([b["action"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
        "current_frame_idx": torch.stack([b["current_frame_idx"] for b in batch], dim=0),
        "target_frame_idx": torch.stack([b["target_frame_idx"] for b in batch], dim=0),
    }
