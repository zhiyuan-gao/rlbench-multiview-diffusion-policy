import json
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


VIEW_NAMES = ("front", "left_shoulder", "right_shoulder")
VIEW_TO_ATTR = {
    "front": "front_rgb",
    "left_shoulder": "left_shoulder_rgb",
    "right_shoulder": "right_shoulder_rgb",
    "overhead": "overhead_rgb",
    "wrist": "wrist_rgb",
}


class DummyRlbenchObject:
    """Loose fallback for reading low_dim_obs.pkl without RLBench imports."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


class LooseUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("rlbench") or module.startswith("pyrep"):
            return DummyRlbenchObject
        return super().find_class(module, name)


@dataclass(frozen=True)
class EpisodeRecord:
    row: dict
    episode_dir: Path


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_tasks(tasks: Optional[Sequence[str]]) -> Optional[set]:
    if not tasks:
        return None
    out = set()
    for item in tasks:
        for task in str(item).replace(",", " ").split():
            task = task.strip()
            if task:
                out.add(task)
    return out or None


def parse_root_mapping(root_200=None, root_400=None, extra_roots=None, root_name: str = "root", required: bool = True):
    roots = {}
    if root_200:
        roots["all200"] = Path(root_200).resolve()
    if root_400:
        roots["all400"] = Path(root_400).resolve()
    for item in extra_roots or []:
        if "=" not in item:
            raise ValueError(f"{root_name} entries must have form source_bundle=/path")
        key, value = item.split("=", 1)
        roots[key.strip()] = Path(value).resolve()
    if required and not roots:
        raise ValueError(f"At least one {root_name} is required.")
    return roots


def source_bundle_to_root(row: dict, roots: Dict[str, Path], root_name: str = "root") -> Path:
    bundle = str(row.get("source_bundle", "all200"))
    if bundle not in roots:
        known = ", ".join(sorted(roots))
        raise KeyError(f"Manifest row source_bundle={bundle!r} has no {root_name}. Known: {known}")
    return Path(roots[bundle]).resolve()


def resolve_episode_dir(row: dict, roots: Dict[str, Path], root_name: str = "root") -> Path:
    root = source_bundle_to_root(row, roots, root_name=root_name)
    rel = row.get("rgb_episode_relpath")
    if rel:
        return root / rel
    task = row["task"]
    variation = row["variation"]
    episode = row["episode"]
    return root / task / variation / "episodes" / episode


def filter_manifest_rows(
    rows: Sequence[dict],
    split: Optional[str],
    tasks: Optional[Sequence[str]],
    max_episodes: Optional[int] = None,
    max_episodes_per_task: Optional[int] = None,
) -> List[dict]:
    task_filter = parse_tasks(tasks)
    selected = []
    per_task = defaultdict(int)
    for row in rows:
        task = row.get("task")
        if split is not None and row.get("split") != split:
            continue
        if task_filter is not None and task not in task_filter:
            continue
        if max_episodes is not None and len(selected) >= int(max_episodes):
            break
        if max_episodes_per_task is not None and per_task[task] >= int(max_episodes_per_task):
            continue
        selected.append(row)
        per_task[task] += 1
    if not selected:
        raise RuntimeError("No manifest rows matched the requested filters.")
    return selected


def load_demo(path: Path, loose: bool = False):
    with Path(path).open("rb") as f:
        if loose:
            return LooseUnpickler(f).load()
        return pickle.load(f)


def load_observations(episode_dir: Path, loose: bool = True):
    demo = load_demo(Path(episode_dir) / "low_dim_obs.pkl", loose=loose)
    observations = getattr(demo, "_observations", demo)
    if not isinstance(observations, (list, tuple)):
        raise ValueError(f"{episode_dir}/low_dim_obs.pkl did not contain a Demo-like observation list")
    return observations


def normalize_quat(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)


def quat_to_rpy(q):
    q = normalize_quat(q)
    x, y, z, w = np.moveaxis(q, -1, 0)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(pitch_arg)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=-1)


def rpy_to_quat(rpy):
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return normalize_quat(
        np.asarray(
            [
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy,
            ],
            dtype=np.float64,
        )
    )


def quat_to_rotvec(q):
    q = normalize_quat(q)
    q = np.where(q[..., 3:4] < 0.0, -q, q)
    xyz = q[..., :3]
    w = np.clip(q[..., 3], -1.0, 1.0)
    sin_half = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, w)
    scale = np.full_like(angle, 2.0)
    np.divide(angle, sin_half, out=scale, where=sin_half > 1e-8)
    return xyz * scale[..., None]


def rotvec_to_quat(rotvec):
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = rotvec / angle
    half = 0.5 * angle
    xyz = axis * math.sin(half)
    return normalize_quat(np.concatenate([xyz, [math.cos(half)]]))


def gripper_open_value(obs, threshold: float = 0.95) -> float:
    joint_positions = getattr(obs, "gripper_joint_positions", None)
    if joint_positions is None:
        return float(getattr(obs, "gripper_open", 0.0))
    joint_positions = np.asarray(joint_positions, dtype=np.float32)
    if joint_positions.size == 0:
        return float(getattr(obs, "gripper_open", 0.0))
    return 1.0 if float(joint_positions[0]) / 0.04 > float(threshold) else 0.0


def obs_to_proprio(obs, mode: str = "ee_rotvec") -> np.ndarray:
    ee_pose = np.asarray(obs.gripper_pose, dtype=np.float32)
    if ee_pose.shape != (7,):
        raise ValueError(f"Expected obs.gripper_pose shape (7,), got {ee_pose.shape}")
    gripper = np.asarray([gripper_open_value(obs)], dtype=np.float32)
    if mode == "ee_rotvec":
        return np.concatenate([ee_pose[:3], quat_to_rotvec(ee_pose[3:7]).astype(np.float32), gripper], axis=0)
    if mode == "ee_rpy":
        return np.concatenate([ee_pose[:3], quat_to_rpy(ee_pose[3:7]).astype(np.float32), gripper], axis=0)
    raise ValueError(f"Unsupported proprio mode: {mode}")


def absolute_rotvec7_from_obs(obs) -> np.ndarray:
    ee_pose = np.asarray(obs.gripper_pose, dtype=np.float32)
    rotvec = quat_to_rotvec(ee_pose[3:7]).astype(np.float32)
    return np.concatenate([ee_pose[:3], rotvec, [gripper_open_value(obs)]], axis=0).astype(np.float32)


def absolute_rotvec7_to_rlbench(action_7d) -> np.ndarray:
    action = np.asarray(action_7d, dtype=np.float64)
    quat = rotvec_to_quat(action[3:6])
    return np.concatenate([action[:3], quat, [float(action[6])]]).astype(np.float64)


def image_path_for_frame(episode_dir: Path, view: str, frame_idx: int) -> Path:
    return Path(episode_dir) / f"{view}_rgb" / f"{int(frame_idx)}.png"


def clean_waypoints(row: dict, num_frames: Optional[int] = None) -> List[int]:
    raw = row.get("full_task_heuristic_waypoints")
    if raw is None:
        raw = row.get("clean_keypoints") or row.get("keypoints") or row.get("event_all_grouping_keypoints")
    if raw is None:
        raise KeyError("Manifest row has no full_task_heuristic_waypoints / clean_keypoints / keypoints field")
    max_frame = None if num_frames is None else max(int(num_frames) - 1, 0)
    points = []
    for point in raw:
        point = int(point)
        if point <= 0:
            continue
        if max_frame is not None:
            point = min(max(point, 0), max_frame)
        points.append(point)
    return sorted(set(points))


def torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def choose_checkpoint(policy_dir: Path, checkpoint: str) -> Path:
    policy_dir = Path(policy_dir)
    if checkpoint in ("latest", "best", "last"):
        path = policy_dir / f"{checkpoint}.pt"
    else:
        path = Path(checkpoint)
        if not path.is_absolute() and not path.exists():
            path = policy_dir / checkpoint
    if not path.exists():
        raise FileNotFoundError(path)
    return path
