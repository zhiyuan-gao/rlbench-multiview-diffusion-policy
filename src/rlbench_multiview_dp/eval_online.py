import argparse
import json
import os
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import (
    VIEW_TO_ATTR,
    absolute_rotvec7_to_rlbench,
    clean_waypoints,
    filter_manifest_rows,
    load_demo,
    parse_root_mapping,
    read_jsonl,
    resolve_episode_dir,
    write_json,
)
from .inference import MultiViewDpPolicyRunner


def camera_config(enabled, image_size):
    from rlbench.observation_config import CameraConfig

    return CameraConfig(rgb=enabled, depth=False, point_cloud=False, mask=False, image_size=(image_size, image_size))


def make_obs_config(enabled_views, image_size):
    from rlbench.observation_config import ObservationConfig

    views = set(enabled_views)
    return ObservationConfig(
        left_shoulder_camera=camera_config("left_shoulder" in views, image_size),
        right_shoulder_camera=camera_config("right_shoulder" in views, image_size),
        overhead_camera=camera_config("overhead" in views, image_size),
        wrist_camera=camera_config("wrist" in views, image_size),
        front_camera=camera_config("front" in views, image_size),
        joint_velocities=True,
        joint_positions=True,
        joint_forces=False,
        gripper_open=True,
        gripper_pose=True,
        gripper_matrix=False,
        gripper_joint_positions=True,
        gripper_touch_forces=False,
        task_low_dim_state=True,
    )


def make_action_mode(arm_mode, collision_checking, absolute_mode=True):
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK, EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete

    if arm_mode == "planning":
        arm = EndEffectorPoseViaPlanning(absolute_mode=absolute_mode, collision_checking=collision_checking)
    elif arm_mode == "ik":
        arm = EndEffectorPoseViaIK(absolute_mode=absolute_mode, collision_checking=collision_checking)
    else:
        raise ValueError(f"Unknown arm_mode={arm_mode}")
    return MoveArmThenGripper(arm, Discrete())


def parse_rgb_roots(args):
    return parse_root_mapping(args.rgb_root_200, args.rgb_root_400, args.rgb_root, root_name="RGB root", required=False)


def parse_lowdim_roots(args, fallback_roots=None):
    roots = parse_root_mapping(
        args.lowdim_root_200,
        args.lowdim_root_400,
        args.lowdim_root,
        root_name="low-dim root",
        required=False,
    )
    roots = roots or dict(fallback_roots or {})
    if not roots:
        raise ValueError("Online eval needs low-dim roots for low_dim_obs.pkl reset demos.")
    return roots


def observation_images(obs, views):
    images = {}
    for view in views:
        frame = getattr(obs, VIEW_TO_ATTR[view], None)
        if frame is None:
            continue
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            if np.nanmax(frame) <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        images[view] = frame
    return images


def observation_frame(obs, views):
    images = observation_images(obs, views)
    frames = [images[v] for v in views if v in images]
    if not frames:
        return None
    return np.concatenate(frames, axis=1)


def summarize(results):
    per_task = defaultdict(lambda: {"episodes": 0, "successes": 0, "invalid_actions": 0, "executed_actions": 0})
    for result in results:
        item = per_task[result["task"]]
        item["episodes"] += 1
        item["successes"] += int(result["success"])
        item["invalid_actions"] += int(result["invalid_actions"])
        item["executed_actions"] += int(result["executed_actions"])
    for item in per_task.values():
        item["success_rate"] = item["successes"] / max(item["episodes"], 1)
    successes = sum(int(r["success"]) for r in results)
    return {
        "episodes": len(results),
        "successes": successes,
        "success_rate": successes / max(len(results), 1),
        "invalid_actions": int(sum(r["invalid_actions"] for r in results)),
        "executed_actions": int(sum(r["executed_actions"] for r in results)),
        "per_task": dict(sorted(per_task.items())),
    }


def run_episode(task_env, policy, row, lowdim_episode_dir, args, out_dir, invalid_action_error):
    demo = load_demo(Path(lowdim_episode_dir) / "low_dim_obs.pkl", loose=False)
    descriptions, obs = task_env.reset_to_demo(demo)
    task_instruction = str(row.get("task_instruction") or (descriptions[0] if descriptions else row["task"]))
    policy.reset(obs)
    waypoints = clean_waypoints(row, int(row.get("num_frames", 0)) or None)
    max_steps = int(args.max_policy_steps) if args.max_policy_steps > 0 else len(waypoints) + int(args.extra_policy_steps)

    writer = None
    mp4_path = None
    if args.record_video:
        import imageio.v2 as imageio

        safe_id = f"{row['task']}__{row['variation']}__{row['episode']}__{row['split']}"
        mp4_path = out_dir / "videos" / f"{safe_id}.mp4"
        mp4_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(mp4_path), fps=args.video_fps)
        frame = observation_frame(obs, args.record_view)
        if frame is not None:
            writer.append_data(frame)

    success = False
    terminate = False
    last_reward = 0.0
    invalid_actions = 0
    executed_actions = 0
    error = None
    step_logs = []
    try:
        for step in range(max_steps):
            action = policy.sample_action(task_instruction)
            rlbench_action = absolute_rotvec7_to_rlbench(action)
            log_item = {
                "policy_step": int(step),
                "action": action.tolist(),
                "rlbench_action": rlbench_action.tolist(),
            }
            try:
                obs, reward, terminate = task_env.step(rlbench_action)
                policy.observe(obs)
                last_reward = float(reward)
                executed_actions += 1
                success = bool(reward >= 1.0)
                log_item.update({"status": "ok", "reward": float(reward), "terminate": bool(terminate)})
            except invalid_action_error as exc:
                invalid_actions += 1
                error = f"{type(exc).__name__}: {exc}"
                log_item.update({"status": "invalid_action", "reward": float(last_reward), "terminate": bool(terminate), "error": error})
                if not args.continue_after_invalid:
                    step_logs.append(log_item)
                    break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                log_item.update({"status": "error", "reward": float(last_reward), "terminate": True, "error": error})
                if args.verbose_errors:
                    traceback.print_exc()
                step_logs.append(log_item)
                break
            step_logs.append(log_item)
            if writer is not None:
                frame = observation_frame(obs, args.record_view)
                if frame is not None:
                    writer.append_data(frame)
            if success or terminate:
                break
    finally:
        if writer is not None:
            writer.close()

    return {
        "task": row["task"],
        "variation": row["variation"],
        "episode": row["episode"],
        "split_unique_id": row.get("split_unique_id"),
        "source_bundle": row.get("source_bundle"),
        "lowdim_episode_dir": str(lowdim_episode_dir),
        "task_instruction": task_instruction,
        "full_task_heuristic_waypoints": waypoints,
        "success": bool(success),
        "terminate": bool(terminate),
        "last_reward": float(last_reward),
        "executed_actions": int(executed_actions),
        "invalid_actions": int(invalid_actions),
        "max_policy_steps": int(max_steps),
        "error": error,
        "video_path": str(mp4_path) if mp4_path is not None else None,
        "step_logs": step_logs if args.save_step_logs else [],
    }


def run_eval(args):
    from rlbench.backend.exceptions import InvalidActionError
    from rlbench.environment import Environment
    from rlbench.utils import name_to_task_class

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_roots = parse_rgb_roots(args)
    lowdim_roots = parse_lowdim_roots(args, fallback_roots=rgb_roots)
    rows = filter_manifest_rows(
        read_jsonl(args.manifest),
        split=args.split,
        tasks=args.task,
        max_episodes=args.max_episodes,
        max_episodes_per_task=args.max_episodes_per_task,
    )
    policy = MultiViewDpPolicyRunner(
        args.policy_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        sample_steps=args.sample_steps,
        use_ema=args.use_ema,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
    )
    enabled_views = set(policy.view_names)
    if args.record_video:
        enabled_views.update(args.record_view)
    camera_image_size = int(args.image_size) if int(args.image_size) > 0 else int(policy.image_size)
    obs_config = make_obs_config(enabled_views, camera_image_size)
    action_mode = make_action_mode(args.arm_mode, args.collision_checking, absolute_mode=True)
    env = Environment(
        action_mode,
        dataset_root="",
        obs_config=obs_config,
        headless=(not bool(args.record_video)) or bool(args.record_video_headless),
        static_positions=False,
        arm_max_velocity=args.arm_max_velocity,
        arm_max_acceleration=args.arm_max_acceleration,
    )

    results = []
    current_task = None
    task_env = None
    try:
        env.launch()
        for episode_i, row in enumerate(rows, start=1):
            task_name = row["task"]
            if current_task != task_name:
                task_env = env.get_task(name_to_task_class(task_name))
                current_task = task_name
            lowdim_episode_dir = resolve_episode_dir(row, lowdim_roots, root_name="low-dim root")
            result = run_episode(task_env, policy, row, lowdim_episode_dir, args, out_dir, InvalidActionError)
            results.append(result)
            status = "SUCCESS" if result["success"] else "FAIL"
            print(
                f"episode={episode_i}/{len(rows)} task={task_name} {row['variation']}/{row['episode']} "
                f"{status} actions={result['executed_actions']} invalid={result['invalid_actions']} error={result['error']}",
                flush=True,
            )
            if args.write_every and episode_i % args.write_every == 0:
                write_json(out_dir / "results.json", results)
                write_json(out_dir / "summary.json", summarize(results))
    finally:
        env.shutdown()

    summary = summarize(results)
    summary.update(
        {
            "manifest": str(Path(args.manifest).resolve()),
            "policy_dir": str(Path(args.policy_dir).resolve()),
            "checkpoint": args.checkpoint,
            "split": args.split,
            "rgb_roots": {k: str(v) for k, v in rgb_roots.items()},
            "lowdim_roots": {k: str(v) for k, v in lowdim_roots.items()},
            "sample_steps": int(args.sample_steps),
            "arm_mode": args.arm_mode,
            "action_format": "absolute_rotvec7",
            "execute_horizon": 1,
            "camera_image_size": int(camera_image_size),
        }
    )
    write_json(out_dir / "results.json", results)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rgb-root-200", default=None)
    parser.add_argument("--rgb-root-400", default=None)
    parser.add_argument("--rgb-root", action="append", default=None)
    parser.add_argument("--lowdim-root-200", default=None)
    parser.add_argument("--lowdim-root-400", default=None)
    parser.add_argument("--lowdim-root", action="append", default=None)
    parser.add_argument("--policy-dir", required=True)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument("--max-policy-steps", type=int, default=0)
    parser.add_argument("--extra-policy-steps", type=int, default=2)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--arm-mode", choices=("planning", "ik"), default="planning")
    parser.add_argument("--collision-checking", action="store_true")
    parser.add_argument("--arm-max-velocity", type=float, default=1.0)
    parser.add_argument("--arm-max-acceleration", type=float, default=4.0)
    parser.add_argument("--image-size", type=int, default=0, help="RLBench camera size. 0 uses the checkpoint image_size.")
    parser.add_argument("--continue-after-invalid", action="store_true")
    parser.add_argument("--verbose-errors", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--record-video-headless", action="store_true")
    parser.add_argument("--record-view", choices=sorted(VIEW_TO_ATTR), action="append", default=["front"])
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--save-step-logs", action="store_true")
    parser.add_argument("--write-every", type=int, default=1)
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
