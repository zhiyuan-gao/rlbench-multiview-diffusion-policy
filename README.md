# RLBench Multi-View Diffusion Policy Baseline

这是一个给 RLBench selected10 heuristic-waypoint imitation 使用的 robomimic 风格 multi-view Diffusion Policy baseline。

这个 repo 和 Wan/VPP 代码是分开的。它不使用 Wan hidden tokens，不使用视频模型特征，不使用 DINO/DINOv2/CLIP 图像特征 cache，也不使用 event/subtask boundary。模型使用：

- `front`、`left_shoulder`、`right_shoulder` 三视角 raw RGB
- 默认 `observation_horizon=2`
- ResNet18Conv/ResNet34Conv 风格视觉 encoder + SpatialSoftmax
- frozen CLIP text embedding 处理 `task_instruction`
- end-effector rotvec pose + gripper state proprio
- ConditionalUnet1D 风格 diffusion head
- 稀疏动作目标：预测下一个 full-task heuristic waypoint

动作格式是 `absolute_rotvec7`：`x, y, z, rotvec_x, rotvec_y, rotvec_z, gripper_open`。训练时 `action_horizon=1`，online eval 时每次预测一个 waypoint，并在执行前转换成 RLBench 需要的 quaternion action。

## 数据格式

repo 内置了 selected10 train100/val25/test25 manifest：

```text
manifests/selected10_fulltask_heuristic_waypoints_train100_val25_test25_from_train450_stratified_20260606.jsonl
```

每一行是一个 episode，训练代码需要这些字段：

- `split`: `train`、`val` 或 `test`
- `source_bundle`: `all200` 或 `all400`
- `rgb_episode_relpath`: 例如 `meat_off_grill/variation0/episodes/episode0`
- `task`、`variation`、`episode`
- `task_instruction`
- `full_task_heuristic_waypoints`
- `num_frames`

RGB 路径会这样重建：

```text
{rgb_root_for_source_bundle}/{rgb_episode_relpath}/{view}_rgb/{frame}.png
```

low-dim observation 会从这里读取：

```text
{lowdim_root_for_source_bundle}/{rgb_episode_relpath}/low_dim_obs.pkl
```

所以 HPC 上需要分别提供 RGB root 和 low-dim metadata root。它们的 episode 层级相同，但文件类型不同：

```text
${RGB_ROOT_200}/meat_off_grill/variation0/episodes/episode0/front_rgb/0.png
${LOWDIM_ROOT_200}/meat_off_grill/variation0/episodes/episode0/low_dim_obs.pkl
```

`source_bundle=all200` 的 episode 使用 `RGB_ROOT_200`，`source_bundle=all400` 的 episode 使用 `RGB_ROOT_400`。
对应的 `low_dim_obs.pkl` 使用 `LOWDIM_ROOT_200` 和 `LOWDIM_ROOT_400`。

## HPC 环境要求

如果 HPC 已经有 RLBench 环境和代码，这个 repo 可以直接安装运行。需要环境里至少有：

- PyTorch / torchvision
- RLBench / PyRep
- CoppeliaSim
- `numpy`、`pillow`、`tqdm`
- `transformers`，用于 CLIP text encoder
- 两个 RGB root：`all200` 和 `all400`
- 两个 low-dim metadata root：`all200` 和 `all400`

online eval 还需要 RLBench/PyRep 可 import，并且 CoppeliaSim 环境变量已经在 job 里生效，例如：

```bash
export COPPELIASIM_ROOT=/path/to/CoppeliaSim
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export PYTHONPATH=/path/to/RLBench:/path/to/PyRep:${PYTHONPATH:-}
```

## 安装

在 HPC 上 clone 这个 repo 后进入目录：

```bash
cd rlbench_multiview_dp_baseline_20260606
pip install -e .
pip install -r requirements.txt
```

如果 HPC 环境里的 PyTorch/RLBench 版本不想被 pip 改动，可以更保守地安装：

```bash
pip install -e . --no-deps
pip install transformers pillow tqdm numpy
```

脚本内部也会把 repo 的 `src/` 加到 `PYTHONPATH`，所以即使只从源码目录运行也可以找到 `rlbench_multiview_dp` 包。

## 先做数据 Smoke Test

正式提交训练作业前，建议先检查 manifest、`RGB_ROOT_200/RGB_ROOT_400`、`LOWDIM_ROOT_200/LOWDIM_ROOT_400`、RGB 图片和 `low_dim_obs.pkl` 是否对齐：

```bash
RGB_ROOT_200=/path/to/rgb_root_200 \
RGB_ROOT_400=/path/to/rgb_root_400 \
LOWDIM_ROOT_200=/path/to/rlbench_local200_nonimage_metadata_20260516 \
LOWDIM_ROOT_400=/path/to/lowdim_root_400 \
bash scripts/smoke_dataset.sh
```

这个命令只会读少量 `low_dim_obs.pkl` 和三视角 RGB，打印 sample shape、当前帧、目标 waypoint 和 `absolute_rotvec7` 动作。它不会启动训练，也不会启动 RLBench/CoppeliaSim。

## 训练

单卡训练：

```bash
RGB_ROOT_200=/path/to/rgb_root_200 \
RGB_ROOT_400=/path/to/rgb_root_400 \
LOWDIM_ROOT_200=/path/to/lowdim_root_200 \
LOWDIM_ROOT_400=/path/to/lowdim_root_400 \
RUN_ROOT=/path/to/runs/selected10_rgb_resnet18conv_clip_dp_h1 \
bash scripts/train_selected10_rgb_resnet18conv_dp_h1.sh
```

多卡训练：

```bash
NUM_GPUS=8 \
RGB_ROOT_200=/path/to/rgb_root_200 \
RGB_ROOT_400=/path/to/rgb_root_400 \
LOWDIM_ROOT_200=/path/to/lowdim_root_200 \
LOWDIM_ROOT_400=/path/to/lowdim_root_400 \
RUN_ROOT=/path/to/runs/selected10_rgb_resnet18conv_clip_dp_h1 \
bash scripts/train_selected10_rgb_resnet18conv_dp_h1.sh
```

训练默认使用：

- `split=train`
- `obs_horizon=2`
- `sample_every_n=0`，每个 segment 只取一个 waypoint-level sample
- 默认 `To=2` 使用最近两个 decision observations，例如 `[previous waypoint obs, current waypoint obs]`
- `front,left_shoulder,right_shoulder`
- `image_size=256`
- `resnet18`
- ImageNet-pretrained visual backbone
- 三个视角各自一个 ResNet visual encoder，不共享权重
- `openai/clip-vit-large-patch14`
- `global_batch_size=256`，脚本会按 `NUM_GPUS` 自动换算每卡 batch size
- action/diffusion lr `1e-4`
- visual encoder lr `1e-5`
- weight decay `1e-6`
- warmup steps `1000`
- cosine lr schedule
- train denoising steps `100`
- 默认 train steps `40000`。当前 manifest 有约 `4643` 个 train waypoint samples，在 global batch `256` 下约等于 `2200` effective epochs，接近 2000 effective epochs 的推荐区间
- checkpoint interval `5000`
- `action_horizon=1`

常用覆盖参数可以通过环境变量传入，例如：

```bash
NUM_GPUS=8 \
GLOBAL_BATCH_SIZE=256 \
MAX_STEPS=40000 \
VISUAL_BACKBONE=resnet34 \
RUN_ROOT=/path/to/run \
RGB_ROOT_200=/path/to/rgb_root_200 \
RGB_ROOT_400=/path/to/rgb_root_400 \
LOWDIM_ROOT_200=/path/to/lowdim_root_200 \
LOWDIM_ROOT_400=/path/to/lowdim_root_400 \
bash scripts/train_selected10_rgb_resnet18conv_dp_h1.sh
```

训练脚本会保存：

```text
latest.pt
last.pt
step_5000.pt
step_10000.pt
...
```

推荐训练结束后对候选 `step_*.pt` checkpoints 跑完整 validation rollouts，再用 validation success 选最终 test checkpoint。

如果只想跑某些 task，可以传 `TASKS`：

```bash
TASKS="meat_off_grill open_drawer" \
RGB_ROOT_200=/path/to/rgb_root_200 \
RGB_ROOT_400=/path/to/rgb_root_400 \
LOWDIM_ROOT_200=/path/to/lowdim_root_200 \
LOWDIM_ROOT_400=/path/to/lowdim_root_400 \
bash scripts/train_selected10_rgb_resnet18conv_dp_h1.sh
```

## CLIP 缓存

默认训练时会通过 HuggingFace 加载：

```text
openai/clip-vit-large-patch14
```

如果 compute node 没有外网，需要提前在登录节点或镜像里缓存 CLIP，并设置好 `HF_HOME` 或 `TRANSFORMERS_CACHE`。离线运行时可以加：

```bash
CLIP_LOCAL_FILES_ONLY=1 \
RGB_ROOT_200=/path/to/rgb_root_200 \
RGB_ROOT_400=/path/to/rgb_root_400 \
LOWDIM_ROOT_200=/path/to/lowdim_root_200 \
LOWDIM_ROOT_400=/path/to/lowdim_root_400 \
bash scripts/train_selected10_rgb_resnet18conv_dp_h1.sh
```

## Online Eval

online eval 会 reset 到 manifest 中对应 demo 的 `low_dim_obs.pkl`，保留最近两个 live observations，预测一个 `absolute_rotvec7` waypoint，转换成 RLBench quaternion action 后执行一个 planning/IK action。

主结果默认使用 `SAMPLE_STEPS=100` denoising steps。若评估太慢，可以额外报告 `SAMPLE_STEPS=16` 或 `32` 的加速版本，但不建议把它作为主表默认。

验证集评估：

```bash
RGB_ROOT_200=/path/to/rgb_root_200 \
RGB_ROOT_400=/path/to/rgb_root_400 \
LOWDIM_ROOT_200=/path/to/lowdim_root_200 \
LOWDIM_ROOT_400=/path/to/lowdim_root_400 \
RUN_ROOT=/path/to/runs/selected10_rgb_resnet18conv_clip_dp_h1 \
SPLIT=val \
bash scripts/eval_selected10_rgb_resnet18conv_dp.sh
```

测试集评估：

```bash
RGB_ROOT_200=/path/to/rgb_root_200 \
RGB_ROOT_400=/path/to/rgb_root_400 \
LOWDIM_ROOT_200=/path/to/lowdim_root_200 \
LOWDIM_ROOT_400=/path/to/lowdim_root_400 \
RUN_ROOT=/path/to/runs/selected10_rgb_resnet18conv_clip_dp_h1 \
SPLIT=test \
bash scripts/eval_selected10_rgb_resnet18conv_dp.sh
```

常用 eval 覆盖参数：

```bash
CHECKPOINT=step_50000.pt \
SAMPLE_STEPS=100 \
MAX_EPISODES_PER_TASK=25 \
ARM_MODE=planning \
RGB_ROOT_200=/path/to/rgb_root_200 \
RGB_ROOT_400=/path/to/rgb_root_400 \
LOWDIM_ROOT_200=/path/to/lowdim_root_200 \
LOWDIM_ROOT_400=/path/to/lowdim_root_400 \
RUN_ROOT=/path/to/run \
bash scripts/eval_selected10_rgb_resnet18conv_dp.sh
```

eval 输出会写到：

```text
${RUN_ROOT}/online_eval_${SPLIT}/results.json
${RUN_ROOT}/online_eval_${SPLIT}/summary.json
```

## 注意事项

- 这是 sparse-waypoint Diffusion Policy baseline，不是 dense per-timestep action chunk 预测。
- waypoint 来自 manifest 的 `full_task_heuristic_waypoints`，不是 metadata keyframe，也不是 event/subtask boundary。
- `source_bundle` 决定使用哪个 RGB root 和 low-dim root，所以 HPC 上要同时配置 `RGB_ROOT_200/RGB_ROOT_400` 与 `LOWDIM_ROOT_200/LOWDIM_ROOT_400`。
- CLIP 只用于 task text embedding，视觉主干是 raw RGB ResNetConv，不是 frozen image feature cache。
- checkpoint 中会保存 action/proprio normalization stats、CLIP text feature table 和 text vocabulary。
