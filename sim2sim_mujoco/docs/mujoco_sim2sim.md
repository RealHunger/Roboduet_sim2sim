# MuJoCo Sim2Sim 使用指南

这份文档记录当前 RoboDuet Go1 + ARX 在 MuJoCo 里的 sim2sim 使用方式、调参结果、验证脚本和注意事项。

## 当前基准

- 训练目录：`runs/overnight_go1_512/dummy-sf8qzpe4_seed6218`
- Checkpoint：`30800`
- MuJoCo 模型：`sim2sim_mujoco/resources/robots/arx5p2Go1_mujoco/mjcf/arx5p2Go1_mujoco.xml`
- 推荐 Python：使用已经安装 Isaac Gym、MuJoCo、PyTorch 和本仓库依赖的 Python 环境。

## 快速启动

交互式 MuJoCo 播放：

```bash
cd RoboDuet-master
python sim2sim_mujoco/scripts/mujoco_policy_rollout.py --interactive-baseline
```

`--interactive-baseline` 会自动启用 viewer、realtime、键盘控制、heading hold、history prefill、狗 idle hold，以及当前调好的 MuJoCo walking 参数。

如果要保存 CSV，建议先手动创建 `outputs/` 目录。

## 键盘控制

默认键位和 Isaac 的 `play_by_key.py` 一致：

```text
NUMPAD_8  狗前进
NUMPAD_5  狗后退
NUMPAD_4  狗左移
NUMPAD_6  狗右移
NUMPAD_7  狗左转
NUMPAD_9  狗右转
NUMPAD_0  清零狗速度

U/O       机械臂上/下
I/K       机械臂前/后
J/L       机械臂左/右
W/S       机械臂 pitch 下/上
A/D       机械臂 roll 左/右
Q/E       机械臂 yaw 左/右
R         清零狗速度并恢复机械臂默认目标
[/]       调整狗速度命令步长
```

如果这些键和 MuJoCo viewer 自带快捷键冲突，可以使用安全键位：

```bash
--keyboard-layout mujoco-safe
```

安全键位里，狗仍然用小键盘，机械臂位置仍然用 `I/K`、`U/O`、`J/L`，机械臂姿态改成 `Z/X`、`C/V`、`B/N`，`M` 用于清零狗速度并恢复机械臂默认目标。

## Viewer 标记

- 红球：机械臂 `l/p/y` command 转换出来的目标位置。
- 蓝球：实际末端点，定义和 Isaac 一致，为 `link6 + local x 方向 0.1m` 的 grasper 点。
- `arm_dist`：终端里打印的红球到蓝球距离，单位是米。

## 当前 MuJoCo 调参结果

- 走路阻尼：`--leg-kd-scale 1.2`
- 交互默认启用：`--heading-hold --heading-kp 2.0`
- 交互默认启用机械臂目标平滑：`--arm-target-alpha 0.25`
- 默认不再额外设置 `--arm-action-limit`，只使用训练时 policy 自己的 `clip_actions`。
- 机械臂 position actuator gain 已对齐到更接近 Isaac：`50/50/70/50/50/50`。
- 夹爪滑动关节 `zarx_j7/j8` 会被固定在默认开口，并增加阻尼，避免 MuJoCo 里夹爪视觉抖动。

## 视觉模型说明

- Go1 的 hip/thigh/calf 已经启用 mesh visual。
- thigh/calf/foot 的简化 collision geom 仍然参与物理和接触，但在 viewer 里设为透明。
- MuJoCo 模型故意保留简单 collision 几何，因为这样接触更稳定。

## 编排动作 Replay

为了直观比较 sim2sim，可以用同一套 dog/arm command timeline 分别在 MuJoCo 和 Isaac 里播放，并输出 CSV 指标。

MuJoCo 可视化 replay：

```bash
python sim2sim_mujoco/utils/mujoco_replay_sequence.py \
  --viewer \
  --realtime \
  --steps 1600 \
  --output outputs/mujoco_sequence.csv
```

如果要录屏，可以让动作循环播放：

```bash
python sim2sim_mujoco/utils/mujoco_replay_sequence.py \
  --viewer \
  --realtime \
  --loop-sequence \
  --steps 100000 \
  --output outputs/mujoco_sequence.csv
```

Isaac 可视化 replay：

```bash
python sim2sim_mujoco/utils/isaac_replay_sequence.py \
  --realtime \
  --steps 1600 \
  --output outputs/isaac_sequence.csv
```

Isaac 也可以循环播放同一套动作：

```bash
python sim2sim_mujoco/utils/isaac_replay_sequence.py \
  --realtime \
  --loop-sequence \
  --steps 100000 \
  --output outputs/isaac_sequence.csv
```

Isaac replay 默认禁用 terminal/reset，避免中途因为环境判定 reset 而“闪现”。如果想调试原始 reset 行为，可以加：

```bash
--allow-resets
```

比较两边 CSV gap：

```bash
python sim2sim_mujoco/utils/compare_sequence_gap.py \
  --isaac outputs/isaac_sequence.csv \
  --mujoco outputs/mujoco_sequence.csv
```

## 验证脚本

- `sim2sim_mujoco/utils/compare_isaac_mujoco_state.py`：强制同一 default state 后，对比 state extraction、坐标系和关节顺序。
- `sim2sim_mujoco/utils/compare_isaac_mujoco_dog_obs.py`：对比 dog observation 和 dog action。
- `sim2sim_mujoco/utils/compare_isaac_mujoco_arm_obs.py`：对比 arm observation 和 arm action。
- `sim2sim_mujoco/utils/mujoco_param_sweep.py`：headless MuJoCo 参数 sweep，用于调接触、摩擦、阻尼。
- `sim2sim_mujoco/utils/mujoco_arm_pose_probe.py`：机械臂 target/q/末端位置 probe。
- `sim2sim_mujoco/utils/mujoco_replay_sequence.py`：MuJoCo 编排动作 replay，并写 CSV 指标。
- `sim2sim_mujoco/utils/isaac_replay_sequence.py`：Isaac 编排动作 replay，并写 CSV 指标。
- `sim2sim_mujoco/utils/compare_sequence_gap.py`：汇总 Isaac 和 MuJoCo 两个 CSV 的 gap。

## 重要结论

- 在强制同一初始状态下，state extraction、坐标系、关节顺序已经对齐。
- dog obs/action mapping 已验证。
- arm obs/action mapping 已验证；早期 action mismatch 是由 obs history 初始化不同造成的。
- MuJoCo 默认额外限制 `arm_action_limit=1.0` 会导致机械臂追踪变差，现在默认已经移除。
- `leg_kd_scale` 从 `1.8` 改到 `1.2` 后，走路明显更自然，torque 饱和少很多。
- MuJoCo 和 Isaac 的接触、reset 动态仍然不完全一致，所以推荐用 scripted replay + CSV gap 做最终对比。

## 重新生成 MJCF

如果需要重新生成 MuJoCo XML：

```bash
python sim2sim_mujoco/utils/mujoco_phase5b_generate_mjcf.py
```

当前生成脚本会保留 actuator、contact、红蓝 marker、Go1 mesh visual、透明 collision geom、夹爪阻尼等设置。
