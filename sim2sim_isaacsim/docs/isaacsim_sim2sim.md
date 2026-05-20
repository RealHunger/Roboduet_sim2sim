# Isaac Sim Sim2Sim 使用指南

这份文档记录当前 RoboDuet Go1 + ARX 在 Isaac Sim / IsaacLab 里的 sim2sim 使用方式、调参结果、验证脚本和注意事项。

## 当前基准

- 训练目录：`runs/overnight_go1_512/dummy-sf8qzpe4_seed6218`
- Checkpoint：`30800`
- Isaac Sim USD：`sim2sim_isaacsim/resources/robots/arx5p2Go1/usd/arx5p2Go1.usd`
- 推荐 Python：使用已经安装 Isaac Sim、IsaacLab、PyTorch 和本仓库依赖的 `isaaclab` 环境。
- 当前关键时间步：`--sim-dt 0.005 --action-decimation 4`

## 快速启动

交互式 Isaac Sim 播放：

```bash
cd RoboDuet-master
python sim2sim_isaacsim/scripts/isaacsim_replay_sequence.py --interactive-baseline
```

`--interactive-baseline` 会自动启用 GUI、键盘控制、free base、PhysX position target、当前调好的腿部增益和红蓝球 marker。

当前 baseline 展开后等价于主要启用：

```text
--physics-targets
--free-base
--keyboard
--sim-dt 0.005
--action-decimation 4
--leg-kp-scale 2.0
--leg-kd-scale 1.5
--effort-limit 300
--velocity-limit 150
--arm-target-alpha 0.25
```

如果只想看狗，不让机械臂 policy 影响姿态，可以加：

```bash
--hold-arm-default
```

## 推荐演示顺序

复现时，建议按下面顺序：

1. 先运行交互式 baseline：`python sim2sim_isaacsim/scripts/isaacsim_replay_sequence.py --interactive-baseline`
2. 如果只想确认狗的行走效果，加 `--hold-arm-default`。
3. 再运行编排动作：`python sim2sim_isaacsim/scripts/isaacsim_replay_sequence.py --sequence-baseline --loop-sequence`
4. 如果需要保存指标，加 `--csv outputs/isaacsim_sequence_baseline.csv`。

## 键盘控制

键盘模式用 Isaac Sim GUI 的 keyboard event。启动后需要让 Isaac Sim 窗口获得焦点。

默认键位：

```text
NUMPAD_8  狗前进
NUMPAD_5  狗后退
NUMPAD_4  狗左移
NUMPAD_6  狗右移
NUMPAD_7  狗左转
NUMPAD_9  狗右转
NUMPAD_0  清零狗速度

I/K       机械臂 l 增/减
U/O       机械臂 p 增/减
J/L       机械臂 y 增/减
W/S       机械臂 pitch 减/增
A/D       机械臂 roll 增/减
Q/E       机械臂 yaw 增/减
R         清零狗速度并恢复机械臂默认 command
```

每次按键会在终端打印当前 command，例如：

```text
keyboard command x=0.35 y=0.00 yaw=0.00 arm_lpy=(0.50,0.20,0.00) arm_rpy=(0.10,0.50,0.00)
```

## Viewer 标记

- 红球：机械臂 `l/p/y` command 转换出来的目标位置。
- 蓝球：实际末端点，定义为 `zarx_body6 + local x 方向 0.1m` 的 grasper 点。
- 默认显示红蓝球；如果不想显示，可以加 `--hide-arm-markers`。

## 编排动作 Replay

Isaac Sim 里已经接入和 MuJoCo 相同的 scripted command timeline。

运行默认编排动作：

```bash
python sim2sim_isaacsim/scripts/isaacsim_replay_sequence.py \
  --sequence-baseline \
  --csv outputs/isaacsim_sequence_baseline.csv
```

循环播放编排动作，适合 GUI 观察或录屏：

```bash
python sim2sim_isaacsim/scripts/isaacsim_replay_sequence.py \
  --sequence-baseline \
  --loop-sequence \
  --csv outputs/isaacsim_sequence_baseline.csv
```

如果只想看狗的步态，不让机械臂 policy 干扰：

```bash
python sim2sim_isaacsim/scripts/isaacsim_replay_sequence.py \
  --sequence-baseline \
  --hold-arm-default \
  --csv outputs/isaacsim_dog_sequence.csv
```

编排动作大致包括：

- 0-2s：站立
- 2-5s：慢速前进
- 5-8s：较快前进
- 8-11s：前进并转向
- 11-15s：原地转向
- 15-17s：后退
- 17s 之后：机械臂目标变化

## 当前 Isaac Sim 调参结果

当前比较稳定的 walking baseline：

- `--sim-dt 0.005`
- `--action-decimation 4`
- `--physics-targets`
- `--free-base`
- `--leg-kp-scale 2.0`
- `--leg-kd-scale 1.5`
- `--effort-limit 300`
- `--velocity-limit 150`
- `--arm-target-alpha 0.25`
- 默认不额外设置 `--arm-action-limit`，只使用训练时 policy 自己的 `clip_actions`。

最关键的结论是：Isaac Sim 必须显式设置 `dt=0.005`。如果不对齐训练和 MuJoCo 的 physics dt，狗会表现为能动但步态很怪、脚抬不起来或像蠕动。

## 验证脚本

- `sim2sim_isaacsim/scripts/isaacsim_view_robot.py`：加载并检查 USD stage、joint、rigid body 和 policy joint 名称。
- `sim2sim_isaacsim/scripts/isaacsim_stand_check.py`：检查 articulation 初始化和 policy joint 顺序。
- `sim2sim_isaacsim/scripts/isaacsim_state_probe.py`：从 Isaac Sim 提取 sim2sim policy state。
- `sim2sim_isaacsim/scripts/isaacsim_obs_probe.py`：构造 dog/arm observation 并跑 policy action。
- `sim2sim_isaacsim/scripts/isaacsim_pd_check.py`：检查 PhysX actuator position target 跟踪。
- `sim2sim_isaacsim/scripts/isaacsim_replay_sequence.py`：当前主入口，支持 free-base、键盘控制、编排动作、红蓝球 marker 和 CSV 输出。

一般使用时优先跑 `--interactive-baseline` 或 `--sequence-baseline`，只有排查 USD、关节、obs/action 或 actuator 问题时才需要单独使用这些 probe/check 脚本。

## 调试参数

### 放大狗速度命令

```bash
--dog-command-scale 1.3
```

只建议用于视觉检查，不建议作为最终对比 baseline。`2.0` 已验证会让动作明显，但容易过激并触发 early stop。

### 打印关节 target/actual/error

```bash
--debug-joints
```

用于判断是 mapping 问题还是 actuator 跟踪问题。

当前排查结论：不是典型关节顺序映射错。policy target 和 actual 都按同名关节对应，主要问题曾经是时间步和 PhysX actuator 跟踪。

### 只看狗

```bash
--hold-arm-default
```

用于排除机械臂 policy 对姿态的干扰。调狗时优先使用。



