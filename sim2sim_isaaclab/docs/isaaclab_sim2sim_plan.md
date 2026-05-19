# RoboDuet Isaac Lab Sim2Sim 计划书

本文档记录从当前 RoboDuet Isaac Gym / MuJoCo sim2sim 工作继续迁移到 Isaac Lab 的执行计划。目标不是马上重新训练，而是先把已有 RoboDuet checkpoint 在 Isaac Lab 中跑通，并逐步完成 state / obs / action 对齐。

## 一、总体目标

将 RoboDuet 的 Go1 + ARX 机械臂策略迁移到 Isaac Lab 环境中，建立一套可复现的 sim2sim 验证流程。

主要目标包括：

- 在 Isaac Lab 中加载原始 RoboDuet Go1 + ARX URDF。
- 提取 Isaac Lab 中的 robot state。
- 对齐 Isaac Gym 与 Isaac Lab 的 state / obs / action。
- 加载已有 RoboDuet checkpoint 做 policy inference。
- 复用 scripted replay 做 Isaac Gym / MuJoCo / Isaac Lab 三方对比。

## 二、使用模型

优先使用原始 RoboDuet URDF：

```text
resources/robots/arx5p2Go1/urdf/arx5p2Go1.urdf
```

原因：

- 该模型最接近原始 Isaac Gym 训练配置。
- 便于和现有 RoboDuet policy 的关节顺序、默认姿态、action mapping 对齐。
- 后续对比 Isaac Gym 和 Isaac Lab 时变量更少。

## 三、目录规划

新增 Isaac Lab 相关目录：

```text
sim2sim_isaaclab/
  README.md
  docs/
    isaaclab_sim2sim_plan.md
    isaaclab_sim2sim.md
  scripts/
    isaaclab_policy_rollout.py
  utils/
    isaaclab_asset_probe.py
    isaaclab_state_probe.py
    compare_isaacgym_isaaclab_state.py
    compare_isaacgym_isaaclab_obs.py
    isaaclab_replay_sequence.py
  resources/
    robots/
      arx5p2Go1/
        usd/
```

第一阶段只需要创建 `utils/` 和 `docs/`，先跑通 asset probe。

## 四、阶段 1：确认 Isaac Lab 环境

### 目标

确认本机 Isaac Lab 可启动，能运行最小 headless 脚本。

### 操作

先确认不要使用 `isaacgym` 环境运行 Isaac Lab。Isaac Lab 使用独立环境：

```bash
source /home/reality-hunger/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
```

首次运行 Isaac Sim / Isaac Lab 时需要接受 NVIDIA Omniverse EULA。非交互命令建议直接设置环境变量：

```bash
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=yes
```

在 Isaac Lab 目录下运行空场景：

```bash
cd /home/reality-hunger/IsaacLab
ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=yes \
CONDA_PREFIX=/home/reality-hunger/miniconda3/envs/env_isaaclab \
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
```

### 验收标准

- Isaac Lab 能正常启动。
- 输出中 Python 路径应为 `env_isaaclab`，不是 `isaacgym`。
- 脚本无 import error。
- `create_empty.py` 会持续运行；看到 Isaac Lab 日志和 headless kit 加载成功即可，手动 Ctrl-C 退出。

### 可能问题

- Isaac Sim / Isaac Lab 环境变量未配置。
- GPU / display / headless 参数问题。
- Python 环境不是 Isaac Lab 自带环境。
- 未接受 EULA 时会出现 `Do you accept the EULA?` 或 `Unable to bootstrap inner kit kernel`。

## 五、阶段 2：加载 RoboDuet URDF

### 目标

在 Isaac Lab 中加载：

```text
resources/robots/arx5p2Go1/urdf/arx5p2Go1.urdf
```

并打印机器人结构信息。

### 新增脚本

```text
sim2sim_isaaclab/utils/isaaclab_asset_probe.py
```

### 脚本功能

- 启动 Isaac Lab。
- 加载地面和灯光。
- 通过 URDF 或转换后的 USD spawn Go1 + ARX。
- 打印：
  - joint names
  - body names
  - DOF 数量
  - default joint position
  - root pose

### 验收标准

- Isaac Lab 中能成功 spawn 机器人。
- joint/body 信息能完整打印。
- DOF 数量和预期一致。
- 没有明显的 missing mesh / invalid joint 错误。

### 可能问题

- URDF 中 mesh 相对路径在 Isaac Lab 中解析失败。
- URDF importer 对部分 joint / mimic / transmission 支持不一致。
- 需要先转 USD 再加载。

## 六、阶段 3：URDF 转 USD

### 目标

如果 Isaac Lab 不能直接稳定加载 URDF，则先转换成 USD。

### 输出路径

```text
sim2sim_isaaclab/resources/robots/arx5p2Go1/usd/arx5p2Go1.usd
```

### 参考命令

```bash
cd /home/reality-hunger/IsaacLab
./isaaclab.sh -p scripts/tools/convert_urdf.py \
  /home/reality-hunger/roboduet_ws/RoboDuet-master/resources/robots/arx5p2Go1/urdf/arx5p2Go1.urdf \
  /home/reality-hunger/roboduet_ws/RoboDuet-master/sim2sim_isaaclab/resources/robots/arx5p2Go1/usd/arx5p2Go1.usd \
  --joint-target-type position \
  --joint-stiffness 0.0 \
  --joint-damping 0.0 \
  --headless
```

### 验收标准

- 成功生成 USD 文件。
- USD 能被 Isaac Lab `UsdFileCfg` 加载。
- joint/body 数量与 URDF 导入结果一致。

## 七、阶段 4：提取 Isaac Lab state

### 目标

实现 Isaac Lab 版 state extraction，输出字段与 MuJoCo 版保持一致。

### 新增脚本

```text
sim2sim_isaaclab/utils/isaaclab_state_probe.py
```

### 输出字段

```python
base_pos
base_quat_wxyz
base_lin_vel_body
base_ang_vel_body
projected_gravity
roll
pitch
q_policy
qd_policy
```

### 验收标准

- 能读取 root pose / velocity。
- 能读取所有 joint position / velocity。
- 能按照 RoboDuet policy joint order 重排为 `q_policy` 和 `qd_policy`。
- default pose 下 `q_policy` 接近 RoboDuet default dof position。

### 重点检查

- Isaac Lab 四元数顺序是否为 `wxyz`。
- root velocity 是 world frame 还是 body frame。
- joint name 是否和 Isaac Gym / MuJoCo 一致。
- floating base 是否影响 joint index 偏移。

## 八、阶段 5：Isaac Gym vs Isaac Lab state 对齐

### 目标

验证 Isaac Gym 和 Isaac Lab 在同一 default state 下的 state mapping 是否一致。

### 新增脚本

```text
sim2sim_isaaclab/utils/compare_isaacgym_isaaclab_state.py
```

### 对比项目

- base position
- base quaternion
- projected gravity
- roll / pitch
- policy joint position
- policy joint velocity

### 验收标准

- forced default state 下，关节顺序完全对齐。
- base 姿态、projected gravity、roll/pitch 数值一致或接近。
- 如果有差异，能明确是坐标系、四元数顺序、还是 joint order 问题。

## 九、阶段 6：构造 Isaac Lab obs

### 目标

在 Isaac Lab state 基础上构造 RoboDuet policy 需要的 dog obs 和 arm obs。

### 可复用逻辑

参考：

```text
sim2sim_mujoco/utils/mujoco_obs_probe.py
```

可复用内容：

- `DEFAULT_DOF_POS_20`
- `build_arm_obs`
- `build_dog_obs`
- command scale
- previous action
- obs history 更新方式

### 验收标准

- arm obs shape 与 `cfg.arm.arm_num_observations` 一致。
- dog obs shape 与 `cfg.dog.dog_num_observations` 一致。
- default state 下 obs 与 Isaac Gym 结果接近。
- policy forward 输出 finite action。

## 十、阶段 7：加载 checkpoint 做 policy inference

### 目标

在 Isaac Lab 中加载 RoboDuet checkpoint，完成一次 dog/arm policy forward。

### 依赖

```text
scripts/load_policy.py
```

### 验收标准

- 成功加载 dog policy 和 arm policy。
- dog action shape 正确。
- arm action shape 正确。
- action 没有 NaN / Inf。
- 与 Isaac Gym default-state action 接近。

## 十一、阶段 8：Isaac Lab policy rollout

### 目标

让已有 RoboDuet policy 在 Isaac Lab 中闭环运行。

### 新增脚本

```text
sim2sim_isaaclab/scripts/isaaclab_policy_rollout.py
```

### 初始策略

先使用保守控制：

- dog command 为 0。
- arm command 为默认目标。
- leg 使用 PD target 或 Isaac Lab actuator target。
- arm 使用 position target。

### 验收标准

- 机器人能在 Isaac Lab 中维持站立若干秒。
- action 和 target joint position 正常更新。
- 没有明显爆炸、NaN、关节顺序错乱。

## 十二、阶段 9：scripted replay 与 CSV 对比

### 目标

复用已有 command timeline，在 Isaac Gym / MuJoCo / Isaac Lab 中做统一 replay。

### 可复用文件

```text
sim2sim_mujoco/utils/sim2sim_sequence.py
sim2sim_mujoco/utils/compare_sequence_gap.py
```

### 新增脚本

```text
sim2sim_isaaclab/utils/isaaclab_replay_sequence.py
```

### 输出指标

- base position
- body velocity
- roll / pitch
- arm target position
- arm end-effector position
- arm distance
- max control / max target delta

### 验收标准

- Isaac Lab 能输出同格式 CSV。
- 能和 MuJoCo / Isaac Gym 的 CSV 做离线比较。
- 能定位主要 gap 来自 mapping、actuator、contact 还是 reset dynamics。

## 十三、推荐执行顺序

建议严格按下面顺序推进：

1. 跑 Isaac Lab 空场景，确认环境可用。
2. 写 `isaaclab_asset_probe.py`，加载 URDF 或 USD。
3. 打印 joint/body names，确认 DOF 数量。
4. 写 `isaaclab_state_probe.py`，提取 state。
5. 对齐 default state。
6. 构造 obs。
7. 加载 checkpoint，跑一次 policy forward。
8. 做短 rollout。
9. 做 scripted replay。
10. 写 `isaaclab_sim2sim.md` 记录结果。

## 十四、第一阶段完成标准

第一阶段只要求完成：

```text
Isaac Lab 能加载 RoboDuet Go1 + ARX 模型，并打印完整 joint/body 信息。
```

完成后再进入 state / obs / action 对齐，不要一开始就直接跑 policy。
