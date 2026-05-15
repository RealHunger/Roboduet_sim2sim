# RoboDuet MuJoCo Sim2Sim 工作总结

本文档总结本阶段在 RoboDuet 的 MuJoCo sim2sim 迁移、调参、验证和文档整理工作。

## 一、工作目标

目标是把 RoboDuet 的 Go1 + ARX 机械臂策略从 Isaac Gym 迁移到 MuJoCo，并验证两边的状态、观测、动作和实际行为尽可能一致。

具体包括：

- 搭建 MuJoCo 主运行流程
- 对齐 Isaac 和 MuJoCo 的 state / obs / action
- 让狗能稳定走路
- 让机械臂能跟随目标
- 加入可视化和对比工具
- 固化调参结果并整理文档

## 二、主要工作内容

### 1. 搭建 MuJoCo sim2sim 主流程

新增并完善了 MuJoCo 主脚本：

- `sim2sim_mujoco/scripts/mujoco_policy_rollout.py`

支持能力包括：

- MuJoCo viewer
- realtime 播放
- 键盘控制
- heading hold
- 红球/蓝球目标可视化
- 交互 baseline 一键启动

当前推荐启动命令：

```bash
python sim2sim_mujoco/scripts/mujoco_policy_rollout.py --interactive-baseline
```

### 2. 对齐 Isaac 与 MuJoCo 的 state / obs / action

新增并使用了以下对比脚本：

- `sim2sim_mujoco/utils/compare_isaac_mujoco_state.py`
- `sim2sim_mujoco/utils/compare_isaac_mujoco_dog_obs.py`
- `sim2sim_mujoco/utils/compare_isaac_mujoco_arm_obs.py`

验证结果：

- state extraction 对齐
- 坐标系对齐
- 关节顺序对齐
- dog observation/action 对齐
- arm observation/action 对齐

其中 arm action 的早期差异，最终定位为 `obs_history` 初始化方式不同，而不是策略本身错了。

### 3. 调通狗的走路 baseline

通过 `sim2sim_mujoco/utils/mujoco_param_sweep.py` 做参数 sweep，主要扫：

- `leg_kd_scale`
- friction
- solref

关键结果：

- `leg_kd_scale = 1.8` 时，腿部阻尼过大，走路像被“刹住”
- `leg_kd_scale = 1.2` 时，走路明显更自然，torque 饱和减少

最终把 MuJoCo walking baseline 固定为：

- `leg_kd_scale = 1.2`
- `friction = 1.0`
- `solref = 0.02`

### 4. 调通机械臂控制

机械臂部分做了以下工作：

- 修复 `obs_history` 初始化问题
- 去掉 MuJoCo-only 的额外 `arm_action_limit=1.0`
- 对齐 arm actuator gain 到更接近 Isaac 的参数
- 让机械臂目标与 Isaac 的 `grasper` 定义一致
- 给机械臂目标添加平滑 `arm_target_alpha`
- 固定夹爪滑动关节，避免视觉抖动

此外还加入了红球/蓝球可视化：

- 红球 = 机械臂 command target
- 蓝球 = 实际 grasper 点
- `arm_dist` = 二者距离

### 5. 改善 MuJoCo 模型显示

为了让可视化更接近真实机器人：

- 给 Go1 腿部补上 mesh visual
- 保留简化 collision 几何用于物理
- 将 collision primitive 设为透明，避免 viewer 中重复显示

### 6. 建立 scripted replay 和 gap 对比工具

新增：

- `sim2sim_mujoco/utils/sim2sim_sequence.py`
- `sim2sim_mujoco/utils/mujoco_replay_sequence.py`
- `sim2sim_mujoco/utils/isaac_replay_sequence.py`
- `sim2sim_mujoco/utils/compare_sequence_gap.py`

用途：

- 统一编排一套固定动作序列
- 在 Isaac 和 MuJoCo 中分别 replay
- 输出 CSV 指标
- 自动比较两边 gap

### 7. 修复 Isaac replay 的黑屏和闪现问题

在 Isaac replay 中遇到：

- viewer 黑屏
- 动作做到一半突然闪现

最终通过以下方式解决：

- 主动设置 camera
- 每步调用 `render_gui()`
- 加 `--realtime`
- 默认禁用 reset / terminal，避免 replay 中途 teleport

### 8. 整理文档和项目结构

新增中文主文档：

- `docs/mujoco_sim2sim.md`

并更新：

- `README.md`
- `TODO.md`

同时清理了明显无用的临时文件和 Python 缓存。

## 三、遇到的主要难题

### 难题 1：刚开始无法判断问题是 state 错还是 physics 错

表现：

- Isaac 和 MuJoCo 行为差异明显
- 不知道是关节顺序、坐标系，还是动力学问题

解决：

- 强制同一初始状态
- 逐项对比 state / obs / action

结论：

- state / obs / action mapping 没问题
- 后续差异主要来自物理、接触、执行器和 history

### 难题 2：狗走路时 torque 经常打满

表现：

- 腿部动作僵硬
- `max_ctrl` 经常接近上限

解决：

- 做参数 sweep
- 把 `leg_kd_scale` 从 `1.8` 调到 `1.2`

### 难题 3：机械臂一开始动作很怪

表现：

- 初始 arm action 很大
- 机械臂不自然

解决：

- 修复 `obs_history`
- 启用 `--prefill-history`

### 难题 4：机械臂跟红球不够紧

表现：

- 蓝球追不上红球
- `arm_dist` 偏大

解决：

- 去掉 MuJoCo-only 的额外 `arm_action_limit`
- 对齐末端点定义到 Isaac 的 grasper
- 调高 arm actuator gain

### 难题 5：MuJoCo 里夹爪和腿的视觉效果不理想

表现：

- 夹爪晃动
- 狗腿显示为简化几何，观感不好

解决：

- 固定夹爪滑动关节并加 damping
- 增加 Go1 mesh visual
- 将 collision primitive 透明化

### 难题 6：Isaac replay 黑屏或中途闪现

解决：

- 修复 camera 设置
- 强制 viewer 刷新
- replay 默认禁用 reset

## 四、当前成果

目前已经具备：

- MuJoCo 交互控制版本
- Isaac / MuJoCo state 对齐验证
- dog / arm obs-action 对齐验证
- scripted sequence replay
- CSV gap 定量比较
- Go1 mesh visual
- 中文 sim2sim 文档

当前较稳定的使用入口：

```bash
python sim2sim_mujoco/scripts/mujoco_policy_rollout.py --interactive-baseline
```

## 五、一句话总结

本阶段完成了 RoboDuet 从 Isaac 到 MuJoCo 的 sim2sim 原型落地：对齐了 state / obs / action，调通了狗和机械臂的联合控制，建立了 replay 和 gap 对比工具，并把结果整理成了可直接使用的中文文档。
