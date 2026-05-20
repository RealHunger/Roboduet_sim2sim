# RoboDuet Isaac Sim Sim2Sim 扩展

本目录整理 RoboDuet Go1 + ARX 在 Isaac Sim / IsaacLab 中的 sim2sim 部署工作。当前已经完成 USD 加载、policy replay、free-base 行走、键盘控制、编排动作和机械臂红蓝球可视化。

## 交付状态

- 主入口：`sim2sim_isaacsim/scripts/isaacsim_replay_sequence.py`
- 使用文档：`sim2sim_isaacsim/docs/isaacsim_sim2sim.md`
- 机器人 USD：`sim2sim_isaacsim/resources/robots/arx5p2Go1/usd/arx5p2Go1.usd`
- 推荐环境：`conda activate isaaclab`
- 当前关键结论：Isaac Sim 必须使用 `--sim-dt 0.005`，否则步态会明显不自然。

## 快速启动

交互式键盘控制 baseline：

```bash
python sim2sim_isaacsim/scripts/isaacsim_replay_sequence.py --interactive-baseline
```

编排动作 baseline：

```bash
python sim2sim_isaacsim/scripts/isaacsim_replay_sequence.py --sequence-baseline --loop-sequence
```

只看狗，不让机械臂 policy 影响姿态：

```bash
python sim2sim_isaacsim/scripts/isaacsim_replay_sequence.py --interactive-baseline --hold-arm-default
```

## 目录说明

- `scripts/`：Isaac Sim USD 检查、state/obs probe、PD 检查、policy replay、键盘控制和 scripted sequence。
- `resources/`：转换后的 Isaac Sim USD 机器人资源。
- `docs/`：使用指南、调参结果和注意事项。

## 演示说明

- 红球表示机械臂 command target。
- 蓝球表示实际 grasper/end-effector 点。
- 小键盘控制狗，`I/K/U/O/J/L/W/S/A/D/Q/E` 控制机械臂 command。
- `R` 清零狗速度并恢复机械臂默认 command。

## 注意事项

- Isaac Sim 启动时会打印很多 warning，大部分是 Kit/visual reference 噪声，不影响当前控制链路。
- MuJoCo 对比脚本依赖 `isaacgym` 环境，不建议在 `isaaclab` 环境直接跑。
- 详细说明见：`sim2sim_isaacsim/docs/isaacsim_sim2sim.md`。
