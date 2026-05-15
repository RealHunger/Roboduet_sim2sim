# RoboDuet MuJoCo Sim2Sim 扩展

本目录是我在原始 RoboDuet 代码基础上整理的 IsaacGym-to-MuJoCo sim2sim 扩展工作。

## 保留原始 RoboDuet 的部分

- `go1_gym/`
- `go1_gym_learn/`
- 仓库根目录 `scripts/` 下的原始 IsaacGym 训练和播放脚本
- 原始 RoboDuet 的 license、citation 和 acknowledgement 信息

## 本目录整理的工作

- MuJoCo rollout 和 probe 脚本
- IsaacGym 与 MuJoCo 的 state / obs / action 对比脚本
- scripted replay 工具
- MuJoCo 专用机器人模型和生成的 MJCF
- sim2sim 过程记录、调参结果和实验总结

## 快速启动

```bash
python sim2sim_mujoco/scripts/mujoco_policy_rollout.py --interactive-baseline
```

完整使用流程见 `sim2sim_mujoco/docs/mujoco_sim2sim.md`。
