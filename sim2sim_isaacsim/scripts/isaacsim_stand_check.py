import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Check RoboDuet USD articulation initialization in Isaac Sim.")
    parser.add_argument(
        "--usd",
        type=str,
        default="sim2sim_isaacsim/resources/robots/arx5p2Go1/usd/arx5p2Go1.usd",
        help="Path to the converted robot USD, relative to repo root or absolute.",
    )
    parser.add_argument("--prim-path", type=str, default="/World/Robot")
    parser.add_argument("--steps", type=int, default=5, help="Number of physics steps to run after initialization.")
    parser.add_argument("--base-z", type=float, default=0.50)
    parser.add_argument("--use-gpu", action="store_true", help="Use CUDA PhysX. Default uses CPU PhysX to avoid Fabric CUDA errors during bring-up.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_JOINT_POS = {
    "FL_hip_joint": 0.1,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "FR_hip_joint": -0.1,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "RL_hip_joint": 0.1,
    "RL_thigh_joint": 1.0,
    "RL_calf_joint": -1.5,
    "RR_hip_joint": -0.1,
    "RR_thigh_joint": 1.0,
    "RR_calf_joint": -1.5,
    "zarx_j1": 0.0,
    "zarx_j2": 0.8,
    "zarx_j3": 0.8,
    "zarx_j4": 0.0,
    "zarx_j5": 0.0,
    "zarx_j6": 0.0,
    "zarx_j7": 0.0,
    "zarx_j8": 0.0,
}

POLICY_JOINT_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "zarx_j1", "zarx_j2", "zarx_j3", "zarx_j4", "zarx_j5", "zarx_j6",
]


def resolve_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def make_robot_cfg(usd_path):
    return ArticulationCfg(
        prim_path=args_cli.prim_path,
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, args_cli.base_z),
            joint_pos=DEFAULT_JOINT_POS,
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=200.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=0.0,
            )
        },
    )


def print_named_values(title, names, values):
    print(f"\n[INFO] {title}")
    flat = values.detach().cpu().reshape(-1)
    for name, value in zip(names, flat):
        print(f"  {name:<24} {value: .6f}")


def main():
    usd_path = resolve_path(args_cli.usd)
    if not usd_path.exists():
        raise FileNotFoundError(f"USD file does not exist: {usd_path}")

    sim_device = args_cli.device if args_cli.use_gpu else "cpu"
    sim_cfg = SimulationCfg(device=sim_device, use_fabric=False)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.0, 1.4], [0.0, 0.0, 0.35])

    ground_cfg = sim_utils.GroundPlaneCfg(size=(6.0, 6.0))
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(make_robot_cfg(usd_path))
    sim.reset()
    robot.reset()

    print(f"[INFO] Repo root: {REPO_ROOT}")
    print(f"[INFO] USD path: {usd_path}")
    print(f"[INFO] Sim device: {sim_device}")
    print(f"[INFO] use_fabric: False")
    print(f"[INFO] num joints: {len(robot.data.joint_names)}")
    print(f"[INFO] num bodies: {len(robot.data.body_names)}")

    print("\n[INFO] Joint names")
    for idx, name in enumerate(robot.data.joint_names):
        print(f"  {idx:02d} {name}")

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] = torch.tensor([0.0, 0.0, args_cli.base_z], dtype=root_state.dtype, device=root_state.device)
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=root_state.dtype, device=root_state.device)
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.set_joint_position_target(joint_pos)
    robot.write_data_to_sim()

    sim_dt = sim.get_physics_dt()
    for _ in range(args_cli.steps):
        sim.step(render=not args_cli.headless)
        robot.update(sim_dt)

    print_named_values("Root state", ["x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz"], robot.data.root_state_w[0])

    name_to_index = {name: idx for idx, name in enumerate(robot.data.joint_names)}
    missing = [name for name in POLICY_JOINT_ORDER if name not in name_to_index]
    if missing:
        print("\n[ERROR] Missing policy joints:", missing)
    else:
        indices = torch.tensor([name_to_index[name] for name in POLICY_JOINT_ORDER], device=robot.data.joint_pos.device)
        print_named_values("Policy-order joint positions", POLICY_JOINT_ORDER, robot.data.joint_pos[0, indices])
        print_named_values("Policy-order joint velocities", POLICY_JOINT_ORDER, robot.data.joint_vel[0, indices])

    print("\n[INFO] Stand check complete.")


if __name__ == "__main__":
    success = False
    try:
        main()
        success = True
    finally:
        if args_cli.headless and success:
            os._exit(0)
        simulation_app.close()
