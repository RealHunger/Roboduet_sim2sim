import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Extract RoboDuet state from Isaac Sim in the sim2sim policy format.")
    parser.add_argument(
        "--usd",
        type=str,
        default="sim2sim_isaacsim/resources/robots/arx5p2Go1/usd/arx5p2Go1.usd",
        help="Path to the converted robot USD, relative to repo root or absolute.",
    )
    parser.add_argument("--prim-path", type=str, default="/World/Robot")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--base-z", type=float, default=0.50)
    parser.add_argument("--use-gpu", action="store_true", help="Use CUDA PhysX. Default uses CPU PhysX for bring-up stability.")
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
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, args_cli.base_z), joint_pos=DEFAULT_JOINT_POS),
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


def quat_wxyz_to_matrix(quat):
    w, x, y, z = quat.unbind(-1)
    row0 = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), dim=-1)
    row1 = torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), dim=-1)
    row2 = torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def quat_apply_wxyz(quat, vec):
    rot = quat_wxyz_to_matrix(quat)
    return torch.matmul(rot, vec.unsqueeze(-1)).squeeze(-1)


def roll_pitch_from_quat_wxyz(quat):
    w, x, y, z = quat.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def tensor_to_list(tensor):
    return [float(x) for x in tensor.detach().cpu().reshape(-1)]


def print_vec(name, tensor):
    values = tensor_to_list(tensor)
    formatted = " ".join(f"{value: .6f}" for value in values)
    print(f"{name}: [{formatted}]")


def initialize_robot(robot, sim):
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] = torch.tensor([0.0, 0.0, args_cli.base_z], dtype=root_state.dtype, device=root_state.device)
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=root_state.dtype, device=root_state.device)
    root_state[:, 7:] = 0.0

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    robot.write_data_to_sim()

    sim_dt = sim.get_physics_dt()
    for _ in range(args_cli.steps):
        sim.step(render=not args_cli.headless)
        robot.update(sim_dt)


def extract_state(robot):
    root = robot.data.root_link_state_w[0]
    base_pos = root[:3]
    base_quat_wxyz = root[3:7]
    base_lin_vel_world = root[7:10]
    base_ang_vel_world = root[10:13]
    rot = quat_wxyz_to_matrix(base_quat_wxyz)
    base_lin_vel_body = rot.transpose(-1, -2) @ base_lin_vel_world
    base_ang_vel_body = rot.transpose(-1, -2) @ base_ang_vel_world
    projected_gravity = rot.transpose(-1, -2) @ torch.tensor([0.0, 0.0, -1.0], dtype=root.dtype, device=root.device)
    roll, pitch = roll_pitch_from_quat_wxyz(base_quat_wxyz)

    joint_index = {name: idx for idx, name in enumerate(robot.data.joint_names)}
    policy_indices = torch.tensor([joint_index[name] for name in POLICY_JOINT_ORDER], device=robot.data.joint_pos.device)
    q_policy = robot.data.joint_pos[0, policy_indices]
    qd_policy = robot.data.joint_vel[0, policy_indices]

    body_index = {name: idx for idx, name in enumerate(robot.data.body_names)}
    ee_idx = body_index["zarx_body6"]
    ee_state = robot.data.body_link_state_w[0, ee_idx]
    ee_pos = ee_state[:3]
    ee_quat = ee_state[3:7]
    grasper = ee_pos + quat_apply_wxyz(ee_quat, torch.tensor([0.1, 0.0, 0.0], dtype=ee_pos.dtype, device=ee_pos.device))

    return {
        "base_pos": base_pos,
        "base_quat_wxyz": base_quat_wxyz,
        "base_lin_vel_world": base_lin_vel_world,
        "base_ang_vel_world": base_ang_vel_world,
        "base_lin_vel_body": base_lin_vel_body,
        "base_ang_vel_body": base_ang_vel_body,
        "projected_gravity": projected_gravity,
        "roll_pitch": torch.stack((roll, pitch)),
        "q_policy": q_policy,
        "qd_policy": qd_policy,
        "zarx_body6_pos": ee_pos,
        "zarx_body6_quat_wxyz": ee_quat,
        "grasper_pos": grasper,
    }


def main():
    usd_path = resolve_path(args_cli.usd)
    if not usd_path.exists():
        raise FileNotFoundError(f"USD file does not exist: {usd_path}")

    sim_device = args_cli.device if args_cli.use_gpu else "cpu"
    sim = SimulationContext(SimulationCfg(device=sim_device, use_fabric=False))
    sim_utils.GroundPlaneCfg(size=(6.0, 6.0)).func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg(size=(6.0, 6.0)))
    robot = Articulation(make_robot_cfg(usd_path))
    sim.reset()
    robot.reset()
    initialize_robot(robot, sim)

    state = extract_state(robot)
    print(f"[INFO] Repo root: {REPO_ROOT}")
    print(f"[INFO] USD path: {usd_path}")
    print(f"[INFO] Sim device: {sim_device}")
    print(f"[INFO] use_fabric: False")
    print(f"[INFO] policy joint indices: {[robot.data.joint_names.index(name) for name in POLICY_JOINT_ORDER]}")
    for key, value in state.items():
        print_vec(key, value)
    print("[INFO] State probe complete.")


if __name__ == "__main__":
    success = False
    try:
        main()
        success = True
    finally:
        if args_cli.headless and success:
            os._exit(0)
        simulation_app.close()
