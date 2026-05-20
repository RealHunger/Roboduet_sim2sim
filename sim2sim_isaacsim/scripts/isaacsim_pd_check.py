import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal RoboDuet PhysX PD position target check.")
    parser.add_argument("--usd", type=str, default="sim2sim_isaacsim/resources/robots/arx5p2Go1/usd/arx5p2Go1.usd")
    parser.add_argument("--prim-path", type=str, default="/World/Robot")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--base-z", type=float, default=0.50)
    parser.add_argument("--leg-kp", type=float, default=35.0)
    parser.add_argument("--leg-kd", type=float, default=1.0)
    parser.add_argument("--arm-kp", type=float, default=25.0)
    parser.add_argument("--arm-kd", type=float, default=2.0)
    parser.add_argument("--target-offset", type=float, default=0.0)
    parser.add_argument("--use-gpu", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

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
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit_sim=200.0,
                velocity_limit_sim=100.0,
                stiffness=args_cli.leg_kp,
                damping=args_cli.leg_kd,
            ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["zarx_j.*"],
                effort_limit_sim=200.0,
                velocity_limit_sim=100.0,
                stiffness=args_cli.arm_kp,
                damping=args_cli.arm_kd,
            ),
        },
    )


def print_usd_drives(stage):
    print("[INFO] USD drive summary")
    count = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive:
                stiffness = drive.GetStiffnessAttr().Get()
                damping = drive.GetDampingAttr().Get()
                print(f"  {prim.GetName():<24} stiffness={stiffness} damping={damping}")
                count += 1
    print(f"[INFO] USD angular drives: {count}")


def reset_robot(robot, sim):
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] = torch.tensor([0.0, 0.0, args_cli.base_z], dtype=root_state.dtype, device=root_state.device)
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=root_state.dtype, device=root_state.device)
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    sim.step(render=not args_cli.headless)
    robot.update(sim.get_physics_dt())


def main():
    usd_path = resolve_path(args_cli.usd)
    sim_device = args_cli.device if args_cli.use_gpu else "cpu"
    sim = SimulationContext(SimulationCfg(device=sim_device, use_fabric=False))
    sim_utils.GroundPlaneCfg(size=(6.0, 6.0)).func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg(size=(6.0, 6.0)))
    robot = Articulation(make_robot_cfg(usd_path))
    sim.reset()
    print_usd_drives(sim.stage)
    reset_robot(robot, sim)

    name_to_index = {name: idx for idx, name in enumerate(robot.data.joint_names)}
    joint_ids = torch.tensor([name_to_index[name] for name in POLICY_JOINT_ORDER], dtype=torch.int64, device=robot.data.joint_pos.device)
    target_q = robot.data.default_joint_pos.clone()
    target_q[:, joint_ids] += args_cli.target_offset
    robot.set_joint_position_target(target_q)

    sim_dt = sim.get_physics_dt()
    print(f"[INFO] PD check start device={sim_device} steps={args_cli.steps} leg=({args_cli.leg_kp},{args_cli.leg_kd}) arm=({args_cli.arm_kp},{args_cli.arm_kd})")
    for step in range(args_cli.steps):
        robot.write_data_to_sim()
        sim.step(render=not args_cli.headless)
        robot.update(sim_dt)
        if step % args_cli.print_every == 0 or step == args_cli.steps - 1:
            q = robot.data.joint_pos[:, joint_ids]
            abs_err = torch.abs(q - target_q[:, joint_ids]).reshape(-1)
            worst_idx = int(torch.argmax(abs_err).item())
            leg_err = torch.max(abs_err[:12]).item()
            arm_err = torch.max(abs_err[12:18]).item()
            err = abs_err[worst_idx].item()
            z = robot.data.root_link_state_w[0, 2].item()
            print(
                f"[STEP {step:04d}] z={z:.4f} max_q_err={err:.5f} "
                f"leg_err={leg_err:.5f} arm_err={arm_err:.5f} worst={POLICY_JOINT_ORDER[worst_idx]}"
            )
    print("[INFO] PD check complete.")


if __name__ == "__main__":
    success = False
    try:
        main()
        success = True
    finally:
        if args_cli.headless and success:
            os._exit(0)
        simulation_app.close()
