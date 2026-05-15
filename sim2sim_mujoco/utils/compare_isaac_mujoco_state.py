import argparse

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import isaacgym  # noqa: F401  # Isaac Gym must be imported before torch.
import mujoco
import numpy as np
import torch

from go1_gym.envs.automatic import HistoryWrapper
from go1_gym.envs.automatic import KeyboardWrapper
from go1_gym.utils.global_switch import global_switch
from sim2sim_mujoco.utils.mujoco_obs_probe import (
    DEFAULT_CKPTID,
    DEFAULT_DOF_POS_20,
    DEFAULT_LOGDIR,
    MODEL_PATH,
    POLICY_QPOS_ADDR_20,
    POLICY_QVEL_ADDR_20,
    ROOT,
    extract_mujoco_state,
    initialize_standing_pose,
    pd_hold_default,
    restore_cfg_from_run,
)


def quat_wxyz_to_roll_pitch(quat):
    w, x, y, z = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    return roll, pitch


def extract_isaac_state(env):
    root = env.root_states[0].detach().cpu().numpy()
    quat_xyzw = root[3:7].copy()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    roll, pitch = quat_wxyz_to_roll_pitch(quat_wxyz)
    return {
        "base_pos": root[0:3].copy(),
        "base_quat_wxyz": quat_wxyz,
        "base_lin_vel_body": env.base_lin_vel[0].detach().cpu().numpy().copy(),
        "base_ang_vel_body": env.base_ang_vel[0].detach().cpu().numpy().copy(),
        "projected_gravity": env.projected_gravity[0].detach().cpu().numpy().copy(),
        "roll": roll,
        "pitch": pitch,
        "q_policy": env.dof_pos[0, :20].detach().cpu().numpy().copy(),
        "qd_policy": env.dof_vel[0, :20].detach().cpu().numpy().copy(),
    }


def print_vec(name, vec):
    vec = np.asarray(vec).reshape(-1)
    print(f"{name}: {np.array2string(vec, precision=5, suppress_small=True)}")


def print_joint_rows(isaac_state, mujoco_state):
    print("joint-wise q/qd diffs")
    print("  idx  joint        q_isaac   q_mj   dq      qd_isaac  qd_mj   dqd")
    for i, name in enumerate([
        "FL_hip", "FL_thigh", "FL_calf",
        "FR_hip", "FR_thigh", "FR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
        "zarx_j1", "zarx_j2", "zarx_j3", "zarx_j4", "zarx_j5", "zarx_j6",
    ]):
        qi = isaac_state["q_policy"][i]
        qm = mujoco_state["q_policy"][i]
        qdi = isaac_state["qd_policy"][i]
        qdm = mujoco_state["qd_policy"][i]
        print(f"  {i:02d}   {name:<10} {qi: 8.4f} {qm: 8.4f} {qm-qi: 8.4f} {qdi: 9.4f} {qdm: 9.4f} {qdm-qdi: 9.4f}")


def get_isaac_state(args):
    global_switch.open_switch()
    cfg = restore_cfg_from_run(args.logdir)
    cfg.terrain.mesh_type = "plane"
    cfg.terrain.teleport_robots = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_gravity = False
    cfg.domain_rand.randomize_restitution = False
    cfg.domain_rand.randomize_motor_offset = False
    cfg.domain_rand.randomize_motor_strength = False
    cfg.domain_rand.randomize_friction_indep = False
    cfg.domain_rand.randomize_ground_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.randomize_Kd_factor = False
    cfg.domain_rand.randomize_Kp_factor = False
    cfg.domain_rand.randomize_joint_friction = False
    cfg.domain_rand.randomize_com_displacement = False
    cfg.domain_rand.randomize_end_effector_force = False
    cfg.env.num_recording_envs = 1
    cfg.env.num_envs = 1
    cfg.terrain.num_rows = 5
    cfg.terrain.num_cols = 5
    cfg.terrain.border_size = 0
    cfg.terrain.center_robots = True
    cfg.terrain.center_span = 1
    cfg.asset.render_sphere = True

    env = KeyboardWrapper(sim_device=args.sim_device, headless=True, cfg=cfg)
    env = HistoryWrapper(env)
    env.reset()
    if args.force_default_state:
        base_env = env.env
        env_ids = torch.tensor([0], dtype=torch.long, device=base_env.device)
        dof_pos = torch.tensor(DEFAULT_DOF_POS_20, dtype=torch.float, device=base_env.device).unsqueeze(0)
        base_state = torch.zeros(1, 13, dtype=torch.float, device=base_env.device)
        base_state[0, :3] = torch.tensor([0.0, 0.0, args.base_z], dtype=torch.float, device=base_env.device)
        base_state[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float, device=base_env.device)
        base_env.set_idx_pose(env_ids, dof_pos, base_state)
        base_env.gym.refresh_dof_state_tensor(base_env.sim)
        base_env.gym.refresh_actor_root_state_tensor(base_env.sim)
        base_env.gym.refresh_rigid_body_state_tensor(base_env.sim)
        base_env.base_pos[:] = base_env.root_states[:base_env.num_envs, 0:3]
        base_env.base_quat[:] = base_env.root_states[:base_env.num_envs, 3:7]
        base_env.base_lin_vel[:] = 0.0
        base_env.base_ang_vel[:] = 0.0
        base_env.projected_gravity[:] = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float, device=base_env.device)
    state = extract_isaac_state(env)
    return state


def get_mujoco_state(args):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    initialize_standing_pose(model, data, args.base_z)
    if not args.force_default_state:
        pd_hold_default(model, data, args.settle_steps)
    return extract_mujoco_state(data)


def main():
    parser = argparse.ArgumentParser(description="Compare Isaac Gym and MuJoCo states.")
    parser.add_argument("--logdir", type=str, default=DEFAULT_LOGDIR)
    parser.add_argument("--ckptid", type=int, default=DEFAULT_CKPTID)
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--base-z", type=float, default=0.50)
    parser.add_argument("--settle-steps", type=int, default=500)
    parser.add_argument("--force-default-state", action="store_true", default=True)
    parser.add_argument("--no-force-default-state", dest="force_default_state", action="store_false")
    args = parser.parse_args()

    isaac_state = get_isaac_state(args)
    mujoco_state = get_mujoco_state(args)

    print("Isaac state")
    print_vec("  base_pos", isaac_state["base_pos"])
    print_vec("  base_quat_wxyz", isaac_state["base_quat_wxyz"])
    print_vec("  base_lin_vel_body", isaac_state["base_lin_vel_body"])
    print_vec("  base_ang_vel_body", isaac_state["base_ang_vel_body"])
    print_vec("  projected_gravity", isaac_state["projected_gravity"])
    print("  roll/pitch:", isaac_state["roll"], isaac_state["pitch"])
    print_vec("  q_policy", isaac_state["q_policy"])
    print_vec("  qd_policy", isaac_state["qd_policy"])
    print()

    print("MuJoCo state")
    print_vec("  base_pos", mujoco_state["base_pos"])
    print_vec("  base_quat_wxyz", mujoco_state["base_quat_wxyz"])
    print_vec("  base_lin_vel_body", mujoco_state["base_lin_vel_body"])
    print_vec("  base_ang_vel_body", mujoco_state["base_ang_vel_body"])
    print_vec("  projected_gravity", mujoco_state["projected_gravity"])
    print("  roll/pitch:", mujoco_state["roll"], mujoco_state["pitch"])
    print_vec("  q_policy", mujoco_state["q_policy"])
    print_vec("  qd_policy", mujoco_state["qd_policy"])
    print()

    print("Diff summary (MuJoCo - Isaac)")
    for key in ["base_pos", "base_quat_wxyz", "base_lin_vel_body", "base_ang_vel_body", "projected_gravity", "q_policy", "qd_policy"]:
        diff = mujoco_state[key] - isaac_state[key]
        print(f"  {key:<20} max_abs={np.max(np.abs(diff)): .6f} mean_abs={np.mean(np.abs(diff)): .6f}")
    print("  roll/pitch          ", mujoco_state["roll"] - isaac_state["roll"], mujoco_state["pitch"] - isaac_state["pitch"])
    print_joint_rows(isaac_state, mujoco_state)


if __name__ == "__main__":
    main()
