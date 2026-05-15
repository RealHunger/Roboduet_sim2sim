import argparse
import pickle as pkl
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

SIM2SIM_ROOT = Path(__file__).resolve().parents[1]
ROOT = ensure_repo_root_on_path()

import isaacgym  # noqa: F401  # Isaac Gym must be imported before torch.
import mujoco
import numpy as np
import torch

from go1_gym.envs.automatic.legged_robot_config import Cfg
from scripts.load_policy import load_arm_policy, load_dog_policy


MODEL_PATH = SIM2SIM_ROOT / "resources/robots/arx5p2Go1_mujoco/mjcf/arx5p2Go1_mujoco.xml"
DEFAULT_LOGDIR = "runs/overnight_go1_512/dummy-sf8qzpe4_seed6218"
DEFAULT_CKPTID = 30800

POLICY_QPOS_ADDR_20 = np.array([
    10, 11, 12,
    7, 8, 9,
    16, 17, 18,
    13, 14, 15,
    19, 20, 21, 22, 23, 24, 25, 26,
])
POLICY_QVEL_ADDR_20 = np.array([
    9, 10, 11,
    6, 7, 8,
    15, 16, 17,
    12, 13, 14,
    18, 19, 20, 21, 22, 23, 24, 25,
])
DEFAULT_DOF_POS_20 = np.array([
    0.1, 0.8, -1.5,
    -0.1, 0.8, -1.5,
    0.1, 1.0, -1.5,
    -0.1, 1.0, -1.5,
    0.0, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0,
])

KP_18 = np.array([
    35, 35, 35,
    35, 35, 35,
    35, 35, 35,
    35, 35, 35,
    40, 70, 70, 25, 25, 25,
])
KD_18 = np.array([
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    3, 15, 15, 2, 2, 2,
])


def restore_cfg_from_run(logdir):
    params_path = ROOT / logdir / "parameters.pkl"
    with params_path.open("rb") as file:
        saved = pkl.load(file)

    cfg = saved["Cfg"]
    for key, value in cfg.items():
        if hasattr(Cfg, key):
            target = getattr(Cfg, key)
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    if (
                        isinstance(value2, dict)
                        and hasattr(target, key2)
                        and hasattr(getattr(target, key2), "__dict__")
                    ):
                        nested = getattr(target, key2)
                        for key3, value3 in value2.items():
                            setattr(nested, key3, value3)
                    else:
                        setattr(target, key2, value2)
            else:
                setattr(Cfg, key, value)
        elif not isinstance(value, dict):
            setattr(Cfg, key, value)

    return Cfg


def describe_tensor(name, tensor):
    print(f"{name} shape: {tuple(tensor.shape)}")
    print(f"{name} finite: {torch.isfinite(tensor).all().item()}")
    print(f"{name} min/max: {tensor.min().item():.6f} / {tensor.max().item():.6f}")


def quat_wxyz_to_matrix(quat):
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_wxyz_to_roll_pitch(quat):
    w, x, y, z = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    return roll, pitch


def initialize_standing_pose(model, data, base_z):
    assert model.nq == 27
    assert model.nv == 26
    assert model.nu == 18

    data.qpos[0:3] = [0.0, 0.0, base_z]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[POLICY_QPOS_ADDR_20] = DEFAULT_DOF_POS_20
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def pd_hold_default(model, data, steps):
    qpos_addr = POLICY_QPOS_ADDR_20[:18]
    qvel_addr = POLICY_QVEL_ADDR_20[:18]
    target_q = DEFAULT_DOF_POS_20[:18]
    ctrl_min = model.actuator_ctrlrange[:, 0]
    ctrl_max = model.actuator_ctrlrange[:, 1]

    for _ in range(steps):
        q = data.qpos[qpos_addr]
        qd = data.qvel[qvel_addr]
        leg_torque = KP_18[:12] * (target_q[:12] - q[:12]) - KD_18[:12] * qd[:12]
        data.ctrl[:12] = np.clip(leg_torque, ctrl_min[:12], ctrl_max[:12])
        data.ctrl[12:18] = np.clip(target_q[12:18], ctrl_min[12:18], ctrl_max[12:18])
        mujoco.mj_step(model, data)


def extract_mujoco_state(data):
    quat_wxyz = data.qpos[3:7].copy()
    rot = quat_wxyz_to_matrix(quat_wxyz)

    base_lin_vel_world = data.qvel[0:3].copy()
    base_ang_vel_world = data.qvel[3:6].copy()
    base_lin_vel_body = rot.T @ base_lin_vel_world
    base_ang_vel_body = rot.T @ base_ang_vel_world
    projected_gravity = rot.T @ np.array([0.0, 0.0, -1.0])
    roll, pitch = quat_wxyz_to_roll_pitch(quat_wxyz)

    return {
        "base_pos": data.qpos[0:3].copy(),
        "base_quat_wxyz": quat_wxyz,
        "base_lin_vel_body": base_lin_vel_body,
        "base_ang_vel_body": base_ang_vel_body,
        "projected_gravity": projected_gravity,
        "roll": roll,
        "pitch": pitch,
        "q_policy": data.qpos[POLICY_QPOS_ADDR_20].copy(),
        "qd_policy": data.qvel[POLICY_QVEL_ADDR_20].copy(),
    }


def build_arm_obs(cfg, state, prev_action_18, commands_arm_obs):
    arm_pos_error = state["q_policy"][12:18] - DEFAULT_DOF_POS_20[12:18]
    pieces = [
        arm_pos_error * cfg.obs_scales.dof_pos,
        prev_action_18[12:18],
        commands_arm_obs[:6],
        np.array([state["roll"], state["pitch"]]),
    ]
    obs = np.concatenate(pieces)
    assert obs.shape == (cfg.arm.arm_num_observations,)
    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)


def build_dog_obs(cfg, state, prev_action_18, commands_dog, commands_arm_obs, clock_inputs=None):
    commands_scale_dog = np.array([
        cfg.obs_scales.lin_vel,
        cfg.obs_scales.lin_vel,
        cfg.obs_scales.ang_vel,
        cfg.obs_scales.body_pitch_cmd,
        cfg.obs_scales.body_roll_cmd,
    ])[:cfg.dog.dog_num_commands]

    pieces = [
        state["projected_gravity"],
        (state["q_policy"][:12] - DEFAULT_DOF_POS_20[:12]) * cfg.obs_scales.dof_pos,
        state["qd_policy"][:12] * cfg.obs_scales.dof_vel,
        prev_action_18[:12],
        commands_dog[:5] * commands_scale_dog[:5],
        commands_arm_obs[:6],
        np.array([state["roll"], state["pitch"]]),
    ]
    if cfg.env.observe_vel:
        pieces = [
            state["base_lin_vel_body"] * cfg.obs_scales.lin_vel,
            state["base_ang_vel_body"] * cfg.obs_scales.ang_vel,
        ] + pieces
    if cfg.env.observe_clock_inputs:
        pieces.append(np.zeros(4) if clock_inputs is None else clock_inputs)
    obs = np.concatenate(pieces)
    assert obs.shape == (cfg.dog.dog_num_observations,)
    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Build RoboDuet observations from a standing MuJoCo state.")
    parser.add_argument("--logdir", type=str, default=DEFAULT_LOGDIR)
    parser.add_argument("--ckptid", type=int, default=DEFAULT_CKPTID)
    parser.add_argument("--base-z", type=float, default=0.50)
    parser.add_argument("--settle-steps", type=int, default=500)
    args = parser.parse_args()

    cfg = restore_cfg_from_run(args.logdir)
    ckpt_id = str(args.ckptid).zfill(6)
    dog_policy = load_dog_policy(str(ROOT / args.logdir), ckpt_id, cfg)
    arm_policy = load_arm_policy(str(ROOT / args.logdir), ckpt_id, cfg)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    initialize_standing_pose(model, data, args.base_z)
    pd_hold_default(model, data, args.settle_steps)

    state = extract_mujoco_state(data)
    prev_action_18 = np.zeros(18)
    commands_dog = np.zeros(cfg.dog.dog_num_commands)
    commands_arm_obs = np.zeros(cfg.arm.arm_num_commands)

    arm_obs_current = build_arm_obs(cfg, state, prev_action_18, commands_arm_obs)
    dog_obs_current = build_dog_obs(cfg, state, prev_action_18, commands_dog, commands_arm_obs)
    arm_obs_history = torch.zeros(1, cfg.arm.arm_num_obs_history)
    dog_obs_history = torch.zeros(1, cfg.dog.dog_num_obs_history)
    arm_obs_history = torch.cat([arm_obs_history[:, cfg.arm.arm_num_observations:], arm_obs_current], dim=-1)
    dog_obs_history = torch.cat([dog_obs_history[:, cfg.dog.dog_num_observations:], dog_obs_current], dim=-1)

    arm_obs = {
        "obs": arm_obs_current,
        "privileged_obs": torch.zeros(1, cfg.arm.arm_num_privileged_obs),
        "obs_history": arm_obs_history,
    }
    dog_obs = {
        "obs": dog_obs_current,
        "privileged_obs": torch.zeros(1, cfg.dog.dog_num_privileged_obs),
        "obs_history": dog_obs_history,
    }

    print("MuJoCo state summary")
    print("  base_pos:", state["base_pos"])
    print("  base_quat_wxyz:", state["base_quat_wxyz"])
    print("  projected_gravity:", state["projected_gravity"])
    print("  roll/pitch:", state["roll"], state["pitch"])
    print("  max abs q error:", np.max(np.abs(state["q_policy"] - DEFAULT_DOF_POS_20)))
    print("  max abs qd:", np.max(np.abs(state["qd_policy"])))

    print("Observation tensors")
    describe_tensor("arm_obs['obs']", arm_obs["obs"])
    describe_tensor("arm_obs['obs_history']", arm_obs["obs_history"])
    describe_tensor("dog_obs['obs']", dog_obs["obs"])
    describe_tensor("dog_obs['obs_history']", dog_obs["obs_history"])

    with torch.no_grad():
        actions_arm = arm_policy(arm_obs)
        actions_dog = dog_policy(dog_obs)

    print("Policy output from MuJoCo observations")
    describe_tensor("actions_arm", actions_arm)
    describe_tensor("actions_dog", actions_dog)
    assert actions_arm.shape == (1, cfg.arm.num_actions_arm_cd)
    assert actions_dog.shape == (1, cfg.dog.dog_actions)
    assert torch.isfinite(actions_arm).all()
    assert torch.isfinite(actions_dog).all()
    print("MuJoCo obs probe passed.")


if __name__ == "__main__":
    main()
