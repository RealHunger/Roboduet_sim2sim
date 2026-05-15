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
from scripts.load_policy import load_arm_policy
from sim2sim_mujoco.utils.mujoco_obs_probe import (
    DEFAULT_CKPTID,
    DEFAULT_DOF_POS_20,
    DEFAULT_LOGDIR,
    MODEL_PATH,
    ROOT,
    build_arm_obs,
    extract_mujoco_state,
    initialize_standing_pose,
    restore_cfg_from_run,
)
from sim2sim_mujoco.scripts.mujoco_policy_rollout import default_arm_command_obs


ARM_SEGMENTS = [
    ("arm_q_error", 6),
    ("prev_arm_action", 6),
    ("arm_command_obs", 6),
    ("roll_pitch", 2),
]


def configure_cfg(args):
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
    return cfg


def force_isaac_default_state(env, args):
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


def print_segmented(title, values, segments):
    values = np.asarray(values).reshape(-1)
    print(title)
    offset = 0
    for name, width in segments:
        segment = values[offset:offset + width]
        print(
            f"  idx {offset:02d}:{offset + width:02d} {name:<18} "
            f"min={segment.min(): .5f} max={segment.max(): .5f} "
            f"values={np.array2string(segment, precision=5, suppress_small=True)}"
        )
        offset += width
    print("  total dim", offset)


def print_action(title, action):
    action = np.asarray(action).reshape(-1)
    print(
        f"{title} shape={action.shape} min={action.min(): .5f} max={action.max(): .5f} "
        f"values={np.array2string(action, precision=5, suppress_small=True)}"
    )


def get_isaac_arm_obs_action(args, cfg):
    global_switch.open_switch()
    env = KeyboardWrapper(sim_device=args.sim_device, headless=True, cfg=cfg)
    env = HistoryWrapper(env)
    policy = load_arm_policy(str(ROOT / args.logdir), str(args.ckptid).zfill(6), cfg)

    env.reset()
    force_isaac_default_state(env, args)
    base_env = env.env
    base_env.actions[:, :] = 0.0
    base_env.commands_arm[:, :] = torch.tensor(
        [[0.5, 0.2, 0.0, 0.1, 0.5, 0.0]], dtype=torch.float, device=base_env.device
    )
    base_env.update_arm_commands()

    obs_dict = env.get_arm_observations()
    with torch.no_grad():
        action = policy(obs_dict)
    return (
        obs_dict["obs"].detach().cpu().numpy()[0],
        obs_dict["obs_history"].detach().cpu().numpy()[0],
        action.detach().cpu().numpy()[0],
        base_env.commands_arm_obs.detach().cpu().numpy()[0].copy(),
    )


def get_mujoco_arm_obs_action(args, cfg, obs_history=None):
    policy = load_arm_policy(str(ROOT / args.logdir), str(args.ckptid).zfill(6), cfg)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    initialize_standing_pose(model, data, args.base_z)
    state = extract_mujoco_state(data)
    prev_action_18 = np.zeros(18)
    commands_arm_obs = default_arm_command_obs()
    obs_t = build_arm_obs(cfg, state, prev_action_18, commands_arm_obs)
    if obs_history is None:
        history_t = obs_t.repeat(1, cfg.arm.arm_num_obs_history // cfg.arm.arm_num_observations)
    else:
        history_t = torch.tensor(obs_history, dtype=torch.float32).unsqueeze(0)
    obs_dict = {
        "obs": obs_t,
        "privileged_obs": torch.zeros(1, cfg.arm.arm_num_privileged_obs),
        "obs_history": history_t,
    }
    with torch.no_grad():
        action = policy(obs_dict)
    return obs_t.numpy()[0], history_t.numpy()[0], action.numpy()[0], commands_arm_obs.copy()


def main():
    parser = argparse.ArgumentParser(description="Compare Isaac Gym and MuJoCo arm observations/actions.")
    parser.add_argument("--logdir", type=str, default=DEFAULT_LOGDIR)
    parser.add_argument("--ckptid", type=int, default=DEFAULT_CKPTID)
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--base-z", type=float, default=0.50)
    args = parser.parse_args()

    cfg = configure_cfg(args)
    isaac_obs, isaac_history, isaac_action, isaac_cmd = get_isaac_arm_obs_action(args, cfg)
    mujoco_obs, mujoco_history, mujoco_action, mujoco_cmd = get_mujoco_arm_obs_action(args, cfg)
    _, matched_history, matched_action, _ = get_mujoco_arm_obs_action(args, cfg, obs_history=isaac_history)

    print_segmented("Isaac arm obs", isaac_obs, ARM_SEGMENTS)
    print_action("Isaac arm action", isaac_action)
    print("Isaac arm obs_history min/max", isaac_history.min(), isaac_history.max())
    print("Isaac commands_arm_obs", np.array2string(isaac_cmd, precision=5, suppress_small=True))
    print()

    print_segmented("MuJoCo arm obs", mujoco_obs, ARM_SEGMENTS)
    print_action("MuJoCo arm action", mujoco_action)
    print("MuJoCo arm obs_history min/max", mujoco_history.min(), mujoco_history.max())
    print("MuJoCo commands_arm_obs", np.array2string(mujoco_cmd, precision=5, suppress_small=True))
    print_action("MuJoCo arm action with Isaac history", matched_action)
    print()

    obs_diff = mujoco_obs - isaac_obs
    action_diff = mujoco_action - isaac_action
    print("Diff summary (MuJoCo - Isaac)")
    print("  obs max_abs", np.max(np.abs(obs_diff)), "mean_abs", np.mean(np.abs(obs_diff)))
    print("  history max_abs", np.max(np.abs(mujoco_history - isaac_history)), "mean_abs", np.mean(np.abs(mujoco_history - isaac_history)))
    print_action("Action diff", action_diff)
    print_action("Action diff with Isaac history", matched_action - isaac_action)


if __name__ == "__main__":
    main()
