import argparse

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import isaacgym  # noqa: F401  # Isaac Gym must be imported before torch.
import mujoco
import numpy as np
import torch

from go1_gym.envs.automatic import KeyboardWrapper
from go1_gym.envs.automatic import HistoryWrapper
from go1_gym.utils.global_switch import global_switch
from scripts.load_policy import load_dog_policy
from sim2sim_mujoco.utils.mujoco_obs_probe import (
    DEFAULT_LOGDIR,
    DEFAULT_CKPTID,
    DEFAULT_DOF_POS_20,
    MODEL_PATH,
    ROOT,
    build_dog_obs,
    extract_mujoco_state,
    initialize_standing_pose,
    restore_cfg_from_run,
)
from sim2sim_mujoco.scripts.mujoco_policy_rollout import apply_pd_control


DOG_SEGMENTS = [
    ("projected_gravity", 3),
    ("leg_q_error", 12),
    ("leg_qd", 12),
    ("prev_dog_action", 12),
    ("dog_command", 5),
    ("arm_command_obs", 6),
    ("roll_pitch", 2),
]


def maybe_prepend_vel(cfg, obs):
    if cfg.env.observe_vel:
        return obs
    return obs


def print_segmented_obs(title, obs):
    obs = np.asarray(obs).reshape(-1)
    print(title)
    offset = 0
    for name, width in DOG_SEGMENTS:
        values = obs[offset:offset + width]
        formatted = np.array2string(values, precision=4, suppress_small=True)
        print(
            f"  idx {offset:02d}:{offset + width:02d} {name:<20} "
            f"min={values.min(): .4f} max={values.max(): .4f} values={formatted}"
        )
        offset += width
    print("  total dim", offset)


def print_action(title, action):
    action = np.asarray(action).reshape(-1)
    print(
        f"{title} shape={action.shape} "
        f"min={action.min(): .4f} max={action.max(): .4f} "
        f"values={np.array2string(action, precision=4, suppress_small=True)}"
    )


def get_isaac_dog_obs_and_action(args):
    global_switch.open_switch()
    cfg = restore_cfg_from_run(args.logdir)
    cfg.terrain.mesh_type = "plane"
    cfg.terrain.teleport_robots = False
    cfg.domain_rand.push_robots = False
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
    cfg.env.episode_length_s = 10000
    cfg.commands.resampling_time = 10000
    cfg.rewards.use_terminal_body_height = False
    cfg.rewards.use_terminal_roll = False
    cfg.rewards.use_terminal_pitch = False
    cfg.hybrid.rewards.use_terminal_body_height = False
    cfg.hybrid.rewards.use_terminal_roll = False
    cfg.hybrid.rewards.use_terminal_pitch = False
    cfg.arm.commands.T_traj = [20000, 30000]

    env = KeyboardWrapper(sim_device=args.sim_device, headless=True, cfg=cfg)
    env = HistoryWrapper(env)
    dog_policy = load_dog_policy(str(ROOT / args.logdir), str(args.ckptid).zfill(6), cfg)

    env.reset()
    env.commands_dog[:, :] = 0.0
    env.commands_arm[:, :] = 0.0
    env.commands_arm_obs[:, :] = 0.0
    env.clock_inputs[:, :] = 0.0

    dog_obs_dict = env.get_dog_observations()
    with torch.no_grad():
        action = dog_policy(dog_obs_dict)

    obs = dog_obs_dict["obs"].detach().cpu().numpy()[0]
    history = dog_obs_dict["obs_history"].detach().cpu().numpy()[0]
    action_np = action.detach().cpu().numpy()[0]
    return cfg, obs, history, action_np


def get_mujoco_dog_obs_and_action(args, cfg):
    dog_policy = load_dog_policy(str(ROOT / args.logdir), str(args.ckptid).zfill(6), cfg)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    initialize_standing_pose(model, data, args.base_z)

    for _ in range(args.warmup_steps):
        apply_pd_control(model, data, DEFAULT_DOF_POS_20[:18])
        mujoco.mj_step(model, data)

    state = extract_mujoco_state(data)
    prev_action_18 = np.zeros(18)
    commands_dog = np.zeros(cfg.dog.dog_num_commands)
    commands_arm_obs = np.zeros(cfg.arm.arm_num_commands)
    obs_t = build_dog_obs(cfg, state, prev_action_18, commands_dog, commands_arm_obs)
    history_t = obs_t.repeat(1, cfg.dog.dog_num_obs_history // cfg.dog.dog_num_observations)
    dog_obs_dict = {
        "obs": obs_t,
        "privileged_obs": torch.zeros(1, cfg.dog.dog_num_privileged_obs),
        "obs_history": history_t,
    }
    with torch.no_grad():
        action = dog_policy(dog_obs_dict)

    return obs_t.numpy()[0], history_t.numpy()[0], action.numpy()[0]


def compare_obs(isaac_obs, mujoco_obs):
    diff = mujoco_obs - isaac_obs
    print("MuJoCo - Isaac obs diff summary")
    print("  max abs diff", np.max(np.abs(diff)))
    print("  mean abs diff", np.mean(np.abs(diff)))
    offset = 0
    for name, width in DOG_SEGMENTS:
        values = diff[offset:offset + width]
        print(
            f"  idx {offset:02d}:{offset + width:02d} {name:<20} "
            f"max_abs={np.max(np.abs(values)): .4f} mean_abs={np.mean(np.abs(values)): .4f}"
        )
        offset += width


def main():
    parser = argparse.ArgumentParser(description="Compare Isaac Gym and MuJoCo dog observations.")
    parser.add_argument("--logdir", type=str, default=DEFAULT_LOGDIR)
    parser.add_argument("--ckptid", type=int, default=DEFAULT_CKPTID)
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--base-z", type=float, default=0.50)
    parser.add_argument("--warmup-steps", type=int, default=500)
    args = parser.parse_args()

    cfg = restore_cfg_from_run(args.logdir)
    print("Loading Isaac Gym observation")
    _, isaac_obs, isaac_history, isaac_action = get_isaac_dog_obs_and_action(args)
    print("Loading MuJoCo observation")
    mujoco_obs, mujoco_history, mujoco_action = get_mujoco_dog_obs_and_action(args, cfg)

    print_segmented_obs("Isaac dog obs", isaac_obs)
    print_action("Isaac dog action", isaac_action)
    print("Isaac dog history min/max", isaac_history.min(), isaac_history.max())
    print()

    print_segmented_obs("MuJoCo dog obs", mujoco_obs)
    print_action("MuJoCo dog action", mujoco_action)
    print("MuJoCo dog history min/max", mujoco_history.min(), mujoco_history.max())
    print()

    compare_obs(isaac_obs, mujoco_obs)
    print_action("Action diff MuJoCo - Isaac", mujoco_action - isaac_action)


if __name__ == "__main__":
    main()
