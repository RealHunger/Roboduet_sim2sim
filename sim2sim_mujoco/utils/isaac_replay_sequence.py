import argparse
import csv
import time

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import isaacgym  # noqa: F401
import numpy as np
import torch
from isaacgym import gymapi

from go1_gym.envs.automatic import HistoryWrapper, KeyboardWrapper
from go1_gym.utils.global_switch import global_switch
from sim2sim_mujoco.utils.compare_isaac_mujoco_state import quat_wxyz_to_roll_pitch
from scripts.load_policy import load_arm_policy, load_dog_policy
from sim2sim_mujoco.utils.mujoco_obs_probe import DEFAULT_CKPTID, DEFAULT_LOGDIR, ROOT, restore_cfg_from_run
from sim2sim_mujoco.utils.sim2sim_sequence import metric_header, metric_row, scripted_command


def configure_cfg(logdir):
    cfg = restore_cfg_from_run(logdir)
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
    cfg.env.episode_length_s = 100000
    cfg.rewards.use_terminal_body_height = False
    cfg.rewards.use_terminal_roll = False
    cfg.rewards.use_terminal_pitch = False
    cfg.rewards.use_terminal_roll_pitch = False
    cfg.hybrid.rewards.use_terminal_roll = False
    cfg.hybrid.rewards.use_terminal_pitch = False
    return cfg


def isaac_arm_target(base_env):
    x, y, z = base_env.lpy_to_world_xyz()
    return np.array([x.item(), y.item(), z.item()])


def isaac_grasper(base_env):
    ee_state = base_env.end_effector_state[0].detach().cpu().numpy()
    # Approximate Isaac draw point; exact value is refreshed by env post-physics each step.
    return ee_state[:3].copy()


def main():
    parser = argparse.ArgumentParser(description="Replay scripted sim2sim sequence in Isaac Gym.")
    parser.add_argument("--logdir", type=str, default=DEFAULT_LOGDIR)
    parser.add_argument("--ckptid", type=int, default=DEFAULT_CKPTID)
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=str, default="/tmp/isaac_sequence.csv")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--allow-resets", action="store_true")
    parser.add_argument("--loop-sequence", action="store_true")
    args = parser.parse_args()

    global_switch.open_switch()
    cfg = configure_cfg(args.logdir)
    env = KeyboardWrapper(sim_device=args.sim_device, headless=args.headless, cfg=cfg)
    env = HistoryWrapper(env)
    dog_policy = load_dog_policy(str(ROOT / args.logdir), str(args.ckptid).zfill(6), cfg)
    arm_policy = load_arm_policy(str(ROOT / args.logdir), str(args.ckptid).zfill(6), cfg)
    env.reset()
    base_env = env.env
    if not args.headless:
        base_env.enable_viewer_sync = True
        base_env.gym.viewer_camera_look_at(
            base_env.viewer,
            base_env.envs[0],
            gymapi.Vec3(1.8, 1.2, 1.0),
            gymapi.Vec3(0.0, 0.0, 0.35),
        )

    rows = []
    dt_policy = base_env.dt
    for step in range(args.steps):
        t = step * dt_policy
        dog_cmd, arm_cmd = scripted_command(t, loop=args.loop_sequence)
        base_env.commands_dog[0, :3] = torch.tensor(dog_cmd, dtype=torch.float, device=base_env.device)
        base_env.commands_arm[0, :6] = torch.tensor(arm_cmd, dtype=torch.float, device=base_env.device)
        base_env.update_arm_commands()

        with torch.no_grad():
            arm_obs = env.get_arm_observations()
            arm_action = arm_policy(arm_obs)
            base_env.plan(arm_action[..., -2:])
            dog_obs = env.get_dog_observations()
            dog_action = dog_policy(dog_obs)
        env.step(dog_action, arm_action[..., :-2])
        if not args.allow_resets:
            base_env.reset_buf[:] = False
            base_env.time_out_buf[:] = False
            base_env.episode_length_buf[:] = torch.minimum(
                base_env.episode_length_buf,
                torch.ones_like(base_env.episode_length_buf) * 100,
            )
        if not args.headless:
            base_env.render_gui(sync_frame_time=args.realtime)
            if args.realtime:
                time.sleep(dt_policy)

        root = base_env.root_states[0].detach().cpu().numpy()
        quat_wxyz = root[3:7][[3, 0, 1, 2]]
        roll, pitch = quat_wxyz_to_roll_pitch(quat_wxyz)
        arm_target = isaac_arm_target(base_env)
        arm_ee = isaac_grasper(base_env)
        max_ctrl = float(torch.max(torch.abs(base_env.torques[0])).detach().cpu().item()) if hasattr(base_env, "torques") else np.nan
        row = metric_row(t, dog_cmd, arm_cmd, root[:3], base_env.base_lin_vel[0, 0].detach().cpu().item(), roll, pitch, arm_target, arm_ee, max_ctrl)
        rows.append(row)
        if step % args.print_every == 0:
            print(f"{step:04d} t={t:5.2f} cmd=({dog_cmd[0]:.2f},{dog_cmd[2]:.2f}) x={row[7]:.3f} vx={row[10]:.3f} arm_dist={row[19]:.3f} max_ctrl={row[20]:.1f}")

    with open(args.output, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(metric_header())
        writer.writerows(rows)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
