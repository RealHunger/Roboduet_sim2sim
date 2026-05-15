import argparse
import csv
import time

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import isaacgym  # noqa: F401
import mujoco
import mujoco.viewer
import numpy as np
import torch

from scripts.load_policy import load_arm_policy, load_dog_policy
from sim2sim_mujoco.utils.mujoco_obs_probe import DEFAULT_CKPTID, DEFAULT_LOGDIR, DEFAULT_DOF_POS_20, MODEL_PATH, ROOT, extract_mujoco_state, initialize_standing_pose, restore_cfg_from_run
from sim2sim_mujoco.scripts.mujoco_policy_rollout import (
    apply_pd_control,
    arm_grasper_world_pos,
    arm_target_world_pos,
    compute_clock_inputs,
    rollout_step,
    tune_arm_servo,
    update_arm_ee_marker,
    update_arm_target_marker,
)
from sim2sim_mujoco.utils.sim2sim_sequence import arm_obs_from_command, metric_header, metric_row, scripted_command


def main():
    parser = argparse.ArgumentParser(description="Replay scripted sim2sim sequence in MuJoCo.")
    parser.add_argument("--logdir", type=str, default=DEFAULT_LOGDIR)
    parser.add_argument("--ckptid", type=int, default=DEFAULT_CKPTID)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--base-z", type=float, default=0.50)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--leg-kd-scale", type=float, default=1.2)
    parser.add_argument("--arm-kp-scale", type=float, default=1.0)
    parser.add_argument("--arm-damping-scale", type=float, default=1.0)
    parser.add_argument("--arm-target-alpha", type=float, default=0.25)
    parser.add_argument("--output", type=str, default="/tmp/mujoco_sequence.csv")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--loop-sequence", action="store_true")
    args = parser.parse_args()

    cfg = restore_cfg_from_run(args.logdir)
    ckpt_id = str(args.ckptid).zfill(6)
    dog_policy = load_dog_policy(str(ROOT / args.logdir), ckpt_id, cfg)
    arm_policy = load_arm_policy(str(ROOT / args.logdir), ckpt_id, cfg)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    tune_arm_servo(model, args.arm_kp_scale, args.arm_damping_scale)
    data = mujoco.MjData(model)
    initialize_standing_pose(model, data, args.base_z)

    command_state = {"l_cmd": 0.5, "p_cmd": 0.2, "y_cmd": 0.0, "roll_cmd": 0.1, "pitch_cmd": 0.5, "arm_yaw_cmd": 0.0}
    buffers = {
        "prev_action_18": np.zeros(18),
        "commands_dog": np.zeros(cfg.dog.dog_num_commands),
        "commands_arm_obs": arm_obs_from_command(np.array(list(command_state.values()))),
        "arm_obs_history": torch.zeros(1, cfg.arm.arm_num_obs_history),
        "dog_obs_history": torch.zeros(1, cfg.dog.dog_num_obs_history),
        "hold_arm_default": False,
        "arm_action_limit": None,
        "arm_target_alpha": args.arm_target_alpha,
        "prev_arm_target_q": DEFAULT_DOF_POS_20[12:18].copy(),
        "arm_ik_control": False,
        "arm_ik_step_size": 0.04,
        "arm_ik_damping": 0.08,
        "arm_ik_max_dq": 0.08,
        "arm_ik_iterations": 8,
        "command_state": command_state,
        "hold_dog_when_idle": True,
        "idle_command_threshold": 0.05,
        "gait_index": 0.0,
        "clock_inputs": np.zeros(4),
    }

    for _ in range(args.warmup_steps):
        apply_pd_control(model, data, DEFAULT_DOF_POS_20[:18], args.leg_kd_scale)
        mujoco.mj_step(model, data)

    rows = []
    dt_policy = model.opt.timestep * cfg.control.decimation

    viewer_context = mujoco.viewer.launch_passive(model, data) if args.viewer else None
    try:
        viewer = viewer_context.__enter__() if viewer_context is not None else None
        for step in range(args.steps):
            t = step * dt_policy
            dog_cmd, arm_cmd = scripted_command(t, loop=args.loop_sequence)
            command_state.update({
                "l_cmd": arm_cmd[0],
                "p_cmd": arm_cmd[1],
                "y_cmd": arm_cmd[2],
                "roll_cmd": arm_cmd[3],
                "pitch_cmd": arm_cmd[4],
                "arm_yaw_cmd": arm_cmd[5],
            })
            buffers["commands_dog"][:3] = dog_cmd
            buffers["commands_arm_obs"] = arm_obs_from_command(arm_cmd)
            update_arm_target_marker(model, data, command_state)
            update_arm_ee_marker(model, data)
            buffers["gait_index"] = (buffers["gait_index"] + dt_policy * 3.0) % 1.0
            buffers["clock_inputs"] = compute_clock_inputs(buffers["gait_index"], buffers["commands_dog"])

            state, dog_action, arm_action, action_18, target_q = rollout_step(cfg, model, data, dog_policy, arm_policy, buffers, False, False, False)
            for _ in range(cfg.control.decimation):
                apply_pd_control(model, data, target_q, args.leg_kd_scale)
                mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()
                if args.realtime:
                    time.sleep(dt_policy)

            arm_target = arm_target_world_pos(data, command_state)
            arm_ee = arm_grasper_world_pos(model, data)
            row = metric_row(t, dog_cmd, arm_cmd, data.qpos[0:3].copy(), state["base_lin_vel_body"][0], state["roll"], state["pitch"], arm_target, arm_ee, float(np.max(np.abs(data.ctrl))))
            rows.append(row)
            if step % args.print_every == 0:
                print(f"{step:04d} t={t:5.2f} cmd=({dog_cmd[0]:.2f},{dog_cmd[2]:.2f}) x={row[7]:.3f} vx={row[10]:.3f} arm_dist={row[19]:.3f} max_ctrl={row[20]:.1f}")
    finally:
        if viewer_context is not None:
            viewer_context.__exit__(None, None, None)

    with open(args.output, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(metric_header())
        writer.writerows(rows)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
