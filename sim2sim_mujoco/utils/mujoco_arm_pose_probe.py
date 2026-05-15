import argparse

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import isaacgym  # noqa: F401  # Isaac Gym must be imported before torch.
import mujoco
import numpy as np

from scripts.load_policy import load_arm_policy, load_dog_policy
from sim2sim_mujoco.utils.mujoco_obs_probe import (
    DEFAULT_CKPTID,
    DEFAULT_DOF_POS_20,
    DEFAULT_LOGDIR,
    MODEL_PATH,
    POLICY_QPOS_ADDR_20,
    ROOT,
    extract_mujoco_state,
    initialize_standing_pose,
    restore_cfg_from_run,
)
from sim2sim_mujoco.scripts.mujoco_policy_rollout import (
    apply_pd_control,
    compute_clock_inputs,
    default_arm_command_obs,
    rollout_step,
    tune_arm_servo,
)


ARM_JOINT_NAMES = ["zarx_j1", "zarx_j2", "zarx_j3", "zarx_j4", "zarx_j5", "zarx_j6"]


def body_pos(model, data, name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[body_id].copy()


def main():
    parser = argparse.ArgumentParser(description="Probe MuJoCo arm pose tracking under RoboDuet policy.")
    parser.add_argument("--logdir", type=str, default=DEFAULT_LOGDIR)
    parser.add_argument("--ckptid", type=int, default=DEFAULT_CKPTID)
    parser.add_argument("--base-z", type=float, default=0.50)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--x-vel", type=float, default=0.0)
    parser.add_argument("--arm-action-limit", type=float, default=1.0)
    parser.add_argument("--arm-kp-scale", type=float, default=1.0)
    parser.add_argument("--arm-damping-scale", type=float, default=0.5)
    parser.add_argument("--leg-kd-scale", type=float, default=1.2)
    parser.add_argument("--hold-dog-default", action="store_true")
    args = parser.parse_args()

    cfg = restore_cfg_from_run(args.logdir)
    ckpt_id = str(args.ckptid).zfill(6)
    dog_policy = load_dog_policy(str(ROOT / args.logdir), ckpt_id, cfg)
    arm_policy = load_arm_policy(str(ROOT / args.logdir), ckpt_id, cfg)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    tune_arm_servo(model, args.arm_kp_scale, args.arm_damping_scale)
    data = mujoco.MjData(model)
    initialize_standing_pose(model, data, args.base_z)

    buffers = {
        "prev_action_18": np.zeros(18),
        "commands_dog": np.zeros(cfg.dog.dog_num_commands),
        "commands_arm_obs": default_arm_command_obs(),
        "arm_obs_history": __import__("torch").zeros(1, cfg.arm.arm_num_obs_history),
        "dog_obs_history": __import__("torch").zeros(1, cfg.dog.dog_num_obs_history),
        "hold_arm_default": False,
        "arm_action_limit": args.arm_action_limit,
        "gait_index": 0.0,
        "clock_inputs": np.zeros(4),
    }

    for _ in range(args.warmup_steps):
        apply_pd_control(model, data, DEFAULT_DOF_POS_20[:18], args.leg_kd_scale)
        mujoco.mj_step(model, data)

    ee_positions = []
    max_arm_err = 0.0
    max_arm_vel = 0.0
    max_arm_action = 0.0

    for step in range(args.steps):
        buffers["commands_dog"][0] = args.x_vel
        buffers["gait_index"] = (buffers["gait_index"] + model.opt.timestep * cfg.control.decimation * 3.0) % 1.0
        buffers["clock_inputs"] = compute_clock_inputs(buffers["gait_index"], buffers["commands_dog"])

        state, dog_action, arm_action, action_18, target_q = rollout_step(
            cfg,
            model,
            data,
            dog_policy,
            arm_policy,
            buffers,
            False,
            args.hold_dog_default,
            False,
        )
        if args.hold_dog_default:
            action_18[:12] = 0.0
            target_q = DEFAULT_DOF_POS_20[:18].copy()
            target_q[12:18] = DEFAULT_DOF_POS_20[12:18] + np.clip(action_18[12:18], -args.arm_action_limit, args.arm_action_limit) * cfg.control.action_scale

        for _ in range(cfg.control.decimation):
            apply_pd_control(model, data, target_q, args.leg_kd_scale)
            mujoco.mj_step(model, data)

        q = data.qpos[POLICY_QPOS_ADDR_20[:18]]
        qd = data.qvel[[18, 19, 20, 21, 22, 23]]
        arm_err = target_q[12:18] - q[12:18]
        max_arm_err = max(max_arm_err, float(np.max(np.abs(arm_err))))
        max_arm_vel = max(max_arm_vel, float(np.max(np.abs(qd))))
        max_arm_action = max(max_arm_action, float(np.max(np.abs(arm_action[:6]))))
        ee_pos = 0.5 * (body_pos(model, data, "zarx_body7") + body_pos(model, data, "zarx_body8"))
        ee_positions.append(ee_pos)

        if step % args.print_every == 0:
            print(
                step,
                "arm_action", np.array2string(arm_action[:6], precision=3, suppress_small=True),
                "target_q", np.array2string(target_q[12:18], precision=3, suppress_small=True),
                "q", np.array2string(q[12:18], precision=3, suppress_small=True),
                "err_max", f"{np.max(np.abs(arm_err)):.4f}",
                "ee", np.array2string(ee_pos, precision=3, suppress_small=True),
            )

    ee_positions = np.asarray(ee_positions)
    print("Summary")
    print("  max_arm_action", max_arm_action)
    print("  max_arm_err", max_arm_err)
    print("  max_arm_vel", max_arm_vel)
    print("  ee_min", np.array2string(ee_positions.min(axis=0), precision=4, suppress_small=True))
    print("  ee_max", np.array2string(ee_positions.max(axis=0), precision=4, suppress_small=True))
    print("  ee_range", np.array2string(ee_positions.ptp(axis=0), precision=4, suppress_small=True))


if __name__ == "__main__":
    main()
