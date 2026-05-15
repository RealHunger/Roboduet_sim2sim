import argparse
import itertools

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


def parse_floats(text):
    return [float(x) for x in text.split(",") if x]


def set_contact_params(model, friction, solref_timeconst):
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        is_ground = name == "ground"
        is_foot_sphere = model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE and np.isclose(model.geom_size[geom_id, 0], 0.02)
        if not (is_ground or is_foot_sphere):
            continue
        model.geom_friction[geom_id] = [friction, 0.1, 0.01]
        model.geom_condim[geom_id] = 6
        model.geom_solref[geom_id] = [solref_timeconst, 1.0]
        model.geom_solimp[geom_id] = [0.9, 0.95, 0.001, 0.5, 2.0]


def run_case(args, cfg, dog_policy, arm_policy, leg_kd_scale, friction, solref_timeconst):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    tune_arm_servo(model, args.arm_kp_scale, args.arm_damping_scale)
    set_contact_params(model, friction, solref_timeconst)
    data = mujoco.MjData(model)
    initialize_standing_pose(model, data, args.base_z)

    buffers = {
        "prev_action_18": np.zeros(18),
        "commands_dog": np.zeros(cfg.dog.dog_num_commands),
        "commands_arm_obs": default_arm_command_obs(),
        "arm_obs_history": __import__("torch").zeros(1, cfg.arm.arm_num_obs_history),
        "dog_obs_history": __import__("torch").zeros(1, cfg.dog.dog_num_obs_history),
        "hold_arm_default": args.hold_arm_default,
        "arm_action_limit": args.arm_action_limit,
        "gait_index": 0.0,
        "clock_inputs": np.zeros(4),
    }

    for _ in range(args.warmup_steps):
        apply_pd_control(model, data, DEFAULT_DOF_POS_20[:18], leg_kd_scale)
        mujoco.mj_step(model, data)

    x0 = float(data.qpos[0])
    max_abs_roll = 0.0
    max_abs_pitch = 0.0
    max_ctrl = 0.0
    saturated_steps = 0
    vx_sum = 0.0
    fell = False

    target_q = DEFAULT_DOF_POS_20[:18]
    for step in range(args.steps):
        ramp = 1.0 if args.ramp_time <= 0 else min(1.0, step * model.opt.timestep * cfg.control.decimation / args.ramp_time)
        buffers["commands_dog"][0] = args.x_vel * ramp
        buffers["commands_dog"][1] = args.y_vel * ramp
        buffers["commands_dog"][2] = args.yaw_vel * ramp
        buffers["gait_index"] = (buffers["gait_index"] + model.opt.timestep * cfg.control.decimation * 3.0) % 1.0
        buffers["clock_inputs"] = compute_clock_inputs(buffers["gait_index"], buffers["commands_dog"])

        state, _, _, _, target_q = rollout_step(
            cfg,
            model,
            data,
            dog_policy,
            arm_policy,
            buffers,
            args.disable_arm_policy,
            False,
            False,
        )
        for _ in range(cfg.control.decimation):
            apply_pd_control(model, data, target_q, leg_kd_scale)
            mujoco.mj_step(model, data)

        ctrl_abs = float(np.max(np.abs(data.ctrl[:12])))
        max_ctrl = max(max_ctrl, ctrl_abs)
        saturated_steps += int(ctrl_abs >= 23.6)
        max_abs_roll = max(max_abs_roll, abs(float(state["roll"])))
        max_abs_pitch = max(max_abs_pitch, abs(float(state["pitch"])))
        vx_sum += float(state["base_lin_vel_body"][0])

        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or data.qpos[2] < args.min_base_z:
            fell = True
            break

    ran_steps = step + 1
    duration = ran_steps * model.opt.timestep * cfg.control.decimation
    dx = float(data.qpos[0] - x0)
    return {
        "leg_kd": leg_kd_scale,
        "friction": friction,
        "solref": solref_timeconst,
        "steps": ran_steps,
        "fell": fell,
        "dx": dx,
        "avg_vx_world": dx / max(duration, 1e-9),
        "avg_vx_body": vx_sum / ran_steps,
        "max_roll": max_abs_roll,
        "max_pitch": max_abs_pitch,
        "max_ctrl": max_ctrl,
        "sat_frac": saturated_steps / ran_steps,
    }


def print_result(row):
    print(
        f"kd={row['leg_kd']:<4.2f} fr={row['friction']:<4.2f} sol={row['solref']:<5.3f} "
        f"fell={str(row['fell']):<5} dx={row['dx']:>7.3f} avg_vx={row['avg_vx_world']:>6.3f} "
        f"roll={row['max_roll']:>5.3f} pitch={row['max_pitch']:>5.3f} "
        f"max_ctrl={row['max_ctrl']:>5.1f} sat={row['sat_frac']:>5.2f}"
    )


def score(row):
    return (
        1000.0 if row["fell"] else 0.0
        + row["sat_frac"] * 10.0
        + row["max_roll"] * 2.0
        + row["max_pitch"] * 2.0
        - row["dx"]
    )


def main():
    parser = argparse.ArgumentParser(description="Sweep MuJoCo walking parameters.")
    parser.add_argument("--logdir", type=str, default=DEFAULT_LOGDIR)
    parser.add_argument("--ckptid", type=int, default=DEFAULT_CKPTID)
    parser.add_argument("--base-z", type=float, default=0.50)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--min-base-z", type=float, default=0.12)
    parser.add_argument("--x-vel", type=float, default=0.5)
    parser.add_argument("--y-vel", type=float, default=0.0)
    parser.add_argument("--yaw-vel", type=float, default=0.0)
    parser.add_argument("--ramp-time", type=float, default=0.0)
    parser.add_argument("--leg-kd-scales", type=str, default="1.0,1.2,1.5,1.8")
    parser.add_argument("--frictions", type=str, default="0.8,1.0,1.2")
    parser.add_argument("--solrefs", type=str, default="0.015,0.02,0.03")
    parser.add_argument("--arm-action-limit", type=float, default=1.0)
    parser.add_argument("--arm-kp-scale", type=float, default=1.0)
    parser.add_argument("--arm-damping-scale", type=float, default=0.5)
    parser.add_argument("--hold-arm-default", action="store_true")
    parser.add_argument("--disable-arm-policy", action="store_true")
    args = parser.parse_args()

    cfg = restore_cfg_from_run(args.logdir)
    ckpt_id = str(args.ckptid).zfill(6)
    dog_policy = load_dog_policy(str(ROOT / args.logdir), ckpt_id, cfg)
    arm_policy = load_arm_policy(str(ROOT / args.logdir), ckpt_id, cfg)

    rows = []
    for leg_kd, friction, solref in itertools.product(
        parse_floats(args.leg_kd_scales), parse_floats(args.frictions), parse_floats(args.solrefs)
    ):
        row = run_case(args, cfg, dog_policy, arm_policy, leg_kd, friction, solref)
        rows.append(row)
        print_result(row)

    print("\nTop candidates")
    for row in sorted(rows, key=score)[:5]:
        print_result(row)


if __name__ == "__main__":
    main()
