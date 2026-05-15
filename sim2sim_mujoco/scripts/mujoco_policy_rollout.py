import argparse
import time

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import isaacgym  # noqa: F401  # Isaac Gym must be imported before torch.
import mujoco
import numpy as np
import torch

try:
    import mujoco.viewer
except ImportError:  # pragma: no cover
    mujoco.viewer = None

from sim2sim_mujoco.utils.mujoco_obs_probe import (
    DEFAULT_DOF_POS_20,
    DEFAULT_LOGDIR,
    DEFAULT_CKPTID,
    KD_18,
    KP_18,
    MODEL_PATH,
    POLICY_QPOS_ADDR_20,
    POLICY_QVEL_ADDR_20,
    ROOT,
    build_arm_obs,
    build_dog_obs,
    extract_mujoco_state,
    initialize_standing_pose,
    restore_cfg_from_run,
)
from scripts.load_policy import load_arm_policy, load_dog_policy


ACTION_JOINT_NAMES_18 = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "zarx_j1", "zarx_j2", "zarx_j3", "zarx_j4", "zarx_j5", "zarx_j6",
]

GRIPPER_QPOS_ADDR = np.array([25, 26])
GRIPPER_QVEL_ADDR = np.array([24, 25])
GRIPPER_DEFAULT_POS = DEFAULT_DOF_POS_20[18:20]


def update_history(history, obs, obs_dim):
    return torch.cat([history[:, obs_dim:], obs], dim=-1)


def repeat_history(obs, history_dim, obs_dim):
    repeats = history_dim // obs_dim
    return obs.repeat(1, repeats)


def quat_xyzw_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def quat_xyzw_apply(quat, vec):
    q_vec = quat[:3]
    q_w = quat[3]
    uv = np.cross(q_vec, vec)
    uuv = np.cross(q_vec, uv)
    return vec + 2 * (q_w * uv + uuv)


def quat_to_angle_xyzw(quat):
    roll_vec = quat_xyzw_apply(quat, np.array([0.0, 1.0, 0.0]))
    alpha = np.arctan2(roll_vec[2], roll_vec[1])
    pitch_vec = quat_xyzw_apply(quat, np.array([0.0, 0.0, 1.0]))
    beta = np.arctan2(pitch_vec[0], pitch_vec[2])
    yaw_vec = quat_xyzw_apply(quat, np.array([1.0, 0.0, 0.0]))
    gamma = np.arctan2(yaw_vec[1], yaw_vec[0])
    return np.array([alpha, beta, gamma])


def euler_xyz_to_quat_xyzw(roll, pitch, yaw):
    q_yaw = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
    q_pitch = np.array([0.0, np.sin(pitch / 2), 0.0, np.cos(pitch / 2)])
    q_roll = np.array([np.sin(roll / 2), 0.0, 0.0, np.cos(roll / 2)])
    return quat_xyzw_multiply(q_yaw, quat_xyzw_multiply(q_pitch, q_roll))


def default_arm_command_obs():
    # Matches scripts/play_by_key.py defaults: l, p, y, roll, pitch, yaw.
    l_cmd, p_cmd, y_cmd = 0.5, 0.2, 0.0
    roll_cmd, pitch_cmd, yaw_cmd = 0.1, 0.5, 0.0
    return arm_command_obs_from_values(l_cmd, p_cmd, y_cmd, roll_cmd, pitch_cmd, yaw_cmd)


def arm_command_obs_from_values(l_cmd, p_cmd, y_cmd, roll_cmd, pitch_cmd, yaw_cmd):
    quat = euler_xyz_to_quat_xyzw(roll_cmd, pitch_cmd, yaw_cmd)
    abg = quat_to_angle_xyzw(quat)
    return np.array([l_cmd, p_cmd, y_cmd, abg[0], abg[1], abg[2]])


def default_arm_command_state():
    return {
        "l_cmd": 0.5,
        "p_cmd": 0.2,
        "y_cmd": 0.0,
        "roll_cmd": 0.1,
        "pitch_cmd": 0.5,
        "arm_yaw_cmd": 0.0,
    }


def reset_arm_command_state(command_state):
    command_state.update(default_arm_command_state())


def arm_command_obs_from_state(command_state):
    return arm_command_obs_from_values(
        command_state["l_cmd"],
        command_state["p_cmd"],
        command_state["y_cmd"],
        command_state["roll_cmd"],
        command_state["pitch_cmd"],
        command_state["arm_yaw_cmd"],
    )


def arm_target_world_pos(data, command_state):
    l_cmd = command_state["l_cmd"]
    p_cmd = command_state["p_cmd"]
    y_cmd = command_state["y_cmd"]
    x_local = l_cmd * np.cos(p_cmd) * np.cos(y_cmd)
    y_local = l_cmd * np.cos(p_cmd) * np.sin(y_cmd)
    z_world = l_cmd * np.sin(p_cmd) + 0.38
    yaw = quat_wxyz_to_yaw(data.qpos[3:7])
    x_world = x_local * np.cos(yaw) - y_local * np.sin(yaw) + data.qpos[0]
    y_world = x_local * np.sin(yaw) + y_local * np.cos(yaw) + data.qpos[1]
    return np.array([x_world, y_world, z_world])


def update_arm_target_marker(model, data, command_state):
    update_mocap_body_pos(model, data, "arm_target_marker", arm_target_world_pos(data, command_state))


def body_world_pos(model, data, name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        return None
    return data.xpos[body_id].copy()


def arm_grasper_world_pos(model, data):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "zarx_body6")
    if body_id < 0:
        return None
    x_axis = data.xmat[body_id].reshape(3, 3)[:, 0]
    # Matches Isaac: grasper_in_world = end_effector_state + quat_rotate(ee_quat, [0.1, 0, 0]).
    return data.xpos[body_id].copy() + 0.1 * x_axis


def update_mocap_body_pos(model, data, body_name, pos):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return
    mocap_id = model.body_mocapid[body_id]
    if mocap_id < 0:
        return
    data.mocap_pos[mocap_id] = pos


def update_arm_ee_marker(model, data):
    ee_pos = arm_grasper_world_pos(model, data)
    if ee_pos is not None:
        update_mocap_body_pos(model, data, "arm_ee_marker", ee_pos)


def arm_ee_jacobian(model, data):
    jac = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "zarx_body6")
    if body_id < 0:
        return None
    grasper_pos = arm_grasper_world_pos(model, data)
    if grasper_pos is None:
        return None
    mujoco.mj_jac(model, data, jac, jac_rot, grasper_pos, body_id)
    return jac


def arm_ik_target_q(model, data, command_state, current_target_q, step_size, damping, max_dq, iterations):
    arm_qvel_addr = POLICY_QVEL_ADDR_20[12:18]
    arm_qpos_addr = POLICY_QPOS_ADDR_20[12:18]
    target_q = current_target_q.copy()
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    ik_q = data.qpos[arm_qpos_addr].copy()
    target_pos = arm_target_world_pos(data, command_state)

    for _ in range(iterations):
        data.qpos[arm_qpos_addr] = ik_q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        ee_pos = arm_grasper_world_pos(model, data)
        jac = arm_ee_jacobian(model, data)
        if ee_pos is None or jac is None:
            break

        err = target_pos - ee_pos
        if np.linalg.norm(err) < 0.01:
            break
        if np.linalg.norm(err) > step_size:
            err = err / np.linalg.norm(err) * step_size

        j_arm = jac[:, arm_qvel_addr]
        dq = j_arm.T @ np.linalg.solve(j_arm @ j_arm.T + (damping ** 2) * np.eye(3), err)
        dq = np.clip(dq, -max_dq, max_dq)
        ik_q = np.clip(ik_q + dq, model.actuator_ctrlrange[12:18, 0], model.actuator_ctrlrange[12:18, 1])

    data.qpos[:] = saved_qpos
    data.qvel[:] = saved_qvel
    mujoco.mj_forward(model, data)
    target_q[12:18] = ik_q
    return target_q


def compute_clock_inputs(gait_index, commands_dog):
    phases, offsets, bounds = 0.5, 0.0, 0.0
    foot_indices = np.array([
        gait_index + phases + offsets + bounds,
        gait_index + offsets,
        gait_index + bounds,
        gait_index + phases,
    ])
    if np.linalg.norm(commands_dog[:3]) < 0.1:
        foot_indices[:] = 0.25
    return np.sin(2 * np.pi * foot_indices)


def quat_wxyz_to_yaw(quat):
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def make_keyboard_callback(command_state):
    keypad_keys = {
        320: "0",
        321: "1",
        322: "2",
        323: "3",
        324: "4",
        325: "5",
        326: "6",
        327: "7",
        328: "8",
        329: "9",
    }

    def on_key(keycode):
        if keycode in keypad_keys:
            key = keypad_keys[keycode]
        else:
            try:
                key = chr(keycode).lower()
            except ValueError:
                return

        layout = command_state["keyboard_layout"]
        step = command_state["step"]
        arm_step = command_state["arm_step"]
        handled = True

        if key == "8":
            command_state["x_vel"] += step
        elif key == "5":
            command_state["x_vel"] -= step
        elif key == "4":
            command_state["y_vel"] += step
        elif key == "6":
            command_state["y_vel"] -= step
        elif key == "7":
            command_state["yaw_vel"] += step
        elif key == "9":
            command_state["yaw_vel"] -= step
        elif key == "0":
            command_state["x_vel"] = 0.0
            command_state["y_vel"] = 0.0
            command_state["yaw_vel"] = 0.0
        elif layout == "isaac" and key == "r":
            command_state["x_vel"] = 0.0
            command_state["y_vel"] = 0.0
            command_state["yaw_vel"] = 0.0
            reset_arm_command_state(command_state)
        elif key == "i":
            command_state["l_cmd"] = min(0.8, command_state["l_cmd"] + 0.5 * arm_step)
        elif key == "k":
            command_state["l_cmd"] = max(0.2, command_state["l_cmd"] - 0.5 * arm_step)
        elif key == "u":
            command_state["p_cmd"] += arm_step
        elif key == "o":
            command_state["p_cmd"] -= arm_step
        elif key == "j":
            command_state["y_cmd"] += arm_step
        elif key == "l":
            command_state["y_cmd"] -= arm_step
        elif layout == "isaac" and key == "w":
            command_state["pitch_cmd"] -= arm_step
        elif layout == "isaac" and key == "s":
            command_state["pitch_cmd"] += arm_step
        elif layout == "isaac" and key == "a":
            command_state["roll_cmd"] += arm_step
        elif layout == "isaac" and key == "d":
            command_state["roll_cmd"] -= arm_step
        elif layout == "isaac" and key == "q":
            command_state["arm_yaw_cmd"] += arm_step
        elif layout == "isaac" and key == "e":
            command_state["arm_yaw_cmd"] -= arm_step
        elif layout == "mujoco-safe" and key == "z":
            command_state["roll_cmd"] += arm_step
        elif layout == "mujoco-safe" and key == "x":
            command_state["roll_cmd"] -= arm_step
        elif layout == "mujoco-safe" and key == "c":
            command_state["pitch_cmd"] += arm_step
        elif layout == "mujoco-safe" and key == "v":
            command_state["pitch_cmd"] -= arm_step
        elif layout == "mujoco-safe" and key == "b":
            command_state["arm_yaw_cmd"] += arm_step
        elif layout == "mujoco-safe" and key == "n":
            command_state["arm_yaw_cmd"] -= arm_step
        elif layout == "mujoco-safe" and key == "m":
            command_state["x_vel"] = 0.0
            command_state["y_vel"] = 0.0
            command_state["yaw_vel"] = 0.0
            reset_arm_command_state(command_state)
        elif key == "]":
            command_state["step"] = min(0.5, command_state["step"] + 0.05)
        elif key == "[":
            command_state["step"] = max(0.05, command_state["step"] - 0.05)
        else:
            handled = False

        if not handled:
            return

        print(
            "keyboard command",
            f"x={command_state['x_vel']:.2f}",
            f"y={command_state['y_vel']:.2f}",
            f"yaw={command_state['yaw_vel']:.2f}",
            f"arm_lpy=({command_state['l_cmd']:.2f},{command_state['p_cmd']:.2f},{command_state['y_cmd']:.2f})",
            f"arm_rpy=({command_state['roll_cmd']:.2f},{command_state['pitch_cmd']:.2f},{command_state['arm_yaw_cmd']:.2f})",
            f"layout={layout}",
            f"step={command_state['step']:.2f}",
            f"arm_step={command_state['arm_step']:.2f}",
        )

    return on_key


def plan_from_arm_action(cfg, commands_dog, arm_action_8):
    num_plan_actions = cfg.arm.num_actions_arm_cd - cfg.arm.num_actions_arm
    if num_plan_actions <= 0:
        return

    plan_action = arm_action_8[-num_plan_actions:] * 0.4
    commands_dog[3] = np.clip(
        plan_action[0],
        cfg.commands.limit_body_pitch[0],
        cfg.commands.limit_body_pitch[1] / 4 * 3.0,
    )
    commands_dog[4] = np.clip(
        plan_action[1],
        cfg.commands.limit_body_roll[0],
        cfg.commands.limit_body_roll[1],
    )


def actions_to_target_q(cfg, action_18):
    action = np.clip(action_18.copy(), -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
    action_scaled = action * cfg.control.action_scale
    action_scaled[[0, 3, 6, 9]] *= cfg.control.hip_scale_reduction
    return DEFAULT_DOF_POS_20[:18] + action_scaled


def clip_policy_action(cfg, action_18):
    return np.clip(action_18, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)


def apply_pd_control(model, data, target_q, leg_kd_scale=1.0):
    q = data.qpos[POLICY_QPOS_ADDR_20[:18]]
    qd = data.qvel[POLICY_QVEL_ADDR_20[:18]]
    torque = KP_18[:12] * (target_q[:12] - q[:12]) - (KD_18[:12] * leg_kd_scale) * qd[:12]
    ctrl_min = model.actuator_ctrlrange[:, 0]
    ctrl_max = model.actuator_ctrlrange[:, 1]
    data.ctrl[:12] = np.clip(torque, ctrl_min[:12], ctrl_max[:12])
    data.ctrl[12:18] = np.clip(target_q[12:18], ctrl_min[12:18], ctrl_max[12:18])
    data.qpos[GRIPPER_QPOS_ADDR] = GRIPPER_DEFAULT_POS
    data.qvel[GRIPPER_QVEL_ADDR] = 0.0


def tune_arm_servo(model, kp_scale, damping_scale):
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if name and name.startswith("zarx_j"):
            model.actuator_gainprm[actuator_id, 0] *= kp_scale
            model.actuator_biasprm[actuator_id, 1] *= kp_scale

    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name in {"zarx_j1", "zarx_j2", "zarx_j3", "zarx_j4", "zarx_j5", "zarx_j6"}:
            dof_id = model.jnt_dofadr[joint_id]
            model.dof_damping[dof_id] *= damping_scale


def print_segment(name, values):
    values = np.asarray(values)
    formatted = np.array2string(values, precision=4, suppress_small=True)
    print(
        f"  {name:<24} shape={values.shape} "
        f"min={values.min(): .4f} max={values.max(): .4f} values={formatted}"
    )


def print_dog_obs_debug(cfg, state, prev_action_18, commands_dog, commands_arm_obs, clock_inputs=None):
    commands_scale_dog = np.array([
        cfg.obs_scales.lin_vel,
        cfg.obs_scales.lin_vel,
        cfg.obs_scales.ang_vel,
        cfg.obs_scales.body_pitch_cmd,
        cfg.obs_scales.body_roll_cmd,
    ])[:cfg.dog.dog_num_commands]

    segments = [
        ("projected_gravity", state["projected_gravity"]),
        ("leg_q_error", (state["q_policy"][:12] - DEFAULT_DOF_POS_20[:12]) * cfg.obs_scales.dof_pos),
        ("leg_qd", state["qd_policy"][:12] * cfg.obs_scales.dof_vel),
        ("prev_dog_action", prev_action_18[:12]),
        ("dog_command", commands_dog[:5] * commands_scale_dog[:5]),
        ("arm_command_obs", commands_arm_obs[:6]),
        ("roll_pitch", np.array([state["roll"], state["pitch"]])),
    ]
    if cfg.env.observe_vel:
        segments = [
            ("base_lin_vel_body", state["base_lin_vel_body"] * cfg.obs_scales.lin_vel),
            ("base_ang_vel_body", state["base_ang_vel_body"] * cfg.obs_scales.ang_vel),
        ] + segments
    if cfg.env.observe_clock_inputs:
        segments.append(("clock_inputs", np.zeros(4) if clock_inputs is None else clock_inputs))

    print("Dog obs debug")
    offset = 0
    for name, values in segments:
        print(f"  idx {offset:02d}:{offset + len(values):02d}")
        print_segment(name, values)
        offset += len(values)
    print("  total dim", offset)


def print_arm_obs_debug(state, prev_action_18, commands_arm_obs):
    arm_q_error = state["q_policy"][12:18] - DEFAULT_DOF_POS_20[12:18]
    segments = [
        ("arm_q_error", arm_q_error),
        ("prev_arm_action", prev_action_18[12:18]),
        ("arm_command_obs", commands_arm_obs[:6]),
        ("roll_pitch", np.array([state["roll"], state["pitch"]])),
    ]
    print("Arm obs debug")
    offset = 0
    for name, values in segments:
        print(f"  idx {offset:02d}:{offset + len(values):02d}")
        print_segment(name, values)
        offset += len(values)
    print("  total dim", offset)


def print_action_debug(data, target_q, action_18):
    q = data.qpos[POLICY_QPOS_ADDR_20[:18]]
    qd = data.qvel[POLICY_QVEL_ADDR_20[:18]]
    print("Action / target / state debug")
    print("  joint             action   target_q        q       err        qd      ctrl")
    for i, name in enumerate(ACTION_JOINT_NAMES_18):
        print(
            f"  {name:<12} "
            f"{action_18[i]: 8.4f} "
            f"{target_q[i]: 9.4f} "
            f"{q[i]: 8.4f} "
            f"{target_q[i] - q[i]: 8.4f} "
            f"{qd[i]: 8.4f} "
            f"{data.ctrl[i]: 8.4f}"
        )


def print_arm_action_debug(data, target_q, action_18, arm_action_8):
    q = data.qpos[POLICY_QPOS_ADDR_20[:18]]
    qd = data.qvel[POLICY_QVEL_ADDR_20[:18]]
    print("Arm action debug")
    print("  arm_action_8", np.array2string(arm_action_8, precision=4, suppress_small=True))
    print("  joint             action   target_q        q       err        qd      ctrl")
    for i in range(12, 18):
        name = ACTION_JOINT_NAMES_18[i]
        print(
            f"  {name:<12} "
            f"{action_18[i]: 8.4f} "
            f"{target_q[i]: 9.4f} "
            f"{q[i]: 8.4f} "
            f"{target_q[i] - q[i]: 8.4f} "
            f"{qd[i]: 8.4f} "
            f"{data.ctrl[i]: 8.4f}"
        )


def print_joint_map_debug(model):
    print("Joint / actuator map debug")
    print("  idx  action_joint   qpos  qvel  actuator            actuator_joint")
    for i, joint_name in enumerate(ACTION_JOINT_NAMES_18):
        qpos = POLICY_QPOS_ADDR_20[i]
        qvel = POLICY_QVEL_ADDR_20[i]
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        trnid = model.actuator_trnid[i]
        actuator_joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, trnid[0])
        print(f"  {i:02d}   {joint_name:<12} {qpos:4d} {qvel:4d}  {str(actuator_name):<18} {str(actuator_joint)}")


def rollout_step(cfg, model, data, dog_policy, arm_policy, buffers, disable_arm_policy, zero_policy_action, debug_dog_obs):
    state = extract_mujoco_state(data)
    arm_obs_current = build_arm_obs(cfg, state, buffers["prev_action_18"], buffers["commands_arm_obs"])
    buffers["arm_obs_history"] = update_history(
        buffers["arm_obs_history"], arm_obs_current, cfg.arm.arm_num_observations
    )
    arm_obs = {
        "obs": arm_obs_current,
        "privileged_obs": torch.zeros(1, cfg.arm.arm_num_privileged_obs),
        "obs_history": buffers["arm_obs_history"],
    }

    if disable_arm_policy:
        arm_action = np.zeros(cfg.arm.num_actions_arm_cd)
    else:
        with torch.no_grad():
            arm_action = arm_policy(arm_obs).cpu().numpy()[0]
        if not zero_policy_action:
            plan_from_arm_action(cfg, buffers["commands_dog"], arm_action)

    state = extract_mujoco_state(data)
    dog_obs_current = build_dog_obs(
        cfg,
        state,
        buffers["prev_action_18"],
        buffers["commands_dog"],
        buffers["commands_arm_obs"],
        buffers["clock_inputs"],
    )
    if debug_dog_obs:
        print_dog_obs_debug(
            cfg,
            state,
            buffers["prev_action_18"],
            buffers["commands_dog"],
            buffers["commands_arm_obs"],
            buffers["clock_inputs"],
        )
    if debug_dog_obs:
        print_arm_obs_debug(state, buffers["prev_action_18"], buffers["commands_arm_obs"])
    buffers["dog_obs_history"] = update_history(
        buffers["dog_obs_history"], dog_obs_current, cfg.dog.dog_num_observations
    )
    dog_obs = {
        "obs": dog_obs_current,
        "privileged_obs": torch.zeros(1, cfg.dog.dog_num_privileged_obs),
        "obs_history": buffers["dog_obs_history"],
    }

    if zero_policy_action:
        dog_action = np.zeros(cfg.dog.dog_actions)
    else:
        with torch.no_grad():
            dog_action = dog_policy(dog_obs).cpu().numpy()[0]
    if buffers["hold_dog_when_idle"] and np.linalg.norm(buffers["commands_dog"][:3]) < buffers["idle_command_threshold"]:
        dog_action = np.zeros(cfg.dog.dog_actions)

    arm_joint_action = np.zeros(cfg.arm.num_actions_arm) if buffers["hold_arm_default"] else arm_action[:cfg.arm.num_actions_arm]
    if buffers["arm_action_limit"] is not None:
        arm_joint_action = np.clip(arm_joint_action, -buffers["arm_action_limit"], buffers["arm_action_limit"])
    action_18 = clip_policy_action(cfg, np.concatenate([dog_action, arm_joint_action]))
    target_q = actions_to_target_q(cfg, action_18)
    if buffers["arm_target_alpha"] < 1.0:
        target_q[12:18] = (
            buffers["prev_arm_target_q"]
            + buffers["arm_target_alpha"] * (target_q[12:18] - buffers["prev_arm_target_q"])
        )
        buffers["prev_arm_target_q"] = target_q[12:18].copy()
    if buffers["arm_ik_control"]:
        target_q = arm_ik_target_q(
            model,
            data,
            buffers["command_state"],
            target_q,
            buffers["arm_ik_step_size"],
            buffers["arm_ik_damping"],
            buffers["arm_ik_max_dq"],
            buffers["arm_ik_iterations"],
        )
        action_18[12:18] = (target_q[12:18] - DEFAULT_DOF_POS_20[12:18]) / cfg.control.action_scale
    buffers["prev_action_18"] = action_18
    return state, dog_action, arm_action, action_18, target_q


def run_rollout(args):
    cfg = restore_cfg_from_run(args.logdir)
    ckpt_id = str(args.ckptid).zfill(6)
    dog_policy = load_dog_policy(str(ROOT / args.logdir), ckpt_id, cfg)
    arm_policy = load_arm_policy(str(ROOT / args.logdir), ckpt_id, cfg)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    tune_arm_servo(model, args.arm_kp_scale, args.arm_damping_scale)
    if args.joint_map_debug:
        print_joint_map_debug(model)
    data = mujoco.MjData(model)
    initialize_standing_pose(model, data, args.base_z)

    buffers = {
        "prev_action_18": np.zeros(18),
        "commands_dog": np.zeros(cfg.dog.dog_num_commands),
        "commands_arm_obs": default_arm_command_obs() if args.default_arm_command else np.zeros(cfg.arm.arm_num_commands),
        "arm_obs_history": torch.zeros(1, cfg.arm.arm_num_obs_history),
        "dog_obs_history": torch.zeros(1, cfg.dog.dog_num_obs_history),
        "hold_arm_default": args.hold_arm_default,
        "arm_action_limit": args.arm_action_limit,
        "arm_target_alpha": args.arm_target_alpha,
        "prev_arm_target_q": DEFAULT_DOF_POS_20[12:18].copy(),
        "arm_ik_control": args.arm_ik_control,
        "arm_ik_step_size": args.arm_ik_step_size,
        "arm_ik_damping": args.arm_ik_damping,
        "arm_ik_max_dq": args.arm_ik_max_dq,
        "arm_ik_iterations": args.arm_ik_iterations,
        "command_state": None,
        "hold_dog_when_idle": args.hold_dog_when_idle,
        "idle_command_threshold": args.idle_command_threshold,
        "gait_index": 0.0,
        "clock_inputs": np.zeros(4),
    }
    command_state = {
        "x_vel": args.x_vel,
        "y_vel": args.y_vel,
        "yaw_vel": args.yaw_vel,
        "step": args.keyboard_step,
        "arm_step": args.arm_command_step,
        "keyboard_layout": args.keyboard_layout,
        "target_heading": quat_wxyz_to_yaw(data.qpos[3:7]),
    }
    command_state.update(default_arm_command_state())
    buffers["command_state"] = command_state

    for _ in range(args.warmup_steps):
        apply_pd_control(model, data, DEFAULT_DOF_POS_20[:18], args.leg_kd_scale)
        mujoco.mj_step(model, data)

    if args.prefill_history:
        state = extract_mujoco_state(data)
        arm_obs_current = build_arm_obs(cfg, state, buffers["prev_action_18"], buffers["commands_arm_obs"])
        dog_obs_current = build_dog_obs(
            cfg,
            state,
            buffers["prev_action_18"],
            buffers["commands_dog"],
            buffers["commands_arm_obs"],
        )
        buffers["arm_obs_history"] = repeat_history(
            arm_obs_current, cfg.arm.arm_num_obs_history, cfg.arm.arm_num_observations
        )
        buffers["dog_obs_history"] = repeat_history(
            dog_obs_current, cfg.dog.dog_num_obs_history, cfg.dog.dog_num_observations
        )

    viewer_context = None
    if args.viewer:
        if mujoco.viewer is None:
            raise RuntimeError("mujoco.viewer is not available in this environment")
        if args.keyboard_control:
            viewer_context = mujoco.viewer.launch_passive(
                model, data, key_callback=make_keyboard_callback(command_state)
            )
        else:
            viewer_context = mujoco.viewer.launch_passive(model, data)

    try:
        viewer = viewer_context.__enter__() if viewer_context is not None else None
        for step in range(args.steps):
            ramp = 1.0 if args.ramp_time <= 0 else min(1.0, step * model.opt.timestep * cfg.control.decimation / args.ramp_time)
            buffers["commands_dog"][0] = command_state["x_vel"] * ramp
            buffers["commands_dog"][1] = command_state["y_vel"] * ramp
            yaw_command = command_state["yaw_vel"] + args.yaw_trim
            if args.heading_hold:
                current_heading = quat_wxyz_to_yaw(data.qpos[3:7])
                if abs(yaw_command) < 1e-6:
                    yaw_error = wrap_to_pi(current_heading - command_state["target_heading"])
                    yaw_command += -args.heading_kp * yaw_error
                else:
                    command_state["target_heading"] = current_heading
            buffers["commands_dog"][2] = yaw_command * ramp
            buffers["commands_arm_obs"] = arm_command_obs_from_state(command_state)
            update_arm_target_marker(model, data, command_state)
            update_arm_ee_marker(model, data)
            buffers["gait_index"] = (buffers["gait_index"] + model.opt.timestep * cfg.control.decimation * 3.0) % 1.0
            buffers["clock_inputs"] = compute_clock_inputs(buffers["gait_index"], buffers["commands_dog"])
            state, dog_action, arm_action, action_18, target_q = rollout_step(
                cfg,
                model,
                data,
                dog_policy,
                arm_policy,
                buffers,
                args.disable_arm_policy,
                args.zero_policy_action,
                args.print_dog_obs and step % args.print_every == 0,
            )

            for _ in range(cfg.control.decimation):
                apply_pd_control(model, data, target_q, args.leg_kd_scale)
                mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()
                if args.realtime:
                    time.sleep(model.opt.timestep * cfg.control.decimation)

            if step % args.print_every == 0:
                q = data.qpos[POLICY_QPOS_ADDR_20[:18]]
                max_err = np.max(np.abs(target_q - q))
                arm_target = arm_target_world_pos(data, command_state)
                arm_ee = arm_grasper_world_pos(model, data)
                arm_target_ee_dist = np.nan if arm_ee is None else np.linalg.norm(arm_target - arm_ee)
                print(
                    step,
                    "base_x", f"{data.qpos[0]:.4f}",
                    "base_z", f"{data.qpos[2]:.4f}",
                    "vx", f"{state['base_lin_vel_body'][0]:.4f}",
                    "cmd_x", f"{buffers['commands_dog'][0]:.4f}",
                    "cmd_yaw", f"{buffers['commands_dog'][2]:.4f}",
                    "arm_target", np.array2string(arm_target, precision=2, suppress_small=True),
                    "arm_ee", np.array2string(arm_ee, precision=2, suppress_small=True) if arm_ee is not None else "None",
                    "arm_dist", f"{arm_target_ee_dist:.3f}",
                    "roll", f"{state['roll']:.4f}",
                    "pitch", f"{state['pitch']:.4f}",
                    "max_ctrl", f"{np.max(np.abs(data.ctrl)):.4f}",
                    "max_err", f"{max_err:.4f}",
                    "dog_action", f"[{dog_action.min():.3f}, {dog_action.max():.3f}]",
                    "arm_action", f"[{arm_action.min():.3f}, {arm_action.max():.3f}]",
                )
                if args.print_action_debug:
                    print_action_debug(data, target_q, action_18)
                    print_arm_action_debug(data, target_q, action_18, arm_action)

            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                print("Simulation exploded: non-finite qpos/qvel")
                break
            if data.qpos[2] < args.min_base_z:
                print("Stopping: base too low, likely fallen")
                break

    finally:
        if viewer_context is not None:
            viewer_context.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser(description="Run zero-command RoboDuet policy in MuJoCo.")
    parser.add_argument("--logdir", type=str, default=DEFAULT_LOGDIR)
    parser.add_argument("--ckptid", type=int, default=DEFAULT_CKPTID)
    parser.add_argument("--base-z", type=float, default=0.50)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--min-base-z", type=float, default=0.12)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--disable-arm-policy", action="store_true")
    parser.add_argument("--zero-policy-action", action="store_true")
    parser.add_argument("--print-dog-obs", action="store_true")
    parser.add_argument("--print-action-debug", action="store_true")
    parser.add_argument("--prefill-history", action="store_true")
    parser.add_argument("--default-arm-command", action="store_true", default=True)
    parser.add_argument("--hold-arm-default", action="store_true")
    parser.add_argument("--arm-action-limit", type=float, default=None)
    parser.add_argument("--arm-command-step", type=float, default=0.15)
    parser.add_argument("--arm-target-alpha", type=float, default=1.0)
    parser.add_argument("--arm-ik-control", action="store_true")
    parser.add_argument("--arm-ik-step-size", type=float, default=0.04)
    parser.add_argument("--arm-ik-damping", type=float, default=0.08)
    parser.add_argument("--arm-ik-max-dq", type=float, default=0.08)
    parser.add_argument("--arm-ik-iterations", type=int, default=8)
    parser.add_argument("--arm-kp-scale", type=float, default=1.0)
    parser.add_argument("--arm-damping-scale", type=float, default=1.0)
    parser.add_argument("--leg-kd-scale", type=float, default=1.2)
    parser.add_argument("--x-vel", type=float, default=0.0)
    parser.add_argument("--y-vel", type=float, default=0.0)
    parser.add_argument("--yaw-vel", type=float, default=0.0)
    parser.add_argument("--yaw-trim", type=float, default=0.0)
    parser.add_argument("--heading-hold", action="store_true")
    parser.add_argument("--heading-kp", type=float, default=0.8)
    parser.add_argument("--ramp-time", type=float, default=3.0)
    parser.add_argument("--keyboard-control", action="store_true")
    parser.add_argument("--keyboard-layout", choices=["isaac", "mujoco-safe"], default="isaac")
    parser.add_argument("--keyboard-step", type=float, default=0.1)
    parser.add_argument("--hold-dog-when-idle", action="store_true")
    parser.add_argument("--idle-command-threshold", type=float, default=0.05)
    parser.add_argument("--joint-map-debug", action="store_true")
    parser.add_argument("--interactive-baseline", action="store_true")
    args = parser.parse_args()
    if args.interactive_baseline:
        args.viewer = True
        args.realtime = True
        args.keyboard_control = True
        args.heading_hold = True
        args.heading_kp = 2.0
        args.prefill_history = True
        args.hold_dog_when_idle = True
        args.arm_target_alpha = min(args.arm_target_alpha, 0.25)
        args.steps = max(args.steps, 100000)
    run_rollout(args)


if __name__ == "__main__":
    main()
