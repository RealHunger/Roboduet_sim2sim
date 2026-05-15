from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import numpy as np

from sim2sim_mujoco.scripts.mujoco_policy_rollout import arm_command_obs_from_values


DEFAULT_ARM_COMMAND = np.array([0.5, 0.2, 0.0, 0.1, 0.5, 0.0], dtype=float)
SEQUENCE_DURATION = 32.0


def scripted_command(t, loop=False):
    if loop:
        t = t % SEQUENCE_DURATION

    dog = np.zeros(3, dtype=float)
    arm = DEFAULT_ARM_COMMAND.copy()

    if 2.0 <= t < 5.0:
        dog[0] = 0.3
    elif 5.0 <= t < 8.0:
        dog[0] = 0.5
    elif 8.0 <= t < 11.0:
        dog[0] = 0.3
        dog[2] = 0.8
    elif 11.0 <= t < 15.0:
        dog[2] = 1.0
    elif 15.0 <= t < 17.0:
        dog[0] = -0.3

    if 17.0 <= t < 20.0:
        arm[0] = 0.7
    elif 20.0 <= t < 23.0:
        arm[1] = 0.7
    elif 23.0 <= t < 26.0:
        arm[2] = -0.6
    elif 26.0 <= t < 29.0:
        arm[3] = 0.5
    elif 29.0 <= t < 32.0:
        dog[0] = 0.3
        arm[0] = 0.7
        arm[1] = 0.5

    return dog, arm


def arm_obs_from_command(arm_command):
    return arm_command_obs_from_values(*arm_command)


def metric_header():
    return [
        "t",
        "cmd_x",
        "cmd_y",
        "cmd_yaw",
        "arm_l",
        "arm_p",
        "arm_y",
        "base_x",
        "base_y",
        "base_z",
        "vx_body",
        "roll",
        "pitch",
        "arm_target_x",
        "arm_target_y",
        "arm_target_z",
        "arm_ee_x",
        "arm_ee_y",
        "arm_ee_z",
        "arm_dist",
        "max_ctrl",
    ]


def metric_row(t, dog_cmd, arm_cmd, base_pos, vx_body, roll, pitch, arm_target, arm_ee, max_ctrl):
    arm_dist = np.nan if arm_target is None or arm_ee is None else np.linalg.norm(arm_target - arm_ee)
    return [
        t,
        dog_cmd[0],
        dog_cmd[1],
        dog_cmd[2],
        arm_cmd[0],
        arm_cmd[1],
        arm_cmd[2],
        base_pos[0],
        base_pos[1],
        base_pos[2],
        vx_body,
        roll,
        pitch,
        *(arm_target if arm_target is not None else [np.nan, np.nan, np.nan]),
        *(arm_ee if arm_ee is not None else [np.nan, np.nan, np.nan]),
        arm_dist,
        max_ctrl,
    ]
