import argparse
import csv
import importlib.util
import math
import os
import pickle as pkl
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Run a minimal RoboDuet policy replay loop in Isaac Sim.")
    parser.add_argument("--interactive-baseline", action="store_true", help="Enable the tuned Isaac Sim GUI keyboard-control baseline.")
    parser.add_argument("--sequence-baseline", action="store_true", help="Enable the tuned Isaac Sim scripted-sequence baseline.")
    parser.add_argument("--logdir", type=str, default="runs/overnight_go1_512/dummy-sf8qzpe4_seed6218")
    parser.add_argument("--ckptid", type=int, default=30800)
    parser.add_argument("--usd", type=str, default="sim2sim_isaacsim/resources/robots/arx5p2Go1/usd/arx5p2Go1.usd")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--base-z", type=float, default=0.34)
    parser.add_argument("--sim-dt", type=float, default=0.005, help="Physics timestep. Training and MuJoCo baseline use 0.005.")
    parser.add_argument("--action-decimation", type=int, default=4)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--debug-joints", action="store_true", help="Print policy-order joint target/actual/error diagnostics.")
    parser.add_argument("--zero-policy-action", action="store_true", help="Run the loop but command zero policy actions.")
    parser.add_argument("--hold-arm-default", action="store_true", help="Ignore arm joint policy outputs and hold default arm joints.")
    parser.add_argument("--arm-action-limit", type=float, default=None, help="Optional clip for arm joint action before scaling.")
    parser.add_argument("--arm-target-alpha", type=float, default=0.25, help="Smooth arm target joints; 1.0 disables smoothing.")
    parser.add_argument("--physics-targets", action="store_true", help="Use actuator position targets instead of directly writing joint states.")
    parser.add_argument("--pd-efforts", action="store_true", help="Apply explicit PD efforts from target_q instead of direct joint-state writes.")
    parser.add_argument("--leg-kp-scale", type=float, default=1.0, help="Scale Isaac Sim leg actuator stiffness / explicit PD kp.")
    parser.add_argument("--leg-kd-scale", type=float, default=1.0, help="Scale Isaac Sim leg actuator damping / explicit PD kd.")
    parser.add_argument("--arm-kp-scale", type=float, default=1.0, help="Scale Isaac Sim arm actuator stiffness / explicit PD kp.")
    parser.add_argument("--arm-kd-scale", type=float, default=1.0, help="Scale Isaac Sim arm actuator damping / explicit PD kd.")
    parser.add_argument("--effort-limit", type=float, default=200.0, help="Actuator effort limit for Isaac Sim articulation config.")
    parser.add_argument("--velocity-limit", type=float, default=100.0, help="Actuator velocity limit for Isaac Sim articulation config.")
    parser.add_argument("--free-base", action="store_true", help="Let the floating base move freely. Default freezes it for scripted replay bring-up.")
    parser.add_argument("--use-gpu", action="store_true", help="Use CUDA PhysX. Default uses CPU PhysX for bring-up stability.")
    parser.add_argument("--min-base-z", type=float, default=0.24, help="Stop rollout if root height falls below this threshold.")
    parser.add_argument("--max-roll-pitch", type=float, default=0.8, help="Stop rollout if abs(roll) or abs(pitch) exceeds this threshold in radians.")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV path for rollout diagnostics.")
    parser.add_argument("--scripted-sequence", action="store_true", help="Replay the shared sim2sim scripted dog/arm command sequence.")
    parser.add_argument("--loop-sequence", action="store_true", help="Loop the scripted command sequence when --scripted-sequence is enabled.")
    parser.add_argument("--dog-command-scale", type=float, default=1.0, help="Scale scripted dog velocity commands for visual/debug checks.")
    parser.add_argument("--idle-command-threshold", type=float, default=0.05, help="Zero dog policy action below this command norm, matching the MuJoCo replay idle hold.")
    parser.add_argument("--no-hold-dog-when-idle", action="store_true", help="Disable dog action zeroing when dog command is near zero.")
    parser.add_argument("--keyboard", action="store_true", help="Enable GUI keyboard command control.")
    parser.add_argument("--keyboard-vx", type=float, default=0.35, help="Keyboard forward/back velocity command step.")
    parser.add_argument("--keyboard-vy", type=float, default=0.25, help="Keyboard lateral velocity command step.")
    parser.add_argument("--keyboard-yaw", type=float, default=0.6, help="Keyboard yaw velocity command step.")
    parser.add_argument("--keyboard-arm-step", type=float, default=0.05, help="Keyboard arm command increment per key press.")
    parser.add_argument("--hide-arm-markers", action="store_true", help="Disable red target / blue grasper visualization markers.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if args.interactive_baseline:
        args.physics_targets = True
        args.free_base = True
        args.keyboard = True
        args.scripted_sequence = False
        args.leg_kp_scale = 2.0
        args.leg_kd_scale = 1.5
        args.effort_limit = 300.0
        args.velocity_limit = 150.0
        args.print_every = max(args.print_every, 100)
        if args.num_steps == 20:
            args.num_steps = 100000

    if args.sequence_baseline:
        args.physics_targets = True
        args.free_base = True
        args.scripted_sequence = True
        args.leg_kp_scale = 2.0
        args.leg_kd_scale = 1.5
        args.effort_limit = 300.0
        args.velocity_limit = 150.0
        args.print_every = max(args.print_every, 100)
        if args.num_steps == 20:
            args.num_steps = 6400

    return args


args_cli = parse_args()
if args_cli.headless and args_cli.keyboard:
    raise ValueError("--keyboard requires GUI mode; remove --headless or omit --keyboard.")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

DEFAULT_ARM_COMMAND = torch.tensor([0.5, 0.2, 0.0, 0.1, 0.5, 0.0], dtype=torch.float32)
SEQUENCE_DURATION = 32.0

DEFAULT_JOINT_POS = {
    "FL_hip_joint": 0.1,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "FR_hip_joint": -0.1,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "RL_hip_joint": 0.1,
    "RL_thigh_joint": 1.0,
    "RL_calf_joint": -1.5,
    "RR_hip_joint": -0.1,
    "RR_thigh_joint": 1.0,
    "RR_calf_joint": -1.5,
    "zarx_j1": 0.0,
    "zarx_j2": 0.8,
    "zarx_j3": 0.8,
    "zarx_j4": 0.0,
    "zarx_j5": 0.0,
    "zarx_j6": 0.0,
    "zarx_j7": 0.0,
    "zarx_j8": 0.0,
}

POLICY_JOINT_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "zarx_j1", "zarx_j2", "zarx_j3", "zarx_j4", "zarx_j5", "zarx_j6",
]


def load_class(module_name, relative_path, class_name):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


Cfg = load_class("roboduet_legged_robot_config", "go1_gym/envs/automatic/legged_robot_config.py", "Cfg")
DogActorCritic = load_class("roboduet_dog_ac", "go1_gym_learn/ppo_cse_automatic/dog_ac.py", "DogActorCritic")
ArmActorCritic = load_class("roboduet_arm_ac", "go1_gym_learn/ppo_cse_automatic/arm_ac.py", "ArmActorCritic")


def resolve_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def make_robot_cfg(usd_path, cfg):
    if args_cli.physics_targets:
        actuators = {
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit_sim=args_cli.effort_limit,
                velocity_limit_sim=args_cli.velocity_limit,
                stiffness=cfg.dog.control.stiffness_leg["joint"] * args_cli.leg_kp_scale,
                damping=cfg.dog.control.damping_leg["joint"] * args_cli.leg_kd_scale,
            ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["zarx_j.*"],
                effort_limit_sim=args_cli.effort_limit,
                velocity_limit_sim=args_cli.velocity_limit,
                stiffness=25.0 * args_cli.arm_kp_scale,
                damping=2.0 * args_cli.arm_kd_scale,
            ),
        }
    else:
        actuators = {
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=args_cli.effort_limit,
                velocity_limit_sim=args_cli.velocity_limit,
                stiffness=0.0,
                damping=0.0,
            )
        }

    return ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, args_cli.base_z), joint_pos=DEFAULT_JOINT_POS),
        actuators=actuators,
    )


def quat_wxyz_to_matrix(quat):
    w, x, y, z = quat.unbind(-1)
    row0 = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), dim=-1)
    row1 = torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), dim=-1)
    row2 = torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def quat_apply_wxyz(quat, vec):
    rot = quat_wxyz_to_matrix(quat)
    return rot @ vec


def yaw_from_quat_wxyz(quat):
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_xyzw_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def euler_xyz_to_quat_xyzw(roll, pitch, yaw):
    q_yaw = [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]
    q_pitch = [0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2)]
    q_roll = [math.sin(roll / 2), 0.0, 0.0, math.cos(roll / 2)]
    return quat_xyzw_multiply(q_yaw, quat_xyzw_multiply(q_pitch, q_roll))


def quat_to_angle_xyzw(quat):
    x, y, z, w = quat
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-8:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(norm, w)
    return [x / norm * angle, y / norm * angle, z / norm * angle]


def arm_command_obs_from_values(l_cmd, p_cmd, y_cmd, roll_cmd, pitch_cmd, yaw_cmd):
    abg = quat_to_angle_xyzw(euler_xyz_to_quat_xyzw(roll_cmd, pitch_cmd, yaw_cmd))
    return torch.tensor([l_cmd, p_cmd, y_cmd, abg[0], abg[1], abg[2]], dtype=torch.float32)


def default_keyboard_command_state():
    return {
        "x_vel": 0.0,
        "y_vel": 0.0,
        "yaw_vel": 0.0,
        "l_cmd": 0.5,
        "p_cmd": 0.2,
        "y_cmd": 0.0,
        "roll_cmd": 0.1,
        "pitch_cmd": 0.5,
        "arm_yaw_cmd": 0.0,
    }


def reset_keyboard_arm_command(command_state):
    command_state.update({
        "l_cmd": 0.5,
        "p_cmd": 0.2,
        "y_cmd": 0.0,
        "roll_cmd": 0.1,
        "pitch_cmd": 0.5,
        "arm_yaw_cmd": 0.0,
    })


def keyboard_state_to_commands(command_state, dog_num_commands):
    dog = torch.zeros(dog_num_commands, dtype=torch.float32)
    if dog_num_commands > 0:
        dog[0] = command_state["x_vel"]
    if dog_num_commands > 1:
        dog[1] = command_state["y_vel"]
    if dog_num_commands > 2:
        dog[2] = command_state["yaw_vel"]
    arm = arm_command_obs_from_values(
        command_state["l_cmd"],
        command_state["p_cmd"],
        command_state["y_cmd"],
        command_state["roll_cmd"],
        command_state["pitch_cmd"],
        command_state["arm_yaw_cmd"],
    )
    return dog, arm


def install_keyboard_callback(command_state):
    import carb.input
    import omni.appwindow

    app_window = omni.appwindow.get_default_app_window()
    if app_window is None:
        raise RuntimeError("Failed to access Isaac Sim app window for keyboard control.")
    keyboard = app_window.get_keyboard()
    input_interface = carb.input.acquire_input_interface()

    def key_name(event):
        value = str(event.input)
        return value.rsplit(".", 1)[-1].lower()

    def on_keyboard_event(event):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True

        key = key_name(event)
        vx_step = args_cli.keyboard_vx
        vy_step = args_cli.keyboard_vy
        yaw_step = args_cli.keyboard_yaw
        arm_step = args_cli.keyboard_arm_step
        handled = True

        if key in ("numpad_8", "num_8"):
            command_state["x_vel"] += vx_step
        elif key in ("numpad_5", "num_5"):
            command_state["x_vel"] -= vx_step
        elif key in ("numpad_4", "num_4"):
            command_state["y_vel"] += vy_step
        elif key in ("numpad_6", "num_6"):
            command_state["y_vel"] -= vy_step
        elif key in ("numpad_7", "num_7"):
            command_state["yaw_vel"] += yaw_step
        elif key in ("numpad_9", "num_9"):
            command_state["yaw_vel"] -= yaw_step
        elif key in ("numpad_0", "num_0"):
            command_state["x_vel"] = 0.0
            command_state["y_vel"] = 0.0
            command_state["yaw_vel"] = 0.0
        elif key == "r":
            command_state["x_vel"] = 0.0
            command_state["y_vel"] = 0.0
            command_state["yaw_vel"] = 0.0
            reset_keyboard_arm_command(command_state)
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
        elif key == "w":
            command_state["pitch_cmd"] -= arm_step
        elif key == "s":
            command_state["pitch_cmd"] += arm_step
        elif key == "a":
            command_state["roll_cmd"] += arm_step
        elif key == "d":
            command_state["roll_cmd"] -= arm_step
        elif key == "q":
            command_state["arm_yaw_cmd"] += arm_step
        elif key == "e":
            command_state["arm_yaw_cmd"] -= arm_step
        else:
            handled = False

        if handled:
            print(
                "keyboard command",
                f"x={command_state['x_vel']:.2f}",
                f"y={command_state['y_vel']:.2f}",
                f"yaw={command_state['yaw_vel']:.2f}",
                f"arm_lpy=({command_state['l_cmd']:.2f},{command_state['p_cmd']:.2f},{command_state['y_cmd']:.2f})",
                f"arm_rpy=({command_state['roll_cmd']:.2f},{command_state['pitch_cmd']:.2f},{command_state['arm_yaw_cmd']:.2f})",
            )
        return True

    subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
    return input_interface, keyboard, subscription


def scripted_command(t, loop=False):
    if loop:
        t = t % SEQUENCE_DURATION

    dog = torch.zeros(3, dtype=torch.float32)
    arm = DEFAULT_ARM_COMMAND.clone()

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


def compute_clock_inputs(gait_index, dog_command):
    foot_indices = torch.tensor(
        [gait_index + 0.5, gait_index, gait_index, gait_index + 0.5],
        dtype=torch.float32,
    )
    if torch.linalg.norm(dog_command[:3]) < 0.1:
        foot_indices[:] = 0.25
    return torch.sin(2 * math.pi * foot_indices)


def roll_pitch_from_quat_wxyz(quat):
    w, x, y, z = quat.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def arm_target_world_pos(root_state, arm_values):
    l_cmd, p_cmd, y_cmd = arm_values[:3]
    x_local = l_cmd * torch.cos(p_cmd) * torch.cos(y_cmd)
    y_local = l_cmd * torch.cos(p_cmd) * torch.sin(y_cmd)
    z_world = l_cmd * torch.sin(p_cmd) + 0.38
    yaw = yaw_from_quat_wxyz(root_state[3:7])
    x_world = x_local * torch.cos(yaw) - y_local * torch.sin(yaw) + root_state[0]
    y_world = x_local * torch.sin(yaw) + y_local * torch.cos(yaw) + root_state[1]
    return torch.stack((x_world, y_world, z_world))


def arm_grasper_world_pos(robot):
    body_index = {name: idx for idx, name in enumerate(robot.data.body_names)}
    if "zarx_body6" not in body_index:
        return None
    ee_state = robot.data.body_link_state_w[0, body_index["zarx_body6"]]
    return ee_state[:3] + quat_apply_wxyz(ee_state[3:7], torch.tensor([0.1, 0.0, 0.0], dtype=ee_state.dtype, device=ee_state.device))


def make_marker(stage, path, color, radius=0.04):
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xformable = UsdGeom.Xformable(sphere.GetPrim())
    translate_op = xformable.AddTranslateOp()
    return translate_op


def set_marker_pos(translate_op, pos):
    values = [float(x.detach().cpu()) for x in pos]
    translate_op.Set(Gf.Vec3d(values[0], values[1], values[2]))


def update_arm_markers(marker_ops, robot, arm_values):
    if marker_ops is None:
        return
    root = robot.data.root_link_state_w[0]
    target = arm_target_world_pos(root, arm_values.to(root.device))
    set_marker_pos(marker_ops["target"], target)
    grasper = arm_grasper_world_pos(robot)
    if grasper is not None:
        set_marker_pos(marker_ops["grasper"], grasper)


def sequence_metric_header():
    return [
        "t", "cmd_x", "cmd_y", "cmd_yaw", "arm_l", "arm_p", "arm_y", "base_x", "base_y", "base_z",
        "vx_body", "roll", "pitch", "arm_target_x", "arm_target_y", "arm_target_z", "arm_ee_x", "arm_ee_y",
        "arm_ee_z", "arm_dist", "max_ctrl",
    ]


def sequence_metric_row(t, dog_cmd, arm_values, robot, state, q_err):
    root = robot.data.root_link_state_w[0]
    target = arm_target_world_pos(root, arm_values.to(root.device))
    ee = arm_grasper_world_pos(robot)
    arm_dist = torch.nan if ee is None else torch.linalg.norm(target - ee)
    ee_values = [math.nan, math.nan, math.nan] if ee is None else [float(x.detach().cpu()) for x in ee]
    return [
        t,
        float(dog_cmd[0]),
        float(dog_cmd[1]),
        float(dog_cmd[2]),
        float(arm_values[0]),
        float(arm_values[1]),
        float(arm_values[2]),
        float(root[0].detach().cpu()),
        float(root[1].detach().cpu()),
        float(root[2].detach().cpu()),
        float(state["base_lin_vel_body"][0].detach().cpu()),
        float(state["roll_pitch"][0].detach().cpu()),
        float(state["roll_pitch"][1].detach().cpu()),
        float(target[0].detach().cpu()),
        float(target[1].detach().cpu()),
        float(target[2].detach().cpu()),
        *ee_values,
        float(arm_dist.detach().cpu()) if ee is not None else math.nan,
        q_err,
    ]


def extract_state(robot):
    root = robot.data.root_link_state_w[0]
    base_quat_wxyz = root[3:7]
    rot = quat_wxyz_to_matrix(base_quat_wxyz)
    base_lin_vel_world = root[7:10]
    base_ang_vel_world = root[10:13]
    roll, pitch = roll_pitch_from_quat_wxyz(base_quat_wxyz)
    joint_index = {name: idx for idx, name in enumerate(robot.data.joint_names)}
    policy_indices = torch.tensor([joint_index[name] for name in POLICY_JOINT_ORDER], device=robot.data.joint_pos.device)
    return {
        "base_lin_vel_body": rot.transpose(-1, -2) @ base_lin_vel_world,
        "base_ang_vel_body": rot.transpose(-1, -2) @ base_ang_vel_world,
        "projected_gravity": rot.transpose(-1, -2) @ torch.tensor([0.0, 0.0, -1.0], dtype=root.dtype, device=root.device),
        "roll_pitch": torch.stack((roll, pitch)),
        "q_policy": robot.data.joint_pos[0, policy_indices],
        "qd_policy": robot.data.joint_vel[0, policy_indices],
    }


def restore_cfg_from_run(logdir):
    with (REPO_ROOT / logdir / "parameters.pkl").open("rb") as file:
        saved = pkl.load(file)
    cfg = saved["Cfg"]
    for key, value in cfg.items():
        if hasattr(Cfg, key):
            target = getattr(Cfg, key)
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    if isinstance(value2, dict) and hasattr(target, key2) and hasattr(getattr(target, key2), "__dict__"):
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


def load_dog_policy(logdir, ckpt_id, cfg):
    actor_critic = DogActorCritic(cfg.dog.dog_num_observations, cfg.dog.dog_num_privileged_obs, cfg.dog.dog_num_obs_history, cfg.dog.dog_actions).to("cpu")
    actor_critic.load_state_dict(torch.load(Path(logdir) / "checkpoints_dog" / f"ac_weights_{ckpt_id}.pt", map_location="cpu"))
    actor_critic.eval()

    def policy(obs):
        with torch.no_grad():
            history = obs["obs_history"].to("cpu")
            latent = actor_critic.adaptation_module(history)
            return actor_critic.actor_body(torch.cat((history, latent), dim=-1))

    return policy


def load_arm_policy(logdir, ckpt_id, cfg):
    actor_critic = ArmActorCritic(cfg.arm.arm_num_observations, cfg.arm.arm_num_privileged_obs, cfg.arm.arm_num_obs_history, cfg.arm.num_actions_arm_cd).to("cpu")
    actor_critic.load_state_dict(torch.load(Path(logdir) / "checkpoints_arm" / f"ac_weights_{ckpt_id}.pt", map_location="cpu"))
    actor_critic.eval()

    def policy(obs):
        with torch.no_grad():
            history = obs["obs_history"].to("cpu")
            hist = actor_critic.actor_history_encoder(history[..., :-cfg.arm.arm_num_observations])
            latent = actor_critic.adaptation_module(history)
            return actor_critic.actor_body(torch.cat((obs["obs"].to("cpu"), latent, hist), dim=-1))

    return policy


def build_arm_obs(cfg, state, prev_action_18, arm_command):
    default_q = torch.tensor([DEFAULT_JOINT_POS[name] for name in POLICY_JOINT_ORDER], dtype=torch.float32)
    q = state["q_policy"].detach().cpu().to(torch.float32)
    return torch.cat(((q[12:18] - default_q[12:18]) * cfg.obs_scales.dof_pos, prev_action_18[12:18], arm_command, state["roll_pitch"].detach().cpu().to(torch.float32))).unsqueeze(0)


def build_dog_obs(cfg, state, prev_action_18, dog_command, arm_command, clock_inputs=None):
    default_q = torch.tensor([DEFAULT_JOINT_POS[name] for name in POLICY_JOINT_ORDER], dtype=torch.float32)
    q = state["q_policy"].detach().cpu().to(torch.float32)
    qd = state["qd_policy"].detach().cpu().to(torch.float32)
    commands_scale_dog = torch.tensor([cfg.obs_scales.lin_vel, cfg.obs_scales.lin_vel, cfg.obs_scales.ang_vel, cfg.obs_scales.body_pitch_cmd, cfg.obs_scales.body_roll_cmd], dtype=torch.float32)[: cfg.dog.dog_num_commands]
    pieces = []
    if cfg.env.observe_vel:
        pieces.extend((state["base_lin_vel_body"].detach().cpu().to(torch.float32) * cfg.obs_scales.lin_vel, state["base_ang_vel_body"].detach().cpu().to(torch.float32) * cfg.obs_scales.ang_vel))
    pieces.extend((state["projected_gravity"].detach().cpu().to(torch.float32), (q[:12] - default_q[:12]) * cfg.obs_scales.dof_pos, qd[:12] * cfg.obs_scales.dof_vel, prev_action_18[:12], dog_command[: cfg.dog.dog_num_commands] * commands_scale_dog, arm_command, state["roll_pitch"].detach().cpu().to(torch.float32)))
    if cfg.env.observe_clock_inputs:
        pieces.append(torch.zeros(4, dtype=torch.float32) if clock_inputs is None else clock_inputs.to(torch.float32))
    return torch.cat(pieces).unsqueeze(0)


def make_obs_dict(obs, history, priv_dim):
    return {"obs": obs, "privileged_obs": torch.zeros(1, priv_dim), "obs_history": history}


def update_history(history, obs):
    obs_dim = obs.shape[1]
    return torch.cat((history[:, obs_dim:], obs), dim=-1)


def initialize_robot(robot, sim, base_z):
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] = torch.tensor([0.0, 0.0, base_z], dtype=root_state.dtype, device=root_state.device)
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=root_state.dtype, device=root_state.device)
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    robot.write_data_to_sim()
    sim.step(render=not args_cli.headless)
    robot.update(sim.get_physics_dt())


def actions_to_target_q(cfg, action_18, device):
    default_q = torch.tensor([DEFAULT_JOINT_POS[name] for name in POLICY_JOINT_ORDER], dtype=torch.float32, device=device)
    clipped = torch.clamp(action_18, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
    scaled = clipped * cfg.control.action_scale
    scaled[[0, 3, 6, 9]] *= cfg.control.hip_scale_reduction
    return default_q + scaled


def smooth_arm_target_q(target_q, prev_arm_target_q):
    if args_cli.arm_target_alpha >= 1.0:
        return target_q, target_q[12:18].clone()
    target_q = target_q.clone()
    target_q[12:18] = prev_arm_target_q + args_cli.arm_target_alpha * (target_q[12:18] - prev_arm_target_q)
    return target_q, target_q[12:18].clone()


def write_policy_target(robot, target_q_policy, base_z, cfg):
    name_to_index = {name: idx for idx, name in enumerate(robot.data.joint_names)}
    joint_ids = torch.tensor([name_to_index[name] for name in POLICY_JOINT_ORDER], dtype=torch.int64, device=robot.data.joint_pos.device)
    full_target = robot.data.joint_pos.clone()
    full_target[:, joint_ids] = target_q_policy.unsqueeze(0).to(full_target.device)
    if args_cli.pd_efforts:
        current_q = robot.data.joint_pos[:, joint_ids]
        current_qd = robot.data.joint_vel[:, joint_ids]
        # Arm gains mirror the MuJoCo rollout's hand-tuned PD gains approximately.
        kp = torch.tensor(
            [cfg.dog.control.stiffness_leg["joint"] * args_cli.leg_kp_scale] * 12
            + [40.0 * args_cli.arm_kp_scale, 70.0 * args_cli.arm_kp_scale, 70.0 * args_cli.arm_kp_scale, 25.0 * args_cli.arm_kp_scale, 25.0 * args_cli.arm_kp_scale, 25.0 * args_cli.arm_kp_scale],
            dtype=current_q.dtype,
            device=current_q.device,
        )
        kd = torch.tensor(
            [cfg.dog.control.damping_leg["joint"] * args_cli.leg_kd_scale] * 12
            + [3.0 * args_cli.arm_kd_scale, 15.0 * args_cli.arm_kd_scale, 15.0 * args_cli.arm_kd_scale, 2.0 * args_cli.arm_kd_scale, 2.0 * args_cli.arm_kd_scale, 2.0 * args_cli.arm_kd_scale],
            dtype=current_q.dtype,
            device=current_q.device,
        )
        effort_policy = kp * (target_q_policy.to(current_q.device).unsqueeze(0) - current_q) - kd * current_qd
        robot.set_joint_effort_target(effort_policy, joint_ids=joint_ids)
    elif args_cli.physics_targets:
        robot.set_joint_position_target(full_target)
    else:
        robot.write_joint_state_to_sim(full_target, torch.zeros_like(full_target))
    if not args_cli.free_base:
        root_state = robot.data.root_link_state_w.clone()
        root_state[:, :3] = torch.tensor([0.0, 0.0, base_z], dtype=root_state.dtype, device=root_state.device)
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=root_state.dtype, device=root_state.device)
        root_state[:, 7:] = 0.0
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_data_to_sim()


def print_joint_debug(robot, target_q):
    name_to_index = {name: idx for idx, name in enumerate(robot.data.joint_names)}
    joint_ids = torch.tensor([name_to_index[name] for name in POLICY_JOINT_ORDER], dtype=torch.int64, device=robot.data.joint_pos.device)
    q = robot.data.joint_pos[0, joint_ids].detach().cpu()
    qd = robot.data.joint_vel[0, joint_ids].detach().cpu()
    target = target_q.detach().cpu()
    err = q - target
    print("[JOINT DEBUG] name                 target    actual       err        qd")
    for name, target_i, q_i, err_i, qd_i in zip(POLICY_JOINT_ORDER, target, q, err, qd):
        print(f"[JOINT DEBUG] {name:<20} {float(target_i): .4f} {float(q_i): .4f} {float(err_i): .4f} {float(qd_i): .4f}")


def main():
    cfg = restore_cfg_from_run(args_cli.logdir)
    ckpt_id = str(args_cli.ckptid).zfill(6)
    dog_policy = load_dog_policy(REPO_ROOT / args_cli.logdir, ckpt_id, cfg)
    arm_policy = load_arm_policy(REPO_ROOT / args_cli.logdir, ckpt_id, cfg)

    usd_path = resolve_path(args_cli.usd)
    sim_device = args_cli.device if args_cli.use_gpu else "cpu"
    sim = SimulationContext(SimulationCfg(device=sim_device, dt=args_cli.sim_dt, use_fabric=False))
    sim_utils.GroundPlaneCfg(size=(6.0, 6.0)).func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg(size=(6.0, 6.0)))
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)).func("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)))
    robot = Articulation(make_robot_cfg(usd_path, cfg))
    sim.reset()
    robot.reset()
    initialize_robot(robot, sim, args_cli.base_z)

    marker_ops = None
    if not args_cli.hide_arm_markers:
        stage = sim.stage
        marker_ops = {
            "target": make_marker(stage, "/World/arm_target_marker", (1.0, 0.0, 0.0)),
            "grasper": make_marker(stage, "/World/arm_grasper_marker", (0.0, 0.3, 1.0)),
        }

    prev_action_18 = torch.zeros(18, dtype=torch.float32)
    dog_command = torch.zeros(cfg.dog.dog_num_commands, dtype=torch.float32)
    arm_command = DEFAULT_ARM_COMMAND.clone()
    arm_history = torch.zeros(1, cfg.arm.arm_num_obs_history)
    dog_history = torch.zeros(1, cfg.dog.dog_num_obs_history)
    target_q = torch.tensor([DEFAULT_JOINT_POS[name] for name in POLICY_JOINT_ORDER], dtype=torch.float32)
    prev_arm_target_q = target_q[12:18].clone()
    sim_dt = sim.get_physics_dt()
    gait_index = 0.0
    clock_inputs = torch.zeros(4, dtype=torch.float32)
    keyboard_command_state = default_keyboard_command_state()
    keyboard_subscription = None
    if args_cli.keyboard:
        _, _, keyboard_subscription = install_keyboard_callback(keyboard_command_state)
        print("[INFO] Keyboard control enabled. Numpad 8/5/4/6/7/9/0 control dog, I/K/U/O/J/L/W/S/A/D/Q/E control arm, R resets commands.")

    csv_file = None
    csv_writer = None
    if args_cli.csv is not None:
        csv_path = Path(args_cli.csv).expanduser()
        if not csv_path.is_absolute():
            csv_path = REPO_ROOT / csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="")
        if args_cli.scripted_sequence:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(sequence_metric_header())
        else:
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=["step", "root_z", "roll", "pitch", "max_q_err", "action_min", "action_max", "target_min", "target_max"],
            )
            csv_writer.writeheader()

    print(f"[INFO] Replay start: steps={args_cli.num_steps} sim_device={sim_device} use_fabric=False")
    print(
        f"[INFO] action_decimation={args_cli.action_decimation} zero_policy_action={args_cli.zero_policy_action} "
        f"physics_targets={args_cli.physics_targets} pd_efforts={args_cli.pd_efforts} free_base={args_cli.free_base} "
        f"scripted_sequence={args_cli.scripted_sequence}"
    )

    stop_reason = None
    for step in range(args_cli.num_steps):
        if step % args_cli.action_decimation == 0:
            t_policy = (step // args_cli.action_decimation) * args_cli.action_decimation * sim_dt
            if args_cli.scripted_sequence:
                dog_values, arm_values = scripted_command(t_policy, loop=args_cli.loop_sequence)
                dog_values = dog_values * args_cli.dog_command_scale
                dog_command[: min(3, cfg.dog.dog_num_commands)] = dog_values[: min(3, cfg.dog.dog_num_commands)]
                arm_command = arm_command_obs_from_values(*[float(x) for x in arm_values])
            elif args_cli.keyboard:
                dog_command, arm_command = keyboard_state_to_commands(keyboard_command_state, cfg.dog.dog_num_commands)
            gait_index = (gait_index + args_cli.action_decimation * sim_dt * 3.0) % 1.0
            clock_inputs = compute_clock_inputs(gait_index, dog_command)
            state = extract_state(robot)
            arm_obs = build_arm_obs(cfg, state, prev_action_18, arm_command)
            arm_history = update_history(arm_history, arm_obs)
            arm_obs_dict = make_obs_dict(arm_obs, arm_history, cfg.arm.arm_num_privileged_obs)
            dog_obs = build_dog_obs(cfg, state, prev_action_18, dog_command, arm_command, clock_inputs)
            dog_history = update_history(dog_history, dog_obs)
            dog_obs_dict = make_obs_dict(dog_obs, dog_history, cfg.dog.dog_num_privileged_obs)

            if args_cli.zero_policy_action:
                dog_action = torch.zeros(12)
                arm_action = torch.zeros(cfg.arm.num_actions_arm_cd)
            else:
                dog_action = dog_policy(dog_obs_dict).reshape(-1).detach().cpu()
                arm_action = arm_policy(arm_obs_dict).reshape(-1).detach().cpu()
            if not args_cli.no_hold_dog_when_idle and torch.linalg.norm(dog_command[:3]) < args_cli.idle_command_threshold:
                dog_action = torch.zeros(12)

            arm_joint_action = torch.zeros(cfg.arm.num_actions_arm) if args_cli.hold_arm_default else arm_action[: cfg.arm.num_actions_arm]
            if args_cli.arm_action_limit is not None:
                arm_joint_action = torch.clamp(arm_joint_action, -args_cli.arm_action_limit, args_cli.arm_action_limit)
            action_18 = torch.cat((dog_action, arm_joint_action)).to(torch.float32)
            target_q = actions_to_target_q(cfg, action_18, device=torch.device("cpu"))
            target_q, prev_arm_target_q = smooth_arm_target_q(target_q, prev_arm_target_q)
            prev_action_18 = torch.clamp(action_18, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

        write_policy_target(robot, target_q, args_cli.base_z, cfg)
        if args_cli.scripted_sequence or args_cli.keyboard:
            update_arm_markers(marker_ops, robot, arm_command)
        sim.step(render=not args_cli.headless)
        robot.update(sim_dt)

        state = extract_state(robot)
        root_z = float(robot.data.root_link_state_w[0, 2].detach().cpu())
        roll = float(state["roll_pitch"][0].detach().cpu())
        pitch = float(state["roll_pitch"][1].detach().cpu())
        q_err = float(torch.max(torch.abs(state["q_policy"].detach().cpu() - target_q)))
        if csv_writer is not None:
            if args_cli.scripted_sequence and step % args_cli.action_decimation == 0:
                t = step * sim_dt
                dog_values, arm_values = scripted_command(t, loop=args_cli.loop_sequence)
                dog_values = dog_values * args_cli.dog_command_scale
                csv_writer.writerow(sequence_metric_row(t, dog_values, arm_values, robot, state, q_err))
            elif not args_cli.scripted_sequence:
                csv_writer.writerow(
                    {
                        "step": step,
                        "root_z": root_z,
                        "roll": roll,
                        "pitch": pitch,
                        "max_q_err": q_err,
                        "action_min": float(prev_action_18.min()),
                        "action_max": float(prev_action_18.max()),
                        "target_min": float(target_q.min()),
                        "target_max": float(target_q.max()),
                    }
                )
        if root_z < args_cli.min_base_z:
            stop_reason = f"base height below threshold: {root_z:.4f} < {args_cli.min_base_z:.4f}"
        elif max(abs(roll), abs(pitch)) > args_cli.max_roll_pitch:
            stop_reason = f"roll/pitch above threshold: roll={roll:.4f}, pitch={pitch:.4f}"

        if step % args_cli.print_every == 0 or step == args_cli.num_steps - 1 or stop_reason is not None:
            cmd_text = ""
            if args_cli.scripted_sequence:
                dog_values, arm_values = scripted_command(step * sim_dt, loop=args_cli.loop_sequence)
                dog_values = dog_values * args_cli.dog_command_scale
                root = robot.data.root_link_state_w[0]
                vx = float(state["base_lin_vel_body"][0].detach().cpu())
                dog_action_abs = float(prev_action_18[:12].abs().max())
                cmd_text = (
                    f" cmd=({float(dog_values[0]):.2f},{float(dog_values[2]):.2f})"
                    f" x={float(root[0].detach().cpu()):.3f} vx={vx:.3f} dog_act={dog_action_abs:.3f}"
                    f" arm=({float(arm_values[0]):.2f},{float(arm_values[1]):.2f},{float(arm_values[2]):.2f})"
                )
            print(
                f"[STEP {step:04d}] z={root_z:.4f} max_q_err={q_err:.4f} "
                f"roll={roll:.3f} pitch={pitch:.3f} "
                f"action=[{float(prev_action_18.min()):.3f},{float(prev_action_18.max()):.3f}] "
                f"target=[{float(target_q.min()):.3f},{float(target_q.max()):.3f}]"
                f"{cmd_text}"
            )
            if args_cli.debug_joints:
                print_joint_debug(robot, target_q)
        if stop_reason is not None:
            print(f"[WARN] Early stop: {stop_reason}")
            break

    if csv_file is not None:
        csv_file.close()
    keyboard_subscription = None
    print("[INFO] Replay complete." if stop_reason is None else "[INFO] Replay stopped early.")


if __name__ == "__main__":
    success = False
    try:
        main()
        success = True
    finally:
        if args_cli.headless and success:
            os._exit(0)
        simulation_app.close()
