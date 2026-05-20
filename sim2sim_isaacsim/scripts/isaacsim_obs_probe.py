import argparse
import importlib.util
import os
import pickle as pkl
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Build RoboDuet observations/actions from Isaac Sim state.")
    parser.add_argument("--logdir", type=str, default="runs/overnight_go1_512/dummy-sf8qzpe4_seed6218")
    parser.add_argument("--ckptid", type=int, default=30800)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--base-z", type=float, default=0.50)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
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


DEFAULT_ARM_COMMAND = torch.tensor([0.5, 0.2, 0.0, 0.1, 0.5, 0.0], dtype=torch.float32)

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


def load_cfg_class():
    cfg_path = REPO_ROOT / "go1_gym/envs/automatic/legged_robot_config.py"
    spec = importlib.util.spec_from_file_location("roboduet_legged_robot_config", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Cfg


def load_policy_class(module_name, relative_path, class_name):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


Cfg = load_cfg_class()
DogActorCritic = load_policy_class("roboduet_dog_ac", "go1_gym_learn/ppo_cse_automatic/dog_ac.py", "DogActorCritic")
ArmActorCritic = load_policy_class("roboduet_arm_ac", "go1_gym_learn/ppo_cse_automatic/arm_ac.py", "ArmActorCritic")


def resolve_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def make_robot_cfg(usd_path):
    return ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, args_cli.base_z), joint_pos=DEFAULT_JOINT_POS),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=200.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=0.0,
            )
        },
    )


def quat_wxyz_to_matrix(quat):
    w, x, y, z = quat.unbind(-1)
    row0 = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), dim=-1)
    row1 = torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), dim=-1)
    row2 = torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def roll_pitch_from_quat_wxyz(quat):
    w, x, y, z = quat.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def initialize_robot(robot, sim):
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] = torch.tensor([0.0, 0.0, args_cli.base_z], dtype=root_state.dtype, device=root_state.device)
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=root_state.dtype, device=root_state.device)
    root_state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    robot.write_data_to_sim()
    sim_dt = sim.get_physics_dt()
    for _ in range(args_cli.steps):
        sim.step(render=not args_cli.headless)
        robot.update(sim_dt)


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
    params_path = REPO_ROOT / logdir / "parameters.pkl"
    with params_path.open("rb") as file:
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
    actor_critic = DogActorCritic(
        cfg.dog.dog_num_observations,
        cfg.dog.dog_num_privileged_obs,
        cfg.dog.dog_num_obs_history,
        cfg.dog.dog_actions,
    ).to("cpu")
    ckpt = torch.load(Path(logdir) / "checkpoints_dog" / f"ac_weights_{ckpt_id}.pt", map_location="cpu")
    actor_critic.load_state_dict(ckpt)
    actor_critic.eval()

    def policy(obs):
        with torch.no_grad():
            latent = actor_critic.adaptation_module(obs["obs_history"].to("cpu"))
            return actor_critic.actor_body(torch.cat((obs["obs_history"].to("cpu"), latent), dim=-1))

    return policy


def load_arm_policy(logdir, ckpt_id, cfg):
    actor_critic = ArmActorCritic(
        cfg.arm.arm_num_observations,
        cfg.arm.arm_num_privileged_obs,
        cfg.arm.arm_num_obs_history,
        cfg.arm.num_actions_arm_cd,
        device="cpu",
    ).to("cpu")
    ckpt = torch.load(Path(logdir) / "checkpoints_arm" / f"ac_weights_{ckpt_id}.pt", map_location="cpu")
    actor_critic.load_state_dict(ckpt)
    actor_critic.eval()

    def policy(obs):
        with torch.no_grad():
            obs_history = obs["obs_history"].to("cpu")
            hist = actor_critic.actor_history_encoder(obs_history[..., :-cfg.arm.arm_num_observations])
            latent = actor_critic.adaptation_module(obs_history)
            return actor_critic.actor_body(torch.cat((obs["obs"].to("cpu"), latent, hist), dim=-1))

    return policy


def build_arm_obs(cfg, state, prev_action_18, arm_command):
    default_q = torch.tensor([DEFAULT_JOINT_POS[name] for name in POLICY_JOINT_ORDER], dtype=torch.float32)
    q = state["q_policy"].detach().cpu().to(torch.float32)
    roll_pitch = state["roll_pitch"].detach().cpu().to(torch.float32)
    obs = torch.cat(
        (
            (q[12:18] - default_q[12:18]) * cfg.obs_scales.dof_pos,
            prev_action_18[12:18],
            arm_command,
            roll_pitch,
        )
    ).unsqueeze(0)
    assert obs.shape[1] == cfg.arm.arm_num_observations, (obs.shape, cfg.arm.arm_num_observations)
    return obs


def build_dog_obs(cfg, state, prev_action_18, dog_command, arm_command):
    default_q = torch.tensor([DEFAULT_JOINT_POS[name] for name in POLICY_JOINT_ORDER], dtype=torch.float32)
    q = state["q_policy"].detach().cpu().to(torch.float32)
    qd = state["qd_policy"].detach().cpu().to(torch.float32)
    commands_scale_dog = torch.tensor(
        [cfg.obs_scales.lin_vel, cfg.obs_scales.lin_vel, cfg.obs_scales.ang_vel, cfg.obs_scales.body_pitch_cmd, cfg.obs_scales.body_roll_cmd],
        dtype=torch.float32,
    )[: cfg.dog.dog_num_commands]
    pieces = []
    if cfg.env.observe_vel:
        pieces.extend((state["base_lin_vel_body"].detach().cpu().to(torch.float32) * cfg.obs_scales.lin_vel,
                       state["base_ang_vel_body"].detach().cpu().to(torch.float32) * cfg.obs_scales.ang_vel))
    pieces.extend(
        (
            state["projected_gravity"].detach().cpu().to(torch.float32),
            (q[:12] - default_q[:12]) * cfg.obs_scales.dof_pos,
            qd[:12] * cfg.obs_scales.dof_vel,
            prev_action_18[:12],
            dog_command[: cfg.dog.dog_num_commands] * commands_scale_dog,
            arm_command,
            state["roll_pitch"].detach().cpu().to(torch.float32),
        )
    )
    if cfg.env.observe_clock_inputs:
        pieces.append(torch.zeros(4, dtype=torch.float32))
    obs = torch.cat(pieces).unsqueeze(0)
    assert obs.shape[1] == cfg.dog.dog_num_observations, (obs.shape, cfg.dog.dog_num_observations)
    return obs


def make_obs_dict(obs, history_dim, priv_dim):
    history = obs.repeat(1, history_dim // obs.shape[1])
    return {"obs": obs, "privileged_obs": torch.zeros(1, priv_dim), "obs_history": history}


def print_tensor(name, tensor):
    flat = tensor.detach().cpu().reshape(-1)
    values = " ".join(f"{float(value): .5f}" for value in flat[:24])
    suffix = " ..." if flat.numel() > 24 else ""
    print(f"{name} shape={tuple(tensor.shape)} min={float(flat.min()):.5f} max={float(flat.max()):.5f} values=[{values}{suffix}]")


def main():
    cfg = restore_cfg_from_run(args_cli.logdir)
    usd_path = resolve_path("sim2sim_isaacsim/resources/robots/arx5p2Go1/usd/arx5p2Go1.usd")
    sim = SimulationContext(SimulationCfg(device="cpu", use_fabric=False))
    sim_utils.GroundPlaneCfg(size=(6.0, 6.0)).func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg(size=(6.0, 6.0)))
    robot = Articulation(make_robot_cfg(usd_path))
    sim.reset()
    robot.reset()
    initialize_robot(robot, sim)
    state = extract_state(robot)

    prev_action_18 = torch.zeros(18, dtype=torch.float32)
    dog_command = torch.zeros(cfg.dog.dog_num_commands, dtype=torch.float32)
    arm_command = DEFAULT_ARM_COMMAND.clone()
    arm_obs = build_arm_obs(cfg, state, prev_action_18, arm_command)
    dog_obs = build_dog_obs(cfg, state, prev_action_18, dog_command, arm_command)
    arm_obs_dict = make_obs_dict(arm_obs, cfg.arm.arm_num_obs_history, cfg.arm.arm_num_privileged_obs)
    dog_obs_dict = make_obs_dict(dog_obs, cfg.dog.dog_num_obs_history, cfg.dog.dog_num_privileged_obs)

    ckpt_id = str(args_cli.ckptid).zfill(6)
    dog_policy = load_dog_policy(REPO_ROOT / args_cli.logdir, ckpt_id, cfg)
    arm_policy = load_arm_policy(REPO_ROOT / args_cli.logdir, ckpt_id, cfg)
    dog_action = dog_policy(dog_obs_dict)
    arm_action = arm_policy(arm_obs_dict)

    print(f"[INFO] cfg dog obs/action: {cfg.dog.dog_num_observations}/{cfg.dog.dog_actions}")
    print(f"[INFO] cfg arm obs/action: {cfg.arm.arm_num_observations}/{cfg.arm.num_actions_arm_cd}")
    print_tensor("dog_obs", dog_obs)
    print_tensor("arm_obs", arm_obs)
    print_tensor("dog_action", dog_action)
    print_tensor("arm_action", arm_action)
    print("[INFO] Observation/action probe complete.")


if __name__ == "__main__":
    success = False
    try:
        main()
        success = True
    finally:
        if args_cli.headless and success:
            os._exit(0)
        simulation_app.close()
