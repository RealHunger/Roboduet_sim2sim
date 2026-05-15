import argparse
import pickle as pkl
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ROOT = ensure_repo_root_on_path()

import isaacgym 
import torch

from go1_gym.envs.automatic.legged_robot_config import Cfg
from scripts.load_policy import load_arm_policy, load_dog_policy


DEFAULT_LOGDIR = "runs/overnight_go1_512/dummy-sf8qzpe4_seed6218"
DEFAULT_CKPTID = 30800


def restore_cfg_from_run(logdir):
    """Restore the training config saved with a RoboDuet run."""
    params_path = ROOT / logdir / "parameters.pkl"
    with params_path.open("rb") as file:
        saved = pkl.load(file)

    cfg = saved["Cfg"]
    for key, value in cfg.items():
        if hasattr(Cfg, key):
            target = getattr(Cfg, key)
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    if (
                        isinstance(value2, dict)
                        and hasattr(target, key2)
                        and hasattr(getattr(target, key2), "__dict__")
                    ):
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


def print_cfg_summary(cfg):
    print("Config summary")
    print("  dog obs dim:", cfg.dog.dog_num_observations)
    print("  dog obs history dim:", cfg.dog.dog_num_obs_history)
    print("  dog action dim:", cfg.dog.dog_actions)
    print("  arm obs dim:", cfg.arm.arm_num_observations)
    print("  arm obs history dim:", cfg.arm.arm_num_obs_history)
    print("  arm action dim:", cfg.arm.num_actions_arm_cd)
    print("  arm joint action dim:", cfg.arm.num_actions_arm)
    print("  use_rot6d:", getattr(cfg, "use_rot6d", None))


def describe_tensor(name, tensor):
    print(f"{name} shape: {tuple(tensor.shape)}")
    print(f"{name} finite: {torch.isfinite(tensor).all().item()}")
    print(f"{name} min/max: {tensor.min().item():.6f} / {tensor.max().item():.6f}")


def main():
    parser = argparse.ArgumentParser(description="Probe RoboDuet policy loading with dummy observations.")
    parser.add_argument("--logdir", type=str, default=DEFAULT_LOGDIR)
    parser.add_argument("--ckptid", type=int, default=DEFAULT_CKPTID)
    args = parser.parse_args()

    logdir = args.logdir
    ckpt_id = str(args.ckptid).zfill(6)

    cfg = restore_cfg_from_run(logdir)
    print_cfg_summary(cfg)

    dog_policy = load_dog_policy(str(ROOT / logdir), ckpt_id, cfg)
    arm_policy = load_arm_policy(str(ROOT / logdir), ckpt_id, cfg)

    dog_obs = {
        "obs": torch.zeros(1, cfg.dog.dog_num_observations),
        "privileged_obs": torch.zeros(1, cfg.dog.dog_num_privileged_obs),
        "obs_history": torch.zeros(1, cfg.dog.dog_num_obs_history),
    }
    arm_obs = {
        "obs": torch.zeros(1, cfg.arm.arm_num_observations),
        "privileged_obs": torch.zeros(1, cfg.arm.arm_num_privileged_obs),
        "obs_history": torch.zeros(1, cfg.arm.arm_num_obs_history),
    }

    print("Dummy observation shapes")
    describe_tensor("dog_obs['obs']", dog_obs["obs"])
    describe_tensor("dog_obs['obs_history']", dog_obs["obs_history"])
    describe_tensor("arm_obs['obs']", arm_obs["obs"])
    describe_tensor("arm_obs['obs_history']", arm_obs["obs_history"])

    with torch.no_grad():
        actions_arm = arm_policy(arm_obs)
        actions_dog = dog_policy(dog_obs)

    print("Policy output")
    describe_tensor("actions_arm", actions_arm)
    describe_tensor("actions_dog", actions_dog)

    assert actions_arm.shape == (1, cfg.arm.num_actions_arm_cd)
    assert actions_dog.shape == (1, cfg.dog.dog_actions)
    assert torch.isfinite(actions_arm).all()
    assert torch.isfinite(actions_dog).all()
    print("Policy probe passed.")


if __name__ == "__main__":
    main()
