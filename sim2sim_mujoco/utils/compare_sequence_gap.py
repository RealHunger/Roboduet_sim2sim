import argparse
import csv

import numpy as np


def load_csv(path):
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    return rows


def col(rows, name):
    return np.array([float(row[name]) for row in rows], dtype=float)


def summarize(name, diff):
    print(f"{name:<14} max_abs={np.nanmax(np.abs(diff)): .5f} mean_abs={np.nanmean(np.abs(diff)): .5f} final={diff[-1]: .5f}")


def main():
    parser = argparse.ArgumentParser(description="Compare Isaac and MuJoCo scripted sequence CSVs.")
    parser.add_argument("--isaac", type=str, default="/tmp/isaac_sequence.csv")
    parser.add_argument("--mujoco", type=str, default="/tmp/mujoco_sequence.csv")
    args = parser.parse_args()

    isaac = load_csv(args.isaac)
    mujoco = load_csv(args.mujoco)
    n = min(len(isaac), len(mujoco))
    isaac = isaac[:n]
    mujoco = mujoco[:n]
    print("rows", n)

    for name in ["base_x", "base_y", "base_z", "vx_body", "roll", "pitch", "arm_dist", "max_ctrl"]:
        summarize(name, col(mujoco, name) - col(isaac, name))

    isaac_xyz = np.stack([col(isaac, "arm_ee_x"), col(isaac, "arm_ee_y"), col(isaac, "arm_ee_z")], axis=1)
    mujoco_xyz = np.stack([col(mujoco, "arm_ee_x"), col(mujoco, "arm_ee_y"), col(mujoco, "arm_ee_z")], axis=1)
    ee_dist = np.linalg.norm(mujoco_xyz - isaac_xyz, axis=1)
    print(f"{'arm_ee_xyz':<14} max={np.nanmax(ee_dist): .5f} mean={np.nanmean(ee_dist): .5f} final={ee_dist[-1]: .5f}")


if __name__ == "__main__":
    main()
