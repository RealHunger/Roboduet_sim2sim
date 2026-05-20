import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Load and inspect the RoboDuet Go1+ARX USD in Isaac Sim.")
    parser.add_argument(
        "--usd",
        type=str,
        default="sim2sim_isaacsim/resources/robots/arx5p2Go1/usd/arx5p2Go1.usd",
        help="Path to the converted robot USD, relative to repo root or absolute.",
    )
    parser.add_argument(
        "--prim-path",
        type=str,
        default="/World/Robot",
        help="Prim path where the robot USD is referenced.",
    )
    parser.add_argument("--max-prims", type=int, default=120, help="Maximum number of prim paths to print.")
    parser.add_argument(
        "--step-physics",
        action="store_true",
        help="Step PhysX while the GUI is open. By default this script only updates the UI for stable visualization.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
import omni.kit.app  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from pxr import UsdGeom, UsdLux, UsdPhysics  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def print_stage_summary(stage, max_prims):
    print("\n[INFO] Stage prims")
    prim_paths = [str(prim.GetPath()) for prim in stage.Traverse()]
    for prim_path in prim_paths[:max_prims]:
        print(f"  {prim_path}")
    if len(prim_paths) > max_prims:
        print(f"  ... ({len(prim_paths) - max_prims} more prims)")
    print(f"[INFO] Total prims: {len(prim_paths)}")


def has_api(prim, api_schema):
    return prim.HasAPI(api_schema)


def print_physics_summary(stage):
    articulation_roots = []
    joints = []
    rigid_bodies = []

    for prim in stage.Traverse():
        if has_api(prim, UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(str(prim.GetPath()))
        if prim.IsA(UsdPhysics.Joint):
            joints.append((str(prim.GetPath()), prim.GetName(), prim.GetTypeName()))
        if has_api(prim, UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(str(prim.GetPath()))

    print("\n[INFO] Articulation roots")
    if articulation_roots:
        for path in articulation_roots:
            print(f"  {path}")
    else:
        print("  none found")

    print("\n[INFO] Joints")
    for path, name, type_name in joints:
        print(f"  {name:<28} {type_name:<18} {path}")
    print(f"[INFO] Total joints: {len(joints)}")

    print("\n[INFO] Rigid bodies")
    for path in rigid_bodies:
        print(f"  {path}")
    print(f"[INFO] Total rigid bodies: {len(rigid_bodies)}")


def print_joint_name_check(stage):
    expected = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "zarx_j1", "zarx_j2", "zarx_j3", "zarx_j4", "zarx_j5", "zarx_j6",
    ]
    found = {prim.GetName() for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)}
    print("\n[INFO] Policy joint name check")
    for name in expected:
        status = "OK" if name in found else "MISSING"
        print(f"  {status:<7} {name}")


def create_visual_stage(usd_path, prim_path):
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    robot_prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    robot_prim.GetReferences().AddReference(str(usd_path))

    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(2500.0)
    return stage, robot_prim


def create_physics_stage(usd_path, prim_path):
    sim_cfg = SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.0, 1.4], [0.0, 0.0, 0.35])

    ground_cfg = sim_utils.GroundPlaneCfg(size=(6.0, 6.0))
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = sim_utils.UsdFileCfg(usd_path=str(usd_path))
    prim = robot_cfg.func(prim_path, robot_cfg, translation=(0.0, 0.0, 0.5))
    if not prim.IsValid():
        raise RuntimeError(f"Failed to spawn robot prim at {prim_path}")

    sim_utils.update_stage()
    return sim, prim


def main():
    usd_path = resolve_path(args_cli.usd)
    if not usd_path.exists():
        raise FileNotFoundError(f"USD file does not exist: {usd_path}")

    print(f"[INFO] Repo root: {REPO_ROOT}")
    print(f"[INFO] USD path: {usd_path}")
    print(f"[INFO] Headless: {args_cli.headless}")
    print(f"[INFO] Step physics: {args_cli.step_physics}")

    sim = None
    if args_cli.step_physics:
        sim, _ = create_physics_stage(usd_path, args_cli.prim_path)
        stage = omni.usd.get_context().get_stage()
    else:
        stage, _ = create_visual_stage(usd_path, args_cli.prim_path)

    if stage is None:
        raise RuntimeError("Failed to access current USD stage")

    print_stage_summary(stage, args_cli.max_prims)
    print_physics_summary(stage)
    print_joint_name_check(stage)

    if sim is not None:
        sim.reset()
    print("\n[INFO] Setup complete.")
    if args_cli.headless:
        print("[INFO] Headless inspection complete. Closing app.")
        return

    print("[INFO] GUI mode: close the Isaac Sim window to exit.")
    if args_cli.step_physics:
        print("[INFO] Stepping physics while rendering.")
    else:
        print("[INFO] UI-only mode: physics is not stepped. Use --step-physics to enable simulation stepping.")

    app = omni.kit.app.get_app_interface()
    while simulation_app.is_running():
        if args_cli.step_physics:
            sim.step(render=True)
        else:
            app.update()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
