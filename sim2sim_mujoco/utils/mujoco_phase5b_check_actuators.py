from pathlib import Path

import mujoco


SIM2SIM_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = SIM2SIM_ROOT / "resources/robots/arx5p2Go1_mujoco/mjcf/arx5p2Go1_mujoco.xml"


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    print("model:", MODEL_PATH)
    print("nq", model.nq)
    print("nv", model.nv)
    print("nu", model.nu)
    print("nbody", model.nbody)
    print("njnt", model.njnt)
    print("ctrl shape", data.ctrl.shape)
    print()

    print("joints:")
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        jtype = model.jnt_type[joint_id]
        qpos_addr = model.jnt_qposadr[joint_id]
        qvel_addr = model.jnt_dofadr[joint_id]
        joint_range = model.jnt_range[joint_id]
        print(
            f"{joint_id:02d} name={name} type={jtype} "
            f"qpos[{qpos_addr}] qvel[{qvel_addr}] "
            f"range=[{joint_range[0]:.6f}, {joint_range[1]:.6f}]"
        )

    print()
    print("actuators:")
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        ctrlrange = model.actuator_ctrlrange[actuator_id]
        trnid = model.actuator_trnid[actuator_id]
        joint_id = trnid[0]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        print(
            f"{actuator_id:02d} name={name} joint={joint_name} "
            f"ctrlrange=[{ctrlrange[0]:.6f}, {ctrlrange[1]:.6f}]"
        )


if __name__ == "__main__":
    main()
