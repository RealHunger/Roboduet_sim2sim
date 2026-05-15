from pathlib import Path

import mujoco

path = Path(__file__).resolve().parents[1] / "resources/robots/arx5p2Go1_mujoco/urdf/arx5p2Go1_mujoco.urdf"
model = mujoco.MjModel.from_xml_path(path)
print("nq", model.nq)
print("nv", model.nv)
print("nu", model.nu)
print("nbody", model.nbody)
print("njnt", model.njnt)
print()

for joint_id in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    jtype = model.jnt_type[joint_id]
    qpos_addr = model.jnt_qposadr[joint_id]
    qvel_addr = model.jnt_dofadr[joint_id]
    joint_range = model.jnt_range[joint_id]
    print(
        f"{joint_id:02d} "
        f"name={name} "
        f"type={jtype} "
        f"qpos[{qpos_addr}] "
        f"qvel[{qvel_addr}] "
        f"range=[{joint_range[0]:.6f}, {joint_range[1]:.6f}]"
    )
    
policy_to_mujoco = [
    3,   # FL_hip_joint
    4,   # FL_thigh_joint
    5,   # FL_calf_joint
    0,   # FR_hip_joint
    1,   # FR_thigh_joint
    2,   # FR_calf_joint
    9,   # RL_hip_joint
    10,  # RL_thigh_joint
    11,  # RL_calf_joint
    6,   # RR_hip_joint
    7,   # RR_thigh_joint
    8,   # RR_calf_joint
    12,  # zarx_j1
    13,  # zarx_j2
    14,  # zarx_j3
    15,  # zarx_j4
    16,  # zarx_j5
    17,  # zarx_j6
    18,  # zarx_j7
    19,  # zarx_j8
]

mujoco_to_policy = [
    3,   # FR_hip_joint
    4,   # FR_thigh_joint
    5,   # FR_calf_joint
    0,   # FL_hip_joint
    1,   # FL_thigh_joint
    2,   # FL_calf_joint
    9,   # RR_hip_joint
    10,  # RR_thigh_joint
    11,  # RR_calf_joint
    6,   # RL_hip_joint
    7,   # RL_thigh_joint
    8,   # RL_calf_joint
    12,  # zarx_j1
    13,  # zarx_j2
    14,  # zarx_j3
    15,  # zarx_j4
    16,  # zarx_j5
    17,  # zarx_j6
    18,  # zarx_j7
    19,  # zarx_j8
]
