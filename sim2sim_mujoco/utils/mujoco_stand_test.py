import mujoco
import mujoco.viewer
from pathlib import Path
import numpy as np

def main():
    sim2sim_root = Path(__file__).resolve().parents[1]
    model_path = sim2sim_root / "resources/robots/arx5p2Go1_mujoco/mjcf/arx5p2Go1_mujoco.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))

    data = mujoco.MjData(model)
    
    print("nq", model.nq)
    print("nv", model.nv)
    print("nu", model.nu)
    assert model.nq == 27
    assert model.nv == 26
    assert model.nu == 18
    
    data.qpos[0:3] = [0.0, 0.0, 0.34]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    
    policy_qpos_addr_20 = [
        10, 11, 12,
        7, 8, 9,
        16, 17, 18,
        13, 14, 15,
        19, 20, 21, 22, 23, 24, 25, 26,
    ]
    
    default_dof_pos_20 = [
        0.1, 0.8, -1.5,
        -0.1, 0.8, -1.5,
        0.1, 1.0, -1.5,
        -0.1, 1.0, -1.5,
        0.0, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    
    data.qpos[policy_qpos_addr_20] = default_dof_pos_20
    
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0   
    mujoco.mj_forward(model, data)
    
    policy_qpos_addr_18 = policy_qpos_addr_20[:18]
    policy_qvel_addr_18 = [
        9, 10, 11,
        6, 7, 8,
        15, 16, 17,
        12, 13, 14,
        18, 19, 20, 21, 22, 23,
    ]
    default_dof_pos_18 = default_dof_pos_20[:18]
    
    kp = [
        35, 35, 35,
        35, 35, 35,
        35, 35, 35,
        35, 35, 35,
        50, 50, 70, 50, 50, 50,
    ]
    kd = [
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
        20, 20, 15, 20, 20, 20,
    ]
    
    policy_qpos_addr_18 = np.array(policy_qpos_addr_18)
    policy_qvel_addr_18 = np.array(policy_qvel_addr_18)
    default_dof_pos_18 = np.array(default_dof_pos_18)
    kp = np.array(kp)
    kd = np.array(kd)
    

    
    with mujoco.viewer.launch_passive(model, data) as viewer:

        for step in range(3000):
            q = data.qpos[policy_qpos_addr_18]
            qd = data.qvel[policy_qvel_addr_18]
            target_q = default_dof_pos_18
            torque = kp * (target_q - q) - kd * qd
            ctrl_min = model.actuator_ctrlrange[:, 0]
            ctrl_max = model.actuator_ctrlrange[:, 1]
            torque = np.clip(torque, ctrl_min, ctrl_max)
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
            

            
            
            viewer.sync()
            
            if step % 100 == 0:
                base_z = data.qpos[2]
                max_ctrl = np.max(np.abs(data.ctrl))
                max_err = np.max(np.abs(target_q - q))
                print(step, "base_z", base_z, "max_ctrl", max_ctrl, "max_err", max_err)
                
            if not np.isfinite(data.qpos).all():
                print("qpos exploded")
                break
            if data.qpos[2] < 0.05:
                print("base too low, likely fallen")
                break
    

if __name__ == "__main__":
    main()
