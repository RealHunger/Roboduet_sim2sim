from pathlib import Path

import mujoco


SIM2SIM_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = SIM2SIM_ROOT / "resources/robots/arx5p2Go1_mujoco/urdf/arx5p2Go1_mujoco.urdf"
MJCF_PATH = SIM2SIM_ROOT / "resources/robots/arx5p2Go1_mujoco/mjcf/arx5p2Go1_mujoco.xml"


ACTUATOR_XML = """
  <actuator>
    <motor name="FL_hip_motor" joint="FL_hip_joint" gear="1" ctrllimited="true" ctrlrange="-23.7 23.7"/>
    <motor name="FL_thigh_motor" joint="FL_thigh_joint" gear="1" ctrllimited="true" ctrlrange="-23.7 23.7"/>
    <motor name="FL_calf_motor" joint="FL_calf_joint" gear="1" ctrllimited="true" ctrlrange="-23.7 23.7"/>
    <motor name="FR_hip_motor" joint="FR_hip_joint" gear="1" ctrllimited="true" ctrlrange="-23.7 23.7"/>
    <motor name="FR_thigh_motor" joint="FR_thigh_joint" gear="1" ctrllimited="true" ctrlrange="-23.7 23.7"/>
    <motor name="FR_calf_motor" joint="FR_calf_joint" gear="1" ctrllimited="true" ctrlrange="-23.7 23.7"/>
    <motor name="RL_hip_motor" joint="RL_hip_joint" gear="1" ctrllimited="true" ctrlrange="-23.7 23.7"/>
    <motor name="RL_thigh_motor" joint="RL_thigh_joint" gear="1" ctrllimited="true" ctrlrange="-23.7 23.7"/>
    <motor name="RL_calf_motor" joint="RL_calf_joint" gear="1" ctrllimited="true" ctrlrange="-23.7 23.7"/>
    <motor name="RR_hip_motor" joint="RR_hip_joint" gear="1" ctrllimited="true" ctrlrange="-23.7 23.7"/>
    <motor name="RR_thigh_motor" joint="RR_thigh_joint" gear="1" ctrllimited="true" ctrlrange="-23.7 23.7"/>
    <motor name="RR_calf_motor" joint="RR_calf_joint" gear="1" ctrllimited="true" ctrlrange="-23.7 23.7"/>
    <position name="zarx_j1_position" joint="zarx_j1" kp="50" ctrllimited="true" ctrlrange="-2.61799 3.14159"/>
    <position name="zarx_j2_position" joint="zarx_j2" kp="50" ctrllimited="true" ctrlrange="0 3.66518"/>
    <position name="zarx_j3_position" joint="zarx_j3" kp="70" ctrllimited="true" ctrlrange="0 3.14158"/>
    <position name="zarx_j4_position" joint="zarx_j4" kp="50" ctrllimited="true" ctrlrange="-1.57079 1.57079"/>
    <position name="zarx_j5_position" joint="zarx_j5" kp="50" ctrllimited="true" ctrlrange="-1.57079 1.57079"/>
    <position name="zarx_j6_position" joint="zarx_j6" kp="50" ctrllimited="true" ctrlrange="-1.57079 1.57079"/>
  </actuator>
"""

CONTACT_PARAMS = 'friction="1.0 0.1 0.01" condim="6" solref="0.02 1" solimp="0.9 0.95 0.001"'
LEG_COLLISION_RGBA = 'rgba="0.913725 0.913725 0.847059 0"'

GROUND_XML = f"""    <geom name="ground" type="plane" pos="0 0 0" size="5 5 0.1" {CONTACT_PARAMS} rgba="0.8 0.8 0.8 1"/>
"""

OPTION_XML = """  <option timestep="0.005"/>
"""

GO1_LEG_MESH_ASSETS = """    <mesh name="hip" file="../meshes/hip.obj"/>
    <mesh name="thigh" file="../meshes/thigh.obj"/>
    <mesh name="thigh_mirror" file="../meshes/thigh_mirror.obj"/>
    <mesh name="calf" file="../meshes/calf.obj"/>
"""

ARM_TARGET_MARKER_XML = """    <body name="arm_target_marker" mocap="true" pos="0.5 0 0.5">
      <geom name="arm_target_marker_geom" type="sphere" size="0.035" contype="0" conaffinity="0" rgba="1 0.1 0.1 0.75"/>
    </body>
"""

ARM_EE_MARKER_XML = """    <body name="arm_ee_marker" mocap="true" pos="0.5 0 0.5">
      <geom name="arm_ee_marker_geom" type="sphere" size="0.025" contype="0" conaffinity="0" rgba="0.1 0.35 1 0.75"/>
    </body>
"""

ARM_MARKERS_XML = ARM_TARGET_MARKER_XML + ARM_EE_MARKER_XML

GO1_LEG_VISUALS = {
    "FR_hip": '        <geom type="mesh" mesh="hip" quat="0 1 0 0" contype="0" conaffinity="0" rgba="0.913725 0.913725 0.847059 1"/>\n',
    "FL_hip": '        <geom type="mesh" mesh="hip" contype="0" conaffinity="0" rgba="0.913725 0.913725 0.847059 1"/>\n',
    "RR_hip": '        <geom type="mesh" mesh="hip" quat="0 1 0 0" contype="0" conaffinity="0" rgba="0.913725 0.913725 0.847059 1"/>\n',
    "RL_hip": '        <geom type="mesh" mesh="hip" contype="0" conaffinity="0" rgba="0.913725 0.913725 0.847059 1"/>\n',
    "FR_thigh": '          <geom type="mesh" mesh="thigh_mirror" contype="0" conaffinity="0" rgba="0.913725 0.913725 0.847059 1"/>\n',
    "FL_thigh": '          <geom type="mesh" mesh="thigh" contype="0" conaffinity="0" rgba="0.913725 0.913725 0.847059 1"/>\n',
    "RR_thigh": '          <geom type="mesh" mesh="thigh_mirror" contype="0" conaffinity="0" rgba="0.913725 0.913725 0.847059 1"/>\n',
    "RL_thigh": '          <geom type="mesh" mesh="thigh" contype="0" conaffinity="0" rgba="0.913725 0.913725 0.847059 1"/>\n',
    "FR_calf": '            <geom type="mesh" mesh="calf" contype="0" conaffinity="0" rgba="0.913725 0.913725 0.847059 1"/>\n',
    "FL_calf": '            <geom type="mesh" mesh="calf" contype="0" conaffinity="0" rgba="0.913725 0.913725 0.847059 1"/>\n',
    "RR_calf": '            <geom type="mesh" mesh="calf" contype="0" conaffinity="0" rgba="0.913725 0.913725 0.847059 1"/>\n',
    "RL_calf": '            <geom type="mesh" mesh="calf" contype="0" conaffinity="0" rgba="0.913725 0.913725 0.847059 1"/>\n',
}


def add_body_visual(xml, body_name, visual_xml):
    body_tag = f'<body name="{body_name}"'
    start = xml.find(body_tag)
    if start < 0:
        return xml
    next_body = xml.find("<body", start + len(body_tag))
    joint_pos = xml.find("<joint", start)
    if joint_pos < 0 or (next_body >= 0 and joint_pos > next_body):
        return xml
    insert_pos = xml.find("\n", joint_pos)
    if insert_pos < 0:
        return xml
    body_end = next_body if next_body >= 0 else xml.find("</body>", start)
    if body_end > 0 and visual_xml.strip() in xml[start:body_end]:
        return xml
    return xml[:insert_pos + 1] + visual_xml + xml[insert_pos + 1:]


def main():
    model = mujoco.MjModel.from_xml_path(str(URDF_PATH))
    MJCF_PATH.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(MJCF_PATH), model)

    xml = MJCF_PATH.read_text()
    if 'mesh name="hip"' not in xml:
        xml = xml.replace("    <mesh name=\"trunk\" file=\"../meshes/trunk.obj\"/>\n", "    <mesh name=\"trunk\" file=\"../meshes/trunk.obj\"/>\n" + GO1_LEG_MESH_ASSETS, 1)
    if "<option" not in xml:
        xml = xml.replace("  <compiler angle=\"radian\"/>\n", "  <compiler angle=\"radian\"/>\n" + OPTION_XML, 1)
    if "name=\"ground\"" not in xml:
        xml = xml.replace("  <worldbody>\n", "  <worldbody>\n" + GROUND_XML, 1)
    if "name=\"arm_target_marker\"" not in xml:
        xml = xml.replace("  <worldbody>\n", "  <worldbody>\n" + ARM_MARKERS_XML, 1)
    elif "name=\"arm_ee_marker\"" not in xml:
        xml = xml.replace("    <body name=\"trunk\"", ARM_EE_MARKER_XML + "    <body name=\"trunk\"", 1)
    xml = xml.replace(
        '<geom name="ground" type="plane" pos="0 0 0" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>',
        f'<geom name="ground" type="plane" pos="0 0 0" size="5 5 0.1" {CONTACT_PARAMS} rgba="0.8 0.8 0.8 1"/>',
    )
    xml = xml.replace(
        '<geom size="0.02" pos="0 0 -0.213" rgba="0.913725 0.913725 0.847059 1"/>',
        f'<geom size="0.02" pos="0 0 -0.213" {CONTACT_PARAMS} {LEG_COLLISION_RGBA}/>',
    )
    xml = xml.replace('type="box" rgba="0.913725 0.913725 0.847059 1"', f'type="box" {LEG_COLLISION_RGBA}')
    xml = xml.replace('solimp="0.9 0.95 0.001" rgba="0.913725 0.913725 0.847059 1"', f'solimp="0.9 0.95 0.001" {LEG_COLLISION_RGBA}')
    for body_name, visual_xml in GO1_LEG_VISUALS.items():
        xml = add_body_visual(xml, body_name, visual_xml)
    if "<actuator>" not in xml:
        xml = xml.replace("</mujoco>", ACTUATOR_XML + "</mujoco>")
    MJCF_PATH.write_text(xml)

    model_with_actuators = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    print("wrote", MJCF_PATH)
    print("nq", model_with_actuators.nq)
    print("nv", model_with_actuators.nv)
    print("nu", model_with_actuators.nu)
    print("nbody", model_with_actuators.nbody)
    print("njnt", model_with_actuators.njnt)

    for actuator_id in range(model_with_actuators.nu):
        name = mujoco.mj_id2name(model_with_actuators, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        trnid = model_with_actuators.actuator_trnid[actuator_id]
        joint_name = mujoco.mj_id2name(model_with_actuators, mujoco.mjtObj.mjOBJ_JOINT, trnid[0])
        ctrlrange = model_with_actuators.actuator_ctrlrange[actuator_id]
        print(
            f"{actuator_id:02d} name={name} joint={joint_name} "
            f"ctrlrange=[{ctrlrange[0]:.6f}, {ctrlrange[1]:.6f}]"
        )


if __name__ == "__main__":
    main()
