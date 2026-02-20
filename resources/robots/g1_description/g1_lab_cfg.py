"""Isaac Lab actuator config for Unitree G1 12-DOF legs."""
from isaaclab.actuators import DCMotorCfg

# Effort/velocity from g1_12dof.urdf; stiffness/damping applied in _compute_torques
G1_ACTUATOR_CFG = {
    "hip_pitch": DCMotorCfg(
        joint_names_expr=[".*_hip_pitch_joint"],
        effort_limit=88.0,
        saturation_effort=88.0,
        velocity_limit=32.0,
        stiffness=0.0,
        damping=0.0,
        friction=0.0,
    ),
    "hip_roll": DCMotorCfg(
        joint_names_expr=[".*_hip_roll_joint"],
        effort_limit=139.0,
        saturation_effort=139.0,
        velocity_limit=20.0,
        stiffness=0.0,
        damping=0.0,
        friction=0.0,
    ),
    "hip_yaw": DCMotorCfg(
        joint_names_expr=[".*_hip_yaw_joint"],
        effort_limit=88.0,
        saturation_effort=88.0,
        velocity_limit=32.0,
        stiffness=0.0,
        damping=0.0,
        friction=0.0,
    ),
    "knee": DCMotorCfg(
        joint_names_expr=[".*_knee_joint"],
        effort_limit=139.0,
        saturation_effort=139.0,
        velocity_limit=20.0,
        stiffness=0.0,
        damping=0.0,
        friction=0.0,
    ),
    "ankle": DCMotorCfg(
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        effort_limit=50.0,
        saturation_effort=50.0,
        velocity_limit=37.0,
        stiffness=0.0,
        damping=0.0,
        friction=0.0,
    ),
}
