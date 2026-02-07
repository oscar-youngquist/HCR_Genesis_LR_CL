from legged_gym.envs.base.legged_robot_ee_config import LeggedRobotEECfg, LeggedRobotEECfgPPO
from legged_gym import SIMULATOR

# TRON1 Sole Foot
class TRON1SF_EECfg( LeggedRobotEECfg ):
    class env( LeggedRobotEECfg.env ):
        num_envs = 4096
        num_single_obs = 37
        frame_stack = 10    # Policy frame stack number
        num_estimator_features = int( num_single_obs * frame_stack )
        num_estimator_labels = 3+9+2+6
        c_frame_stack = 10  # Critic frame stack number
        single_critic_obs_len = num_single_obs + 26 + 2 + 9 + 49 + 6 + 18
        num_privileged_obs = int( single_critic_obs_len * c_frame_stack )
        num_actions = 8
        env_spacing = 2.0
    
    class terrain( LeggedRobotEECfg.terrain ):
        if SIMULATOR == "genesis":
            mesh_type = "heightfield" # for genesis
        else:
            mesh_type = "trimesh"  # for isaacgym
        restitution = 0.
        border_size = 15.0 # [m]
        curriculum = True
        # rough terrain only:
        obtain_terrain_info_around_feet = True
        measure_heights = True
        measured_points_x = [-0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3] # 7x7=49
        measured_points_y = [-0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3]
        terrain_length = 8.0
        terrain_width = 8.0
        platform_size = 4.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
        
    class init_state( LeggedRobotEECfg.init_state ):
        pos = [0.0, 0.0, 0.85]     # x,y,z [m]
        default_joint_angles = {  # target angles when action = 0.0
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "ankle_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "ankle_R_Joint": 0.0
        }
        # sit mode pose
        sit_pos = [0.0, 0.0, 0.6] # x,y,z [m]
        sit_joint_angles = {  # target angles when action = 0.0
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.58,
            "knee_L_Joint": 1.35,
            "ankle_L_Joint": -0.8,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": -0.58,
            "knee_R_Joint": -1.35,
            "ankle_R_Joint": -0.8
        }
        sit_init_percent = 0.5  # probability of resetting env in sit pose

    class control( LeggedRobotEECfg.control ):
        # PD Drive parameters:
        # control_type = 'P'
        stiffness = {
            "abad_L_Joint": 45,
            "hip_L_Joint": 45,
            "knee_L_Joint": 45,
            "abad_R_Joint": 45,
            "hip_R_Joint": 45,
            "knee_R_Joint": 45,

            "ankle_L_Joint": 45,
            "ankle_R_Joint": 45,
        }  # [N*m/rad]
        damping = {
            "abad_L_Joint": 1.5,
            "hip_L_Joint": 1.5,
            "knee_L_Joint": 1.5,
            "abad_R_Joint": 1.5,
            "hip_R_Joint": 1.5,
            "knee_R_Joint": 1.5,

            "ankle_L_Joint": 0.8,
            "ankle_R_Joint": 0.8,
        }  # [N*m*s/rad]
        action_scale = 0.25 # action scale: target angle = actionScale * action + defaultAngle
        decimation = 4      # decimation: Number of control action updates @ sim DT per policy DT

    class asset( LeggedRobotEECfg.asset ):
        # Common
        name = "tron1_sf"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/SF_TRON1A/urdf/robot.urdf'
        foot_name = "ankle"
        penalize_contacts_on = ["knee", "hip"]
        terminate_after_contacts_on = ["base", "abad"]
        obtain_link_contact_states = True
        contact_state_link_names = ["base", "abad", "hip", "knee", "ankle"]
        # For Genesis
        dof_names = [           # align with the real robot
            "abad_L_Joint",
            "hip_L_Joint",
            "knee_L_Joint",
            "ankle_L_Joint",
            "abad_R_Joint",
            "hip_R_Joint",
            "knee_R_Joint",
            "ankle_R_Joint",
        ]
        links_to_keep = []
        # For IsaacGym
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
  
    class rewards( LeggedRobotEECfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.75
        foot_clearance_target = 0.1  # desired foot clearance above ground [m]
        foot_height_offset = 0.055   # height of the foot coordinate origin above ground [m]
        foot_clearance_tracking_sigma = 0.01
        base_height_tracking_sigma = 0.01
        foot_distance_threshold = 0.115
        max_projected_gravity = -0.4
        only_positive_rewards = False
        class scales( LeggedRobotEECfg.rewards.scales ):
            # limitation
            keep_balance = 1.0
            dof_pos_limits = -2.0
            collision = -1.0
            feet_distance = -100.0
            # command tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            tracking_base_height = 0.3
            # smooth
            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            orientation = -5.0
            dof_power = -2.e-4
            dof_acc = -2.e-7
            foot_acc = -1.e-5
            action_rate = -0.01
            action_smoothness = -0.01
            # gait
            foot_clearance = 0.5
            hip_pos_zero_command = -10.0
            ankle_torque_limits = -0.1
            biped_periodic_gait = 1.0
        
        class periodic_reward_framework:
            '''Periodic reward framework in OSU's paper(https://arxiv.org/abs/2011.01387)'''
            gait_function_type = "step" # can be "step" or "smooth"
            kappa = 20
            # start of swing(a_swing) is all the same
            b_swing = 0.5
            # phase offset of left and right legs
            theta_left = 0.0
            theta_right = 0.5
            gait_period = 0.5  # [s]
    
    class commands( LeggedRobotEECfg.commands ):
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges( LeggedRobotEECfg.commands.ranges ):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class domain_rand( LeggedRobotEECfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.0, 1.7]
        randomize_base_mass = True
        added_mass_range = [-0.5, 1.]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 0.5
        randomize_com_displacement = True
        com_pos_x_range = [-0.03, 0.03]
        com_pos_y_range = [-0.03, 0.03]
        com_pos_z_range = [-0.03, 0.03]
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        # armature/damping/friction randomization, only used in isaacgym
        randomize_joint_armature = True
        joint_armature_range = [0.11, 0.13]
        randomize_joint_friction = True
        joint_friction_range = [0.00, 0.01]
        randomize_joint_damping = True
        joint_damping_range = [1.4, 1.45]

class TRON1SF_EECfgPPO( LeggedRobotEECfgPPO ):
    class runner( LeggedRobotEECfgPPO.runner ):
        run_name = 'ee'
        if SIMULATOR == "genesis":
            run_name += "_genesis"
        elif SIMULATOR == "isaacgym":
            run_name += "_isaacgym"
        experiment_name = 'tron1_sf_rough'
        save_interval = 500
        max_iterations = 5000