from legged_gym.envs.base.legged_robot_cts_config import LeggedRobotCTSCfg, LeggedRobotCTSCfgPPO
from legged_gym import *

class Go2CTSCfg( LeggedRobotCTSCfg ):
    class env( LeggedRobotCTSCfg.env ):
        num_envs = 4096
        num_teacher = num_envs // 4 * 3 # a quarter of the total environments are student envs, the rest are teacher envs
        num_observations = 45  # num_obs
        num_privileged_obs = 94
        frame_stack = 20    # number of frames to stack for obs_history
        num_history_obs = int(num_observations * frame_stack)
        num_latent_dims = num_privileged_obs
        c_frame_stack = 5
        single_critic_obs_len = num_observations + 31 + 81 + 12 + 3
        num_critic_obs = c_frame_stack * single_critic_obs_len
        num_actions = 12
        env_spacing = 0.5
    
    class terrain( LeggedRobotCTSCfg.terrain ):
        if SIMULATOR == "genesis":
            mesh_type = "heightfield"
        else:
            mesh_type = "trimesh"
        restitution = 0.
        border_size = 10.0 # [m]
        curriculum = True
        # rough terrain only:
        obtain_terrain_info_around_feet = True
        measure_heights = True
        measured_points_x = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4] # 9x9=81
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.1, 0.2, 0.2, 0.2, 0.0, 0.0, 0.1]
        
    class init_state( LeggedRobotCTSCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 0.8,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 0.8,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCTSCfg.control ):
        # PD Drive parameters:
        # control_type = 'P'
        stiffness = {'joint': 20.}   # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        action_scale = 0.25 # action scale: target angle = actionScale * action + defaultAngle
        dt =  0.02  # control frequency 50Hz
        decimation = 4 # decimation: Number of control action updates @ sim DT per policy DT

    class asset( LeggedRobotCTSCfg.asset ):
        # Common: 
        name = "go2"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        obtain_link_contact_states = True
        contact_state_link_names = ["thigh", "calf", "foot"]
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base", "Head"]
        terminate_after_contacts_on = []
        # Genesis: 
        dof_names = [        # specify the sequence of actions
            'FR_hip_joint',
            'FR_thigh_joint',
            'FR_calf_joint',
            'FL_hip_joint',
            'FL_thigh_joint',
            'FL_calf_joint',
            'RR_hip_joint',
            'RR_thigh_joint',
            'RR_calf_joint',
            'RL_hip_joint',
            'RL_thigh_joint',
            'RL_calf_joint',]
        links_to_keep = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        # IsaacGym:
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCTSCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.34
        foot_clearance_target = 0.09 # desired foot clearance above ground [m]
        foot_height_offset = 0.022   # height of the foot coordinate origin above ground [m]
        foot_clearance_tracking_sigma = 0.01
        only_positive_rewards = True
        class scales( LeggedRobotCTSCfg.rewards.scales ):
            # limitation
            dof_pos_limits = -5.0
            collision = -1.0
            # command tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # smooth
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_power = -2.e-4
            dof_acc = -2.e-7
            action_rate = -0.01
            action_smoothness = -0.01
            # gait
            feet_air_time = 1.0
            foot_clearance = 0.2
            feet_contact_stand_still = 0.5

    class commands( LeggedRobotCTSCfg.commands ):
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges( LeggedRobotCTSCfg.commands.ranges ):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
            
    class domain_rand(LeggedRobotCTSCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.7]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.
        randomize_com_displacement = True
        com_pos_x_range = [-0.03, 0.03]
        com_pos_y_range = [-0.03, 0.03]
        com_pos_z_range = [-0.03, 0.03]
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        randomize_joint_armature = False
        joint_armature_range = [0.015, 0.025]  # [N*m*s/rad]
        randomize_joint_stiffness = False
        joint_stiffness_range = [0.01, 0.02]
        randomize_joint_damping = False
        joint_damping_range = [0.25, 0.3]

class Go2CTSCfgPPO( LeggedRobotCTSCfgPPO ):
    class policy( LeggedRobotCTSCfgPPO.policy ):
        critic_hidden_dims = [1024, 256, 128]
        privilege_encoder_hidden_dims = [256, 128]
        history_encoder_hidden_dims = [256, 128]
    class algorithm( LeggedRobotCTSCfgPPO.algorithm ):
        encoder_lr = 5.e-4
        num_encoder_epochs = 1
    class runner( LeggedRobotCTSCfgPPO.runner ):
        run_name = 'gs_cts'
        experiment_name = 'go2_rough'
        save_interval = 500
        load_run = "Oct17_11-08-37_gs_cts"
        checkpoint = -1
        max_iterations = 4000