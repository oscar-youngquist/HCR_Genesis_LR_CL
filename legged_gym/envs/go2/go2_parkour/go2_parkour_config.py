from legged_gym import *
from legged_gym.envs.base.legged_robot_parkour_config import LeggedRobotParkourCfg, LeggedRobotParkourCfgPPO



go2_parkour_action_scale = 0.5
go2_const_dof_range = dict(
    Hip_max= 1.0472,
    Hip_min= -1.0472,
    Front_Thigh_max= 3.4907,
    Front_Thigh_min= -1.5708,
    Rear_Thingh_max= 4.5379,
    Rear_Thingh_min= -0.5236,
    Calf_max= -0.83776,
    Calf_min= -2.7227,
)

class Go2ParkourCfg( LeggedRobotParkourCfg ):
    class env( LeggedRobotParkourCfg.env ):
        nums_envs = 4096
        num_actions = 12
        num_observations = 45  # num_obs
        _depth_res = LeggedRobotParkourCfg.sensor.depth_camera_config.resolution
        _depth_obs_len = int(_depth_res[0] * _depth_res[1])
        num_observations = 45 + _depth_obs_len  # proprio + flattened depth
        frame_stack = 5    # number of frames to stack for obs_history
        num_history_obs = int(num_observations * frame_stack)
        num_latent_dims = 161
        vae_kld_weight = 2.0
        num_explicit_dims = 24  # base linear velocity
        num_decoder_output = num_observations
        c_frame_stack = 5
        single_critic_obs_len = num_observations + 31 + 81 + 17 + 3
        num_privileged_obs = c_frame_stack * single_critic_obs_len

    class terrain( LeggedRobotParkourCfg.terrain ):
        selected = "TerrainPerlin"
        mesh_type = None
        measure_heights = True
        # x: [-0.5, 1.5], y: [-0.5, 0.5] range for go2
        measured_points_x = [i for i in np.arange(-0.5, 1.51, 0.1)]
        measured_points_y = [i for i in np.arange(-0.5, 0.51, 0.1)]
        horizontal_scale = 0.025 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 4.
        terrain_width = 4.
        num_rows= 16 # number of terrain rows (levels)
        num_cols = 16 # number of terrain cols (types)
        slope_treshold = 1.

        TerrainPerlin_kwargs = dict(
            zScale= 0.07,
            frequency= 10,
        )

    class sensor( LeggedRobotParkourCfg.sensor ):
        add_depth = True
        use_warp = False

    class commands( LeggedRobotParkourCfg.commands ):
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges( LeggedRobotParkourCfg.commands.ranges ):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
    class init_state( LeggedRobotParkourCfg.init_state ):
        pos = [0.0, 0.0, 0.5] # [m]
        default_joint_angles = { # 12 joints in the order of simulation
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": 0.7,
            "FL_calf_joint": -1.5,
            "FR_hip_joint": -0.1,
            "FR_thigh_joint": 0.7,
            "FR_calf_joint": -1.5,
            "RL_hip_joint": 0.1,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,
            "RR_hip_joint": -0.1,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,
        }
    class control( LeggedRobotParkourCfg.control ):
        stiffness = {'joint': 40.}
        damping = {'joint': 1.}
        action_scale = go2_parkour_action_scale
        computer_clip_torque = False
        motor_clip_torque = True
        # decimation = 4 # decimation: Number of control action updates @ sim DT per policy DT
        # dt = 0.02  # control frequency 50Hz
    class asset( LeggedRobotParkourCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        front_hip_names = ["FL_hip_joint", "FR_hip_joint"]
        rear_hip_names = ["RL_hip_joint", "RR_hip_joint"]
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        sdk_dof_range = go2_const_dof_range
        dof_velocity_override = 35.
    class termination( LeggedRobotParkourCfg.termination ):
        termination_terms = [
            "roll",
            "pitch",
        ]
        roll_kwargs = dict(
            threshold= 3.0, # [rad]
        )
        pitch_kwargs = dict(
            threshold= 3.0, # [rad] # for leap, jump
        )
    class domain_rand( LeggedRobotParkourCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0., 2.]
        randomize_base_mass = True
        added_mass_range = [1.0, 3.0]
        randomize_com_displacement = True
        com_pos_x_range = [-0.2, 1.5]
        com_pos_y_range = [-0.1, 0.1]
        com_pos_z_range = [-0.05, 0.05]
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        randomize_joint_armature = False
        joint_armature_range = [0.015, 0.025]  # [N*m*s/rad]
        randomize_joint_friction = False
        randomize_joint_damping = False
        joint_damping_range = [0.25, 0.3]
    class rewards( LeggedRobotParkourCfg.rewards ):
        class scales:
            tracking_lin_vel = 1.
            tracking_ang_vel = 1.
            energy_substeps = -2e-5
            stand_still = -2.
            dof_error_named = -1.
            dof_error = -0.01
            # penalty for hardware safety
            exceed_dof_pos_limits = -0.4
            exceed_torque_limits_l1norm = -0.4
            dof_vel_limits = -0.4
        dof_error_names = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"]
        only_positive_rewards = False
        soft_dof_vel_limit = 0.9
        soft_dof_pos_limit = 0.9

    class normalization( LeggedRobotParkourCfg.normalization ):
        class obs_scales( LeggedRobotParkourCfg.normalization.obs_scales ):
            lin_vel = 1.
        height_measurements_offset = -0.2
        clip_actions_method = None # let the policy learns not to exceed the limits

    class noise( LeggedRobotParkourCfg.noise ):
        add_noise = False

    class viewer( LeggedRobotParkourCfg.viewer ):
        pos = [-1., -1., 0.4]
        lookat = [0., 0., 0.3]

    class sim( LeggedRobotParkourCfg.sim ):
        body_measure_points = { # transform are related to body frame
            "base": dict(
                x= [i for i in np.arange(-0.24, 0.41, 0.03)],
                y= [-0.08, -0.04, 0.0, 0.04, 0.08],
                z= [i for i in np.arange(-0.061, 0.071, 0.03)],
                transform= [0., 0., 0.005, 0., 0., 0.],
            ),
            "thigh": dict(
                x= [
                    -0.16, -0.158, -0.156, -0.154, -0.152,
                    -0.15, -0.145, -0.14, -0.135, -0.13, -0.125, -0.12, -0.115, -0.11, -0.105, -0.1, -0.095, -0.09, -0.085, -0.08, -0.075, -0.07, -0.065, -0.05,
                    0.0, 0.05, 0.1,
                ],
                y= [-0.015, -0.01, 0.0, -0.01, 0.015],
                z= [-0.03, -0.015, 0.0, 0.015],
                transform= [0., 0., -0.1,   0., 1.57079632679, 0.],
            ),
            "calf": dict(
                x= [i for i in np.arange(-0.13, 0.111, 0.03)],
                y= [-0.015, 0.0, 0.015],
                z= [-0.015, 0.0, 0.015],
                transform= [0., 0., -0.11,   0., 1.57079632679, 0.],
            ),
        }

logs_root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))), "logs")
class Go2ParkourCfgPPO( LeggedRobotParkourCfgPPO ):
    class algorithm( LeggedRobotParkourCfgPPO.algorithm ):
        entropy_coef = 0.01
        clip_min_std = 0.2
        learning_rate = 1e-4
        optimizer_class_name = "AdamW"
    class policy( LeggedRobotParkourCfgPPO.policy ):
        critic_hidden_dims = [1024, 256, 128]
        encoder_hidden_dims = [256, 128]
        decoder_hidden_dims = [256, 128]
    class algorithm( LeggedRobotParkourCfgPPO.algorithm ):
        encoder_lr = 2.e-4
        num_encoder_epochs = 1
        vae_kld_weight = 2.0
    class runner( LeggedRobotParkourCfgPPO.runner ):
        run_name = "parkour"
        policy_class_name = "EncoderStateAcRecurrent"
        algorithm_class_name = "PPO_Parkour"
        experiment_name = "parkour_go2"
        save_interval = 2000
        max_iterations = 2000
        if SIMULATOR == "genesis":
            run_name += "_genesis"
        elif SIMULATOR == "isaacgym":
            run_name += "_isaacgym"
        elif SIMULATOR == "isaaclab":
            run_name += "_isaaclab"