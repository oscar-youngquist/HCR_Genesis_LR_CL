from legged_gym import *
from legged_gym.simulator.simulator import Simulator
from PIL import Image as im
import cv2 as cv
import torch
import numpy as np
import os
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math_utils import *
# from isaacgym.torch_utils import *
if SIMULATOR == "isaaclab":
    from isaaclab.app import AppLauncher
    import carb
    GROUND_PATH = "/World/ground"

""" ********** IsaacLab Simulator ********** """
class IsaacLabSimulator(Simulator):
    """Simulator class for IsaacLab"""
    def __init__(self, cfg, sim_params: dict, device, headless):
        self.sim_params = sim_params
        super().__init__(cfg, sim_params, device, headless)
    
    def _parse_cfg(self):
        self.debug = self.cfg.env.debug
        self.control_dt = self.cfg.sim.dt * self.cfg.control.decimation
        if self.cfg.sensor.add_depth:
            self.frame_count = 0
            
    def _create_sim(self):
        self.app_launcher = AppLauncher({"headless": self.headless, "device": self.device})
        
        import isaaclab.sim as sim_utils
        from isaacsim.core.utils.stage import get_current_stage
        
        sim_cfg = sim_utils.SimulationCfg(device=self.device, dt=self.sim_params["dt"],
                                          render_interval=self.cfg.control.decimation)

        sim_cfg.physx.bounce_threshold_velocity = 0.2
        sim_cfg.physx.max_position_iteration_count = 4
        sim_cfg.physx.max_velocity_iteration_count = 0
        sim_cfg.physx.gpu_max_rigid_contact_count = 8 * 1024 * 1024
        sim_cfg.physics_material.static_friction = 1.0
        sim_cfg.physics_material.dynamic_friction = 1.0
        
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.stage = get_current_stage()
        
        # disable delays during rendering
        carb_settings = carb.settings.get_settings()
        carb_settings.set_bool("/app/runLoops/main/rateLimitEnabled", False)

        # add terrain
        mesh_type = self.cfg.terrain.mesh_type
        from isaaclab.terrains import TerrainImporterCfg, TerrainImporter
        import isaaclab.sim as sim_utils
        if mesh_type == "plane":
            
            # from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
            # import omni.kit.commands
            # from pxr import UsdPhysics
            
            # ground_col = np.array([1.0, 0.9, 0.75])
            # ground_col *= 0.017
            # ground_path = GROUND_PATH

            # physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=self.cfg.terrain.static_friction, 
            #                                                 dynamic_friction=self.cfg.terrain.dynamic_friction,
            #                                                 restitution=self.cfg.terrain.restitution)
            # plane_cfg = GroundPlaneCfg(physics_material=physics_material, color=ground_col)
            # self.ground = spawn_ground_plane(prim_path=ground_path, cfg=plane_cfg)

            # # add rigid body schema to terrain to enable contact sensors
            # UsdPhysics.RigidBodyAPI.Apply(self.stage.GetPrimAtPath(ground_path))
            # UsdPhysics.RigidBodyAPI.Get(self.stage, ground_path).GetKinematicEnabledAttr().Set(True)

            # shader_path = ground_path + "/Looks/theGrid/Shader"
            # shader_prim = self.stage.GetPrimAtPath(shader_path)
            # shader_prim.GetAttribute("inputs:albedo_add").Set(10.0)
            
            terrain_cfg = TerrainImporterCfg(
                num_envs=self.num_envs,
                env_spacing=self.cfg.env.env_spacing,
                prim_path=GROUND_PATH,
                terrain_type="plane",
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=self.cfg.terrain.static_friction,
                    dynamic_friction=self.cfg.terrain.dynamic_friction,
                    restitution=self.cfg.terrain.restitution
                )
            )
            self.terrain = TerrainImporter(terrain_cfg)
        elif mesh_type == "heightfield":
            raise NotImplementedError("Heightfield terrain not implemented for IsaacLabSimulator yet")
        elif mesh_type == "trimesh":
            raise NotImplementedError("Trimesh terrain not implemented for IsaacLabSimulator yet")
        else:
            raise NameError(f"Unknown terrain mesh type: {mesh_type}")
        
        # specify the boundary of the heightfield
        self.terrain_x_range = torch.zeros(2, device=self.device)
        self.terrain_y_range = torch.zeros(2, device=self.device)
        if self.cfg.terrain.mesh_type == 'heightfield':
            # give a small margin(1.0m)
            self.terrain_x_range[0] = -self.cfg.terrain.border_size + 1.0
            self.terrain_x_range[1] = self.cfg.terrain.border_size + \
                self.cfg.terrain.num_rows * self.cfg.terrain.terrain_length - 1.0
            self.terrain_y_range[0] = -self.cfg.terrain.border_size + 1.0
            self.terrain_y_range[1] = self.cfg.terrain.border_size + \
                self.cfg.terrain.num_cols * self.cfg.terrain.terrain_width - 1.0
        elif self.cfg.terrain.mesh_type == 'plane':  # the plane used has limited size,
            # and the origin of the world is at the center of the plane
            self.terrain_x_range[0] = -self.cfg.terrain.plane_length/2+1
            self.terrain_x_range[1] = self.cfg.terrain.plane_length/2-1
            # the plane is a square
            self.terrain_y_range[0] = -self.cfg.terrain.plane_length/2+1
            self.terrain_y_range[1] = self.cfg.terrain.plane_length/2-1
        
    def _create_envs(self):
        """ Creates environments, adds the robot asset to each environment, sets DOF properties and calls callbacks to process rigid shape, rigid body and DOF properties.
        """
        from isaacsim.core.cloner import Cloner
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
        from isaaclab.actuators import DCMotorCfg
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaaclab.sensors import ContactSensorCfg, ContactSensor
        
        self.cloner = Cloner(self.stage)
        source_env_path = "/World/envs/env_0"
        prim_paths = self.cloner.generate_paths("/World/envs/env", self.num_envs)
        add_reference_to_stage(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
            prim_path=f"{source_env_path}/robot",
        )
        self.cloner.clone(source_prim_path=source_env_path,
                          prim_paths=prim_paths,
                          replicate_physics=False,
                          copy_from_source=True,
                          enable_env_ids=True)
        
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, 
                    fix_root_link=self.cfg.asset.fix_base_link,
                    solver_position_iteration_count=4, 
                    solver_velocity_iteration_count=0)
        
        rigid_props = sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=10.0,
                                                       angular_damping=0.01,
                                                       max_linear_velocity=1000.0,
                                                       max_angular_velocity=1000.0)

        print(f"Loading robot from {ISAAC_NUCLEUS_DIR}")
        usd_asset_file = f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd"
        usd_cfg = sim_utils.UsdFileCfg(usd_path=usd_asset_file,
                                       articulation_props=articulation_props, 
                                       visual_material=None, 
                                       rigid_props=rigid_props,
                                       activate_contact_sensors=True)
        rot_sim = self.cfg.init_state.rot
        # convert xyzw to wxyz
        rot_sim = [rot_sim[3], rot_sim[0], rot_sim[1], rot_sim[2]]
        print(f"default_joint_angles: {self.cfg.init_state.default_joint_angles}")
        init_state = ArticulationCfg.InitialStateCfg(pos=self.cfg.init_state.pos, 
                                                     rot=rot_sim,
                                                     joint_pos=self.cfg.init_state.default_joint_angles)
        
        actuator_cfg = DCMotorCfg(joint_names_expr=[".*"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        )
        
        articulation_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/robot",
            spawn=usd_cfg,
            collision_group=0,
            init_state=init_state,
            actuator_value_resolution_debug_print=False,
            actuators={"legs": actuator_cfg}
        )
        
        self.robot = Articulation(articulation_cfg)
        
        # Add contact sensors to the feet
        contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/robot/.*", # track all links of the robot, but only the ones specified in cfg will be used for termination and penalty
            update_period=0.0,                      # update every control step
            history_length=2,                       # keep contact history of last 2 steps
            debug_vis=not self.headless,            # visualize contact points if not headless
        )
        
        self.contact_sensors = ContactSensor(contact_sensor_cfg)
        
        self.sim.reset()
        
        # print info after reset the simulation, to make sure the sensors are initialized
        # links in contact sensors have different order from the body names in the robot articulation, 
        # so we need to print them to make sure we get the correct indices for termination and penalty
        print(f"Created contact sensors: {self.contact_sensors}")
        
        self._get_env_origins()
        
        self.dof_names = self.robot.joint_names
        print(f"DOF names: {self.dof_names}")
        self.num_dof = len(self.dof_names)
        self.num_bodies = len(self.robot.body_names)
        self._init_domain_params()
        
        def find_link_contact_indices(names: list[str]) -> list[int]:
            """find link indices in bodies of contact sensors based on link names specified in the config for termination and penalty.

            Args:
                names (list[str]): List of link names to find indices for

            Returns:
                list[int]: List of indices corresponding to the given link names
            """
            link_indices = list()
            for link in self.contact_sensors.body_names:
                flag = False
                for name in names:
                    if name in link:
                        flag = True
                if flag:
                    link_indices.append(self.contact_sensors.body_names.index(link))
            return link_indices
        
        def find_link_indices(names: list[str]) -> list[int]:
            """find link indices in bodies of the robot based on link names specified in the config for feet

            Args:
                names (list[str]): List of link names to find indices for
            Returns:
                list[int]: List of indices corresponding to the given link names
            """
            link_indices = list()
            for link in self.robot.body_names:
                flag = False
                for name in names:
                    if name in link:
                        flag = True
                if flag:
                    link_indices.append(self.robot.body_names.index(link))
            return link_indices

        self._termination_contact_indices = find_link_contact_indices(
            self.cfg.asset.terminate_after_contacts_on)
        print(f"All link names: {self.robot.body_names}")
        print(f"Termination contact link indices: {self._termination_contact_indices}")
        self._penalized_contact_indices = find_link_contact_indices(
            self.cfg.asset.penalize_contacts_on)
        print(f"Penalized contact link indices: {self._penalized_contact_indices}")
        self._feet_names = [
            link for link in self.robot.body_names if self.cfg.asset.foot_name in link
        ]
        # the order of bodies in contact sensors is different from the order of bodies in the robot articulation, so we need to find indices separately
        self._feet_contact_indices = find_link_contact_indices(self._feet_names)
        self._feet_indices = find_link_indices(self._feet_names)
        print(f"feet names: {self._feet_names}")
        assert len(self._feet_indices) > 0
        # get base link index in the robot articulation
        self._base_link_index = self.robot.body_names.index(self.cfg.asset.base_link_name)
        
        if self.cfg.asset.obtain_link_contact_states:
            self.contact_state_link_indices = find_link_indices(
                self.cfg.asset.contact_state_link_names
            )
        
        self.dof_pos_limits = self.robot.data.joint_pos_limits[0].to(self.device)
        print(f"DOF position limits: {self.dof_pos_limits}")
        self.dof_vel_limits = self.robot.data.joint_vel_limits[0].to(self.device)
        self.torque_limits = self.robot.data.joint_effort_limits[0].to(self.device)
        for i in range(self.dof_pos_limits.shape[0]):
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = (
                m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            )
            self.dof_pos_limits[i, 1] = (
                m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            )
        
        # randomize friction
        if self.cfg.domain_rand.randomize_friction:
            self._randomize_friction(torch.arange(self.num_envs))
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(np.arange(self.num_envs))
        # # randomize COM displacement
        # if self.cfg.domain_rand.randomize_com_displacement:
        #     self._randomize_com_displacement(np.arange(self.num_envs))
        # # randomize joint armature
        # if self.cfg.domain_rand.randomize_joint_armature:
        #     self._randomize_joint_armature(np.arange(self.num_envs))
        # # randomize joint friction
        # if self.cfg.domain_rand.randomize_joint_friction:
        #     self._randomize_joint_friction(np.arange(self.num_envs))
        # # randomize joint damping
        # if self.cfg.domain_rand.randomize_joint_damping:
        #     self._randomize_joint_damping(np.arange(self.num_envs))
        # # randomize pd gain
        # if self.cfg.domain_rand.randomize_pd_gain:
        #     self._randomize_pd_gain(np.arange(self.num_envs))
    
    def _init_buffers(self):
        self.base_init_pos = torch.tensor(
            self.cfg.init_state.pos, device=self.device
        )
        self.base_init_quat = torch.tensor(
            self.cfg.init_state.rot, device=self.device
        )
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float)
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float)
        self.last_base_lin_vel = torch.zeros_like(self.base_lin_vel)
        self.last_base_ang_vel = torch.zeros_like(self.base_ang_vel)
        self.projected_gravity = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float).repeat(
            self.num_envs, 1
        )
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_pos = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float)
        self.dof_vel = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float)
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=torch.float)
        self.base_quat_sim = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=torch.float) # quaternion in isaacsim definition, wxyz
        self.base_euler = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float)
        self.link_contact_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float
        )
        self.feet_pos = torch.zeros(
            (self.num_envs, len(self.feet_indices), 3), device=self.device, dtype=torch.float
        )
        self.feet_vel = torch.zeros(
            (self.num_envs, len(self.feet_indices), 3), device=self.device, dtype=torch.float
        )
        self.last_feet_vel = torch.zeros_like(self.feet_vel)
        # depth images
        if self.cfg.sensor.add_depth:
            self.depth_images = torch.zeros(
                (self.num_envs, 
                 self.cfg.sensor.depth_camera_config.num_history,
                 self.cfg.sensor.depth_camera_config.resolution[1], 
                 self.cfg.sensor.depth_camera_config.resolution[0]), 
                device=self.device, 
                dtype=torch.float
            )
        
        # Terrain information around feet
        if self.cfg.terrain.obtain_terrain_info_around_feet:
            self.normal_vector_around_feet = torch.zeros(
                self.num_envs, len(self.feet_indices) * 3, dtype=torch.float, device=self.device, requires_grad=False)
            self.height_around_feet = torch.zeros(
                self.num_envs, len(self.feet_indices), 9, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.asset.obtain_link_contact_states:
            self.link_contact_states = torch.zeros(
                self.num_envs, len(self.contact_state_link_indices), dtype=torch.float, device=self.device, requires_grad=False)
        
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def step(self, actions):
        # actions = torch.zeros_like(actions).to(self.device)
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(actions)
            self.robot.set_joint_effort_target(
                self.torques
            )
            self.robot.write_data_to_sim()
            self.sim.step(render=False)
            self.robot.update(self.sim_params["dt"])
            self.contact_sensors.update(self.sim_params["dt"])
            self.dof_pos[:] = self.robot.data.joint_pos[:]
            self.dof_vel[:] = self.robot.data.joint_vel[:]
        self.sim.render()

    def post_physics_step(self):
        self.base_pos[:] = self.robot.data.root_link_state_w[:, :3]
        self._check_base_pos_out_of_bound()       # check if the pos of the robot is out of terrain bounds
        self.base_pos[:] = self.robot.data.root_link_state_w[:, :3]
        self.base_quat_sim[:] = self.robot.data.root_link_state_w[:, 3:7]
        # convert wxyz to xyzw
        self.base_quat[:, -1] = self.base_quat_sim[:, 0]
        self.base_quat[:, :3] = self.base_quat_sim[:, 1:]
        self.base_euler[:] = get_euler_xyz(self.base_quat)
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.robot.data.root_link_state_w[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.robot.data.root_link_state_w[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.global_gravity)
        self.dof_pos[:] = self.robot.data.joint_pos[:]
        self.dof_vel[:] = self.robot.data.joint_vel[:]
        self.feet_pos[:] = self.robot.data.body_link_pose_w[:, self.feet_indices, :3]
        self.feet_vel[:] = self.robot.data.body_link_vel_w[:, self.feet_indices, :3]
        self.link_contact_forces[:] = self.contact_sensors.data.net_forces_w[:]
        # Link contact state
        if self.cfg.asset.obtain_link_contact_states:
            self.link_contact_states = 1. * (torch.norm(
                self.link_contact_forces[:, self.contact_state_link_indices, :], dim=-1) > 1.)
        
    def _update_surrounding_heights(self):
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError(
                "Can't measure height with terrain mesh type 'none'")

        points = quat_apply_yaw(self.base_quat.repeat(
                1, self.num_height_points), self.height_points) + (self.base_pos[:, :3]).unsqueeze(1)

        # When acquiring heights, the points need to add border_size
        # because in the height_samples, the origin of the terrain is at (border_size, border_size)
        points += self.cfg.terrain.border_size
        points = (points/self.cfg.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        self.measured_heights = heights.view(self.num_envs, -1) * self.cfg.terrain.vertical_scale

    def push_robots(self):
        max_push_vel_xy = self.cfg.domain_rand.max_push_vel_xy
        cur_root_vel = self.robot.data.root_link_vel_w[:, :3]
        push_vel = torch_rand_float(-max_push_vel_xy,
                                    max_push_vel_xy, (self.num_envs, 2), device=self.device)
        self._rand_push_vels[:, :2] = push_vel.detach().clone()
        cur_root_vel[:, :2] += push_vel
        root_vel = torch.cat([cur_root_vel, self.robot.data.root_link_vel_w[:, 3:6]], dim=-1)
        self.robot.write_root_link_velocity_to_sim(root_vel, None)
    
    def reset_idx(self, env_ids):
        # domain randomization
        if self.cfg.domain_rand.randomize_friction:
            self._randomize_friction(env_ids)
        if self.cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(env_ids)
        # if self.cfg.domain_rand.randomize_com_displacement:
        #     self._randomize_com_displacement(env_ids)
        # if self.cfg.domain_rand.randomize_joint_armature:
        #     self._randomize_joint_armature(env_ids)
        # if self.cfg.domain_rand.randomize_joint_friction:
        #     self._randomize_joint_friction(env_ids)
        # if self.cfg.domain_rand.randomize_joint_damping:
        #     self._randomize_joint_damping(env_ids)
        # if self.cfg.domain_rand.randomize_pd_gain:
        #     self._randomize_pd_gain(env_ids)
        
        self.robot.reset(env_ids)
        self.contact_sensors.reset(env_ids)
        
        self.last_dof_vel[env_ids] = 0.
        self.last_feet_vel[env_ids] = 0.
        self.last_base_lin_vel[env_ids] = 0.
        self.last_base_ang_vel[env_ids] = 0.
    
    def reset_dofs(self, env_ids, dof_pos, dof_vel):
        self.dof_pos[env_ids] = dof_pos[:]
        self.dof_vel[env_ids] = dof_vel[:]
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, None, env_ids)
        
    def reset_root_states(self, env_ids, base_pos, base_quat, base_lin_vel, base_ang_vel):
        # base pos
        self.base_pos[env_ids, :] = base_pos[:]
        # base quat
        self.base_quat[env_ids, :] = base_quat[:]
        self.base_quat_sim[env_ids, 0] = base_quat[:, 3]  # w
        self.base_quat_sim[env_ids, 1:] = base_quat[:, :3]  # xyz
        # updated projected gravity
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.global_gravity)
        # base lin and ang vel
        self.base_lin_vel[env_ids] = base_lin_vel[:]
        self.base_ang_vel[env_ids] = base_ang_vel[:]
        # concatenate pos, quat, lin vel and ang vel to write to sim
        root_states = torch.cat([base_pos, 
                                 base_quat, 
                                 base_lin_vel, 
                                 base_ang_vel], dim=-1)
        self.robot.write_root_link_state_to_sim(root_states, env_ids)
    
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(
                0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (
                self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(
                self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels,
                                                       self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(
                num_rows), torch.arange(num_cols), indexing='ij')
            # plane has limited size, we need to specify spacing base on num_envs, to make sure all robots are within the plane
            # restrict envs to a square of [plane_length/2, plane_length/2]
            spacing = self.cfg.env.env_spacing
            if num_rows * self.cfg.env.env_spacing > self.cfg.terrain.plane_length / 2 or \
                    num_cols * self.cfg.env.env_spacing > self.cfg.terrain.plane_length / 2:
                spacing = min((self.cfg.terrain.plane_length / 2) / (num_rows-1),
                              (self.cfg.terrain.plane_length / 2) / (num_cols-1))
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
            self.env_origins[:, 0] -= self.cfg.terrain.plane_length / 4
            self.env_origins[:, 1] -= self.cfg.terrain.plane_length / 4
    
    def _init_domain_params(self):
        self._friction_values = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        # save original base mass for the first time
        self._default_base_mass = torch.ones(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False) * \
                self.robot.data.default_mass[:, self._base_link_index].unsqueeze(1)
        self._added_base_mass = torch.ones(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self._rand_push_vels = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self._base_com_bias = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self._joint_armature = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self._joint_friction = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self._joint_damping = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self._kp_scale = torch.ones(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._kd_scale = torch.ones(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
    
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self._kp_scale * self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self._kd_scale * self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self._kp_scale * self.p_gains*(actions_scaled - self.dof_vel) - self._kd_scale * self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def _check_base_pos_out_of_bound(self):
        """ Check if the base position is out of the terrain bounds
        """
        x_out_of_bound = (self.base_pos[:, 0] >= self.terrain_x_range[1]) | (
            self.base_pos[:, 0] <= self.terrain_x_range[0])
        y_out_of_bound = (self.base_pos[:, 1] >= self.terrain_y_range[1]) | (
            self.base_pos[:, 1] <= self.terrain_y_range[0])
        out_of_bound_buf = x_out_of_bound | y_out_of_bound
        env_ids = out_of_bound_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        else:
            # reset base position to initial position
            self.base_pos[env_ids] = self.base_init_pos
            self.base_pos[env_ids] += self.env_origins[env_ids]
            root_pose = torch.cat([self.base_pos[env_ids], 
                                   self.base_quat_sim[env_ids]], dim=-1)
            self.robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
    
    def _randomize_friction(self, env_ids):
        if len(env_ids) == 0:
            return
        min_friction, max_friction = self.cfg.domain_rand.friction_range

        # refer to https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.set_material_properties
        articulation_view = self.robot.root_physx_view
        # All shapes in the same env have the same static friction and dynamic friction values
        friction_ratios = torch.rand((len(env_ids), 1, 1), 
                            dtype=torch.float,
                            device=self.device).repeat(1, articulation_view.max_shapes, 2) \
                            * (max_friction - min_friction) + min_friction
        # save values to domain randomization params
        self._friction_values[env_ids] = friction_ratios[:,0,0].unsqueeze(1).detach().clone()
        
        raw_material_props = articulation_view.get_material_properties()
        target_material_props = torch.cat((
            friction_ratios.to("cpu"),
            raw_material_props[:, :, 2].unsqueeze(-1), # use original restitution
        ), dim=-1)
        # tensors passed to set_material_properties must be on CPU
        articulation_view.set_material_properties(
            target_material_props, env_ids.to("cpu")
        )
    
    def _randomize_base_mass(self, env_ids):
        if len(env_ids) == 0:
            return
        min_mass, max_mass = self.cfg.domain_rand.added_mass_range
        base_link_id = 1
        added_mass = gs.rand((len(env_ids), 1), dtype=float) * \
            (max_mass - min_mass) + min_mass
        self._added_base_mass[env_ids] = added_mass[:].detach().clone()
        self.robot.set_mass_shift(added_mass, [base_link_id, ], env_ids)
        