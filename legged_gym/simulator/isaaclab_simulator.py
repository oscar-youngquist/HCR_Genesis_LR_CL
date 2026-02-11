from legged_gym import *
from legged_gym.simulator.simulator import Simulator
from PIL import Image as im
import cv2 as cv
import torch
import numpy as np
import os
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math_utils import *
if SIMULATOR == "isaaclab":
    from isaaclab.app import AppLauncher
    import carb
    GROUND_PATH = "/World/ground"

""" ********** IsaacLab Simulator ********** """
class IsaacLabSimulator(Simulator):
    """Simulator class for IsaacLab"""
    def __init__(self, cfg, sim_params: dict, device, headless):
        self._sim_params = sim_params
        super().__init__(cfg, sim_params, device, headless)
    
    def _parse_cfg(self):
        self._debug = self._cfg.env.debug
        if self._cfg.sensor.add_depth:
            self._frame_count = 0
            
    def _create_sim(self):
        self._app_launcher = AppLauncher({"headless": self._headless, "device": self._device})
        
        import isaaclab.sim as sim_utils
        from isaacsim.core.utils.stage import get_current_stage
        # create simulation context
        sim_cfg = sim_utils.SimulationCfg(device=self._device, dt=self._sim_params["dt"],
                                          render_interval=self._cfg.control.decimation)

        sim_cfg.physx.bounce_threshold_velocity = 0.2
        sim_cfg.physx.max_position_iteration_count = 4
        sim_cfg.physx.max_velocity_iteration_count = 0
        sim_cfg.physx.gpu_max_rigid_contact_count = 8 * 1024 * 1024
        sim_cfg.physics_material.static_friction = 1.0
        sim_cfg.physics_material.dynamic_friction = 1.0
        
        self._sim = sim_utils.SimulationContext(sim_cfg)
        self._stage = get_current_stage()
        
        # disable delays during rendering
        carb_settings = carb.settings.get_settings()
        carb_settings.set_bool("/app/runLoops/main/rateLimitEnabled", False)

        # add terrain
        mesh_type = self._cfg.terrain.mesh_type
        from isaaclab.terrains import TerrainImporterCfg, TerrainImporter
        import isaaclab.sim as sim_utils
        if mesh_type == "plane":
            terrain_cfg = TerrainImporterCfg(
                num_envs=self._num_envs,
                env_spacing=self._cfg.env.env_spacing,
                prim_path=GROUND_PATH,
                terrain_type="plane",
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=self._cfg.terrain.static_friction,
                    dynamic_friction=self._cfg.terrain.dynamic_friction,
                    restitution=self._cfg.terrain.restitution
                )
            )
            self._terrain = TerrainImporter(terrain_cfg)
        elif mesh_type == "heightfield":
            raise NotImplementedError("Heightfield terrain not implemented for IsaacLabSimulator yet")
        elif mesh_type == "trimesh":
            raise NotImplementedError("Trimesh terrain not implemented for IsaacLabSimulator yet")
        else:
            raise NameError(f"Unknown terrain mesh type: {mesh_type}")
        
        # specify the boundary of the heightfield
        self._terrain_x_range = torch.zeros(2, device=self._device)
        self._terrain_y_range = torch.zeros(2, device=self._device)
        if self._cfg.terrain.mesh_type == 'heightfield':
            # give a small margin(1.0m)
            self._terrain_x_range[0] = -self._cfg.terrain.border_size + 1.0
            self._terrain_x_range[1] = self._cfg.terrain.border_size + \
                self._cfg.terrain.num_rows * self._cfg.terrain.terrain_length - 1.0
            self._terrain_y_range[0] = -self._cfg.terrain.border_size + 1.0
            self._terrain_y_range[1] = self._cfg.terrain.border_size + \
                self._cfg.terrain.num_cols * self._cfg.terrain.terrain_width - 1.0
        elif self._cfg.terrain.mesh_type == 'plane':  # the plane used has limited size,
            # and the origin of the world is at the center of the plane
            self._terrain_x_range[0] = -self._cfg.terrain.plane_length/2+1
            self._terrain_x_range[1] = self._cfg.terrain.plane_length/2-1
            # the plane is a square
            self._terrain_y_range[0] = -self._cfg.terrain.plane_length/2+1
            self._terrain_y_range[1] = self._cfg.terrain.plane_length/2-1
        
    def _create_envs(self):
        """ Creates environments, adds the robot asset to each environment, sets DOF properties and calls callbacks to process rigid shape, rigid body and DOF properties.
        """
        from isaacsim.core.cloner import Cloner
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaaclab.sensors import ContactSensorCfg, ContactSensor
        
        self._cloner = Cloner(self._stage)
        source_env_path = "/World/envs/env_0"
        prim_paths = self._cloner.generate_paths("/World/envs/env", self._num_envs)
        add_reference_to_stage(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
            prim_path=f"{source_env_path}/robot",
        )
        self._cloner.clone(source_prim_path=source_env_path,
                          prim_paths=prim_paths,
                          replicate_physics=False,
                          copy_from_source=True,
                          enable_env_ids=True)
        
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, 
                    fix_root_link=self._cfg.asset.fix_base_link,
                    solver_position_iteration_count=4, 
                    solver_velocity_iteration_count=0)
        
        rigid_props = sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0,
                                                       angular_damping=0.0,
                                                       linear_damping=0.0,
                                                       disable_gravity=False,
                                                       retain_accelerations=False,
                                                       max_linear_velocity=1000.0,
                                                       max_angular_velocity=1000.0)

        if self._cfg.asset.name == "go2":
            usd_asset_file = f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd"
        else:
            raise NameError(f"Unknown robot name: {self._cfg.asset.name}")
        usd_cfg = sim_utils.UsdFileCfg(usd_path=usd_asset_file,
                                       articulation_props=articulation_props, 
                                       visual_material=None, 
                                       rigid_props=rigid_props,
                                       activate_contact_sensors=True)
        # convert xyzw to wxyz
        rot_sim = [self._cfg.init_state.rot[3], 
                   self._cfg.init_state.rot[0], 
                   self._cfg.init_state.rot[1], 
                   self._cfg.init_state.rot[2]]
        init_state = ArticulationCfg.InitialStateCfg(pos=self._cfg.init_state.pos, 
                                                     rot=rot_sim,
                                                     joint_pos=self._cfg.init_state.default_joint_angles)
        
        if self._cfg.asset.name == "go2":
            from resources.robots.go2.go2_lab_cfg import GO2_ACTUATOR_CFG
            actuator_cfg = GO2_ACTUATOR_CFG
        else:
            raise NameError(f"Unknown robot name: {self._cfg.asset.name}")
        
        articulation_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/robot",
            spawn=usd_cfg,  # loading urdf needs to be converted to usd, which is time-consuming
            collision_group=0,
            init_state=init_state,
            actuator_value_resolution_debug_print=False,
            actuators=actuator_cfg
        )
        
        self._robot = Articulation(articulation_cfg)
        
        # Add contact sensors to the feet
        contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/robot/.*", # track all links of the robot, but only the ones specified in cfg will be used for termination and penalty
            update_period=0.0,                      # update every control step
            history_length=2,                       # keep contact history of last 2 steps
            debug_vis=not self._headless,            # visualize contact points if not headless
        )
        
        self._contact_sensors = ContactSensor(contact_sensor_cfg)
        
        self._sim.reset()
        
        # print info after reset the simulation, to make sure the sensors are initialized
        # links in contact sensors have different order from the body names in the robot articulation, 
        # so we need to print them to make sure we get the correct indices for termination and penalty
        print(f"Created contact sensors: {self._contact_sensors}")
        
        self._get_env_origins()
        
        self.dof_names = self._robot.joint_names
        print(f"DOF names: {self.dof_names}")
        self.num_dof = len(self.dof_names)
        self.num_bodies = len(self._robot.body_names)
        
        def find_link_contact_indices(names: list[str]) -> list[int]:
            """find link indices in bodies of contact sensors based on link names specified in the config for termination and penalty.

            Args:
                names (list[str]): List of link names to find indices for

            Returns:
                list[int]: List of indices corresponding to the given link names
            """
            link_indices = list()
            for link in self._contact_sensors.body_names:
                flag = False
                for name in names:
                    if name in link:
                        flag = True
                if flag:
                    link_indices.append(self._contact_sensors.body_names.index(link))
            return link_indices
        
        def find_link_indices(names: list[str]) -> list[int]:
            """find link indices in bodies of the robot based on link names specified in the config for feet

            Args:
                names (list[str]): List of link names to find indices for
            Returns:
                list[int]: List of indices corresponding to the given link names
            """
            link_indices = list()
            for link in self._robot.body_names:
                flag = False
                for name in names:
                    if name in link:
                        flag = True
                if flag:
                    link_indices.append(self._robot.body_names.index(link))
            return link_indices

        self._termination_contact_indices = find_link_contact_indices(
            self._cfg.asset.terminate_after_contacts_on)
        print(f"All link names: {self._robot.body_names}")
        print(f"Termination contact link indices: {self._termination_contact_indices}")
        self._penalized_contact_indices = find_link_contact_indices(
            self._cfg.asset.penalize_contacts_on)
        print(f"Penalized contact link indices: {self._penalized_contact_indices}")
        self._feet_names = [
            link for link in self._robot.body_names if self._cfg.asset.foot_name in link
        ]
        # the order of bodies in contact sensors is different from the order of bodies in the robot articulation, so we need to find indices separately
        self._feet_contact_indices = find_link_contact_indices(self._feet_names)
        self._feet_indices = find_link_indices(self._feet_names)
        print(f"feet names: {self._feet_names}")
        assert len(self._feet_indices) > 0
        # get base link index in the robot articulation
        self._base_link_index = self._robot.body_names.index(self._cfg.asset.base_link_name)
        
        if self._cfg.asset.obtain_link_contact_states:
            self._contact_state_link_indices = find_link_indices(
                self._cfg.asset.contact_state_link_names
            )
        
        self._dof_pos_limits = self._robot.data.joint_pos_limits[0].to(self._device)
        print(f"DOF position limits: {self._dof_pos_limits}")
        self._dof_vel_limits = self._robot.data.joint_vel_limits[0].to(self._device)
        self._torque_limits = self._robot.data.joint_effort_limits[0].to(self._device)
        for i in range(self._dof_pos_limits.shape[0]):
            # soft limits
            m = (self._dof_pos_limits[i, 0] + self._dof_pos_limits[i, 1]) / 2
            r = self._dof_pos_limits[i, 1] - self._dof_pos_limits[i, 0]
            self._dof_pos_limits[i, 0] = (
                m - 0.5 * r * self._cfg.rewards.soft_dof_pos_limit
            )
            self._dof_pos_limits[i, 1] = (
                m + 0.5 * r * self._cfg.rewards.soft_dof_pos_limit
            )
        
        self._init_domain_params()
        # randomize friction
        if self._cfg.domain_rand.randomize_friction:
            self._randomize_friction(torch.arange(self._num_envs))
        # randomize base mass
        if self._cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(torch.arange(self._num_envs))
        # randomize COM displacement
        if self._cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(torch.arange(self._num_envs))
        # randomize joint armature
        if self._cfg.domain_rand.randomize_joint_armature:
            self._randomize_joint_armature(torch.arange(self._num_envs))
        # randomize joint friction
        if self._cfg.domain_rand.randomize_joint_friction:
            self._randomize_joint_friction(torch.arange(self._num_envs))
        # randomize joint damping
        if self._cfg.domain_rand.randomize_joint_damping:
            self._randomize_joint_damping(torch.arange(self._num_envs))
        # randomize pd gain
        if self._cfg.domain_rand.randomize_pd_gain:
            self._randomize_pd_gain(torch.arange(self._num_envs))
    
    def _init_buffers(self):
        self._base_init_pos = torch.tensor(
            self._cfg.init_state.pos, device=self._device
        )
        self._base_init_quat = torch.tensor(
            self._cfg.init_state.rot, device=self._device
        )
        self._base_lin_vel = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float)
        self._base_ang_vel = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float)
        self._last_base_lin_vel = torch.zeros_like(self._base_lin_vel)
        self._last_base_ang_vel = torch.zeros_like(self._base_ang_vel)
        self._projected_gravity = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float)
        self._global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self._device, dtype=torch.float).repeat(
            self._num_envs, 1)
        self._torques = torch.zeros(self._num_envs, self._num_actions, dtype=torch.float, device=self._device, requires_grad=False)
        self._p_gains = torch.zeros(self._num_actions, dtype=torch.float, device=self._device, requires_grad=False)
        self._d_gains = torch.zeros(self._num_actions, dtype=torch.float, device=self._device, requires_grad=False)
        self._dof_pos = torch.zeros(self._num_envs, self._num_actions, device=self._device, dtype=torch.float)
        self._dof_vel = torch.zeros(self._num_envs, self._num_actions, device=self._device, dtype=torch.float)
        self._last_dof_vel = torch.zeros_like(self._dof_vel)
        self._base_pos = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float)
        self._base_quat = torch.zeros(
            (self._num_envs, 4), device=self._device, dtype=torch.float)
        self._base_quat_sim = torch.zeros(
            (self._num_envs, 4), device=self._device, dtype=torch.float) # quaternion in isaacsim definition, wxyz
        self._base_euler = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float)
        self._link_contact_forces = torch.zeros(
            (self._num_envs, self.num_bodies, 3), device=self._device, dtype=torch.float
        )
        self._feet_pos = torch.zeros(
            (self._num_envs, len(self.feet_indices), 3), device=self._device, dtype=torch.float
        )
        self._feet_vel = torch.zeros(
            (self._num_envs, len(self.feet_indices), 3), device=self._device, dtype=torch.float
        )
        self._last_feet_vel = torch.zeros_like(self._feet_vel)
        # depth images
        if self._cfg.sensor.add_depth:
            self._depth_images = torch.zeros(
                (self._num_envs, 
                 self._cfg.sensor.depth_camera_config.num_history,
                 self._cfg.sensor.depth_camera_config.resolution[1], 
                 self._cfg.sensor.depth_camera_config.resolution[0]), 
                device=self._device, 
                dtype=torch.float
            )
        
        # Terrain information around feet
        if self._cfg.terrain.obtain_terrain_info_around_feet:
            self._normal_vector_around_feet = torch.zeros(
                self._num_envs, len(self.feet_indices) * 3, dtype=torch.float, device=self._device, requires_grad=False)
            self._height_around_feet = torch.zeros(
                self._num_envs, len(self.feet_indices), 9, dtype=torch.float, device=self._device, requires_grad=False)
        
        if self._cfg.asset.obtain_link_contact_states:
            self._link_contact_states = torch.zeros(
                self._num_envs, len(self._contact_state_link_indices), dtype=torch.float, device=self._device, requires_grad=False)
        
        # joint positions offsets and PD gains
        self._default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self._device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            self._default_dof_pos[i] = self._cfg.init_state.default_joint_angles[name]
            found = False
            for dof_name in self._cfg.control.stiffness.keys():
                if dof_name in name:
                    self._p_gains[i] = self._cfg.control.stiffness[dof_name]
                    self._d_gains[i] = self._cfg.control.damping[dof_name]
                    found = True
            if not found:
                self._p_gains[i] = 0.
                self._d_gains[i] = 0.
                if self._cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self._default_dof_pos = self._default_dof_pos.unsqueeze(0)
        
        self._init_height_points()
        self._measured_heights = torch.zeros(self._num_envs, self._num_height_points, device=self._device, requires_grad=False)
        
    def _init_height_points(self):
        y = torch.tensor(self._cfg.terrain.measured_points_y,
                         device=self._device, requires_grad=False)
        x = torch.tensor(self._cfg.terrain.measured_points_x,
                         device=self._device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        self._num_height_points = grid_x.numel()
        self._height_points = torch.zeros(self._num_envs, self._num_height_points,
                             3, device=self._device, requires_grad=False)
        self._height_points[:, :, 0] = grid_x.flatten()
        self._height_points[:, :, 1] = grid_y.flatten()

    def step(self, actions):
        # actions = torch.zeros_like(actions).to(self._device)
        self._last_dof_vel[:] = self._dof_vel[:]
        for _ in range(self._cfg.control.decimation):
            self._torques = self._compute_torques(actions)
            self._robot.set_joint_effort_target(
                self._torques
            )
            self._robot.write_data_to_sim()
            self._sim.step(render=False)
            self._robot.update(self._sim_params["dt"])
            self._contact_sensors.update(self._sim_params["dt"])
            self._dof_pos[:] = self._robot.data.joint_pos[:]
            self._dof_vel[:] = self._robot.data.joint_vel[:]
        self._sim.render()

    def post_physics_step(self):
        self._base_pos[:] = self._robot.data.root_link_state_w[:, :3]
        self._check_base_pos_out_of_bound()       # check if the pos of the robot is out of terrain bounds
        self._base_pos[:] = self._robot.data.root_link_state_w[:, :3]
        self._base_quat_sim[:] = self._robot.data.root_link_state_w[:, 3:7]
        # convert wxyz to xyzw
        self._base_quat[:, -1] = self._base_quat_sim[:, 0]
        self._base_quat[:, :3] = self._base_quat_sim[:, 1:]
        self._base_euler[:] = get_euler_xyz(self._base_quat)
        self._base_lin_vel[:] = quat_rotate_inverse(self._base_quat, self._robot.data.root_link_state_w[:, 7:10])
        self._base_ang_vel[:] = quat_rotate_inverse(self._base_quat, self._robot.data.root_link_state_w[:, 10:13])
        self._projected_gravity[:] = quat_rotate_inverse(self._base_quat, self._global_gravity)
        self._dof_pos[:] = self._robot.data.joint_pos[:]
        self._dof_vel[:] = self._robot.data.joint_vel[:]
        self._feet_pos[:] = self._robot.data.body_link_pose_w[:, self.feet_indices, :3]
        self._last_feet_vel[:] = self._feet_vel[:]
        self._feet_vel[:] = self._robot.data.body_link_vel_w[:, self.feet_indices, :3]
        self._link_contact_forces[:] = self._contact_sensors.data.net_forces_w[:]
        # Link contact state
        if self._cfg.asset.obtain_link_contact_states:
            self._link_contact_states = 1. * (torch.norm(
                self._link_contact_forces[:, self._contact_state_link_indices, :], dim=-1) > 1.)
        # update terrain heights info
        if self._cfg.terrain.measure_heights:
            self._update_surrounding_heights()
            if self._cfg.terrain.obtain_terrain_info_around_feet:
                self._calc_terrain_info_around_feet()
        
    def _update_surrounding_heights(self):
        if self._cfg.terrain.mesh_type == 'plane':
            self._measured_heights = torch.zeros(self._num_envs, self._num_height_points, device=self._device, requires_grad=False)
        elif self._cfg.terrain.mesh_type == 'none':
            raise NameError(
                "Can't measure height with terrain mesh type 'none'")

        points = quat_apply_yaw(self._base_quat.repeat(
                1, self._num_height_points), self._height_points) + (self._base_pos[:, :3]).unsqueeze(1)

        # When acquiring heights, the points need to add border_size
        # because in the height_samples, the origin of the terrain is at (border_size, border_size)
        points += self._cfg.terrain.border_size
        points = (points/self._cfg.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self._height_samples.shape[0]-2)
        py = torch.clip(py, 0, self._height_samples.shape[1]-2)

        heights1 = self._height_samples[px, py]
        heights2 = self._height_samples[px+1, py]
        heights3 = self._height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        self._measured_heights = heights.view(self._num_envs, -1) * self._cfg.terrain.vertical_scale

    def push_robots(self):
        max_push_vel_xy = self._cfg.domain_rand.max_push_vel_xy
        cur_root_vel = self._robot.data.root_link_vel_w[:, :3]
        push_vel = torch_rand_float(-max_push_vel_xy,
                                    max_push_vel_xy, (self._num_envs, 2), device=self._device)
        self._rand_push_vels[:, :2] = push_vel.detach().clone()
        cur_root_vel[:, :2] += push_vel
        root_vel = torch.cat([cur_root_vel, self._robot.data.root_link_vel_w[:, 3:6]], dim=-1)
        self._robot.write_root_link_velocity_to_sim(root_vel, None)
    
    def reset_idx(self, env_ids):
        # domain randomization
        if self._cfg.domain_rand.randomize_friction:
            self._randomize_friction(env_ids)
        if self._cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(env_ids)
        if self._cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(env_ids)
        if self._cfg.domain_rand.randomize_joint_armature:
            self._randomize_joint_armature(env_ids)
        if self._cfg.domain_rand.randomize_joint_friction:
            self._randomize_joint_friction(env_ids)
        if self._cfg.domain_rand.randomize_joint_damping:
            self._randomize_joint_damping(env_ids)
        if self._cfg.domain_rand.randomize_pd_gain:
            self._randomize_pd_gain(env_ids)
        
        self._robot.reset(env_ids)
        self._contact_sensors.reset(env_ids)
        
        self._last_dof_vel[env_ids] = 0.
        self._last_feet_vel[env_ids] = 0.
        self._last_base_lin_vel[env_ids] = 0.
        self._last_base_ang_vel[env_ids] = 0.
    
    def reset_dofs(self, env_ids, dof_pos, dof_vel):
        self._dof_pos[env_ids] = dof_pos[:]
        self._dof_vel[env_ids] = dof_vel[:]
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, None, env_ids)
        
    def reset_root_states(self, env_ids, base_pos, base_quat, base_lin_vel, base_ang_vel):
        # base pos
        self._base_pos[env_ids, :] = base_pos[:]
        # base quat
        self._base_quat[env_ids, :] = base_quat[:]
        self._base_quat_sim[env_ids, 0] = base_quat[:, 3]  # w
        self._base_quat_sim[env_ids, 1:] = base_quat[:, :3]  # xyz
        # updated projected gravity
        self._projected_gravity = quat_rotate_inverse(self._base_quat, self._global_gravity)
        # base lin and ang vel
        self._base_lin_vel[env_ids] = base_lin_vel[:]
        self._base_ang_vel[env_ids] = base_ang_vel[:]
        # concatenate pos, quat, lin vel and ang vel to write to sim
        root_states = torch.cat([base_pos, 
                                 base_quat, 
                                 base_lin_vel, 
                                 base_ang_vel], dim=-1)
        self._robot.write_root_link_state_to_sim(root_states, env_ids)
    
    def update_sensors(self):
        return super().update_sensors()
    
    def update_terrain_curriculum(self, env_ids, move_up, move_down):
        self._terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self._terrain_levels[env_ids] = torch.where(self._terrain_levels[env_ids] >=self._max_terrain_level,
                                                   torch.randint_like(
                                                       self._terrain_levels[env_ids], self._max_terrain_level),
                                                   torch.clip(self._terrain_levels[env_ids], 0))  # (the minumum level is zero)
        self._env_origins[env_ids] = self._terrain_origins[self._terrain_levels[env_ids],
            self._terrain_types[env_ids]]
    
    def draw_debug_vis(self):
        return super().draw_debug_vis()
    
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self._cfg.terrain.mesh_type in ["heightfield"]:
            self._custom_origins = True
            self._env_origins = torch.zeros(
                self._num_envs, 3, device=self._device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self._cfg.terrain.max_init_terrain_level
            if not self._cfg.terrain.curriculum:
                max_init_level = self._cfg.terrain.num_rows - 1
            self._terrain_levels = torch.randint(
                0, max_init_level+1, (self._num_envs,), device=self._device)
            self._terrain_types = torch.div(torch.arange(self._num_envs, device=self._device), (
                self._num_envs/self._cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self._max_terrain_level = self._cfg.terrain.num_rows
            self._terrain_origins = torch.from_numpy(
                self.terrain.env_origins).to(self._device).to(torch.float)
            self._env_origins[:] = self._terrain_origins[self._terrain_levels,
                                                       self._terrain_types]
        else:
            self._custom_origins = False
            self._env_origins = torch.zeros(
                self._num_envs, 3, device=self._device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self._num_envs))
            num_rows = np.ceil(self._num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(
                num_rows), torch.arange(num_cols), indexing='ij')
            # plane has limited size, we need to specify spacing base on num_envs, to make sure all robots are within the plane
            # restrict envs to a square of [plane_length/2, plane_length/2]
            spacing = self._cfg.env.env_spacing
            if num_rows * self._cfg.env.env_spacing > self._cfg.terrain.plane_length / 2 or \
                    num_cols * self._cfg.env.env_spacing > self._cfg.terrain.plane_length / 2:
                spacing = min((self._cfg.terrain.plane_length / 2) / (num_rows-1),
                              (self._cfg.terrain.plane_length / 2) / (num_cols-1))
            self._env_origins[:, 0] = spacing * xx.flatten()[:self._num_envs]
            self._env_origins[:, 1] = spacing * yy.flatten()[:self._num_envs]
            self._env_origins[:, 2] = 0.
            self._env_origins[:, 0] -= self._cfg.terrain.plane_length / 4
            self._env_origins[:, 1] -= self._cfg.terrain.plane_length / 4
    
    def _init_domain_params(self):
        self._friction_values = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        # save original base mass for the first time
        self._default_base_mass = torch.ones(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False) * \
                self._robot.data.default_mass[:, self._base_link_index].unsqueeze(1).to(self._device)
        self._added_base_mass = torch.ones(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._rand_push_vels = torch.zeros(
            self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        # save original COM position for the first time
        body_coms = self._robot.root_physx_view.get_coms().to(self._device)
        self._default_com_pos = torch.zeros(
            self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        self._default_com_pos[:] = body_coms[:, self._base_link_index, :3]
        self._base_com_bias = torch.zeros(
            self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_armature = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_friction = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_damping = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._kp_scale = torch.ones(
            self._num_envs, self.num_dof, dtype=torch.float, device=self._device, requires_grad=False)
        self._kd_scale = torch.ones(
            self._num_envs, self.num_dof, dtype=torch.float, device=self._device, requires_grad=False)
    
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
        actions_scaled = actions * self._cfg.control.action_scale
        control_type = self._cfg.control.control_type
        if control_type=="P":
            torques = self._kp_scale * self._p_gains*(actions_scaled + self._default_dof_pos - self._dof_pos) - self._kd_scale * self._d_gains*self._dof_vel
        elif control_type=="V":
            torques = self._kp_scale * self._p_gains*(actions_scaled - self._dof_vel) - self._kd_scale * self._d_gains*(self._dof_vel - self._last_dof_vel)/self._sim_params["dt"]
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def _check_base_pos_out_of_bound(self):
        """ Check if the base position is out of the terrain bounds
        """
        x_out_of_bound = (self._base_pos[:, 0] >= self._terrain_x_range[1]) | (
            self._base_pos[:, 0] <= self._terrain_x_range[0])
        y_out_of_bound = (self._base_pos[:, 1] >= self._terrain_y_range[1]) | (
            self._base_pos[:, 1] <= self._terrain_y_range[0])
        out_of_bound_buf = x_out_of_bound | y_out_of_bound
        env_ids = out_of_bound_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        else:
            # reset base position to initial position
            self._base_pos[env_ids] = self._base_init_pos
            self._base_pos[env_ids] += self.env_origins[env_ids]
            root_pose = torch.cat([self._base_pos[env_ids], 
                                   self._base_quat_sim[env_ids]], dim=-1)
            self._robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
    
    def _randomize_friction(self, env_ids):
        if len(env_ids) == 0:
            return
        min_friction, max_friction = self._cfg.domain_rand.friction_range

        # refer to https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.set_material_properties
        # All shapes in the same env have the same static friction and dynamic friction values
        friction_ratios = torch.rand((len(env_ids), 1, 1), 
                            dtype=torch.float,
                            device=self._device).repeat(1, self._robot.root_physx_view.max_shapes, 2) \
                            * (max_friction - min_friction) + min_friction
        # save values to domain randomization params
        self._friction_values[env_ids] = friction_ratios[:,0,0].unsqueeze(1).detach().clone()
        
        raw_material_props = self._robot.root_physx_view.get_material_properties().to(self._device)
        target_material_props = raw_material_props.clone()
        target_material_props[env_ids, :, 0:2] = friction_ratios[:]
        all_indices = torch.arange(self._robot.root_physx_view.count, device="cpu")
        # tensors passed to set_material_properties must be on CPU
        self._robot.root_physx_view.set_material_properties(
            target_material_props.to('cpu'), all_indices
        )
    
    def _randomize_base_mass(self, env_ids):
        if len(env_ids) == 0:
            return
        min_mass, max_mass = self._cfg.domain_rand.added_mass_range
        # refer to https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.set_masses
        added_mass = torch.rand((len(env_ids), 1), 
                                dtype=torch.float,
                                device=self._device) * \
                        (max_mass - min_mass) + min_mass
        self._added_base_mass[env_ids] = added_mass[:].detach().clone()
        raw_mass = self._robot.root_physx_view.get_masses().to(self._device)
        base_mass_after_dr = self._default_base_mass[env_ids] + added_mass
        mass_after_dr = raw_mass.clone()
        mass_after_dr[env_ids, self._base_link_index] = base_mass_after_dr.squeeze(1)
        all_indices = torch.arange(self._robot.root_physx_view.count, device="cpu")
        # tensors passed to set_masses must be on CPU
        self._robot.root_physx_view.set_masses(mass_after_dr.to("cpu"), all_indices)
        
    def _randomize_com_displacement(self, env_ids):
        if len(env_ids) == 0:
            return
        min_displacement_x, max_displacement_x = self._cfg.domain_rand.com_pos_x_range
        min_displacement_y, max_displacement_y = self._cfg.domain_rand.com_pos_y_range
        min_displacement_z, max_displacement_z = self._cfg.domain_rand.com_pos_z_range
        
        com_displacement = torch.zeros((len(env_ids), 1, 3), dtype=torch.float, device=self._device)
        com_displacement[:, 0, 0] = torch.rand((len(env_ids), 1), dtype=torch.float, device=self._device).squeeze(1) \
            * (max_displacement_x - min_displacement_x) + min_displacement_x
        com_displacement[:, 0, 1] = torch.rand((len(env_ids), 1), dtype=torch.float, device=self._device).squeeze(1) \
            * (max_displacement_y - min_displacement_y) + min_displacement_y
        com_displacement[:, 0, 2] = torch.rand((len(env_ids), 1), dtype=torch.float, device=self._device).squeeze(1) \
            * (max_displacement_z - min_displacement_z) + min_displacement_z
        self._base_com_bias[env_ids] = com_displacement[:, 0, :].detach().clone()

        base_com_after_dr = self._default_com_pos[env_ids] + com_displacement[:, 0, :]
        # refer to https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.set_coms
        raw_coms = self._robot.root_physx_view.get_coms().to(self._device)
        com_after_dr = raw_coms.clone()
        com_after_dr[env_ids, self._base_link_index, :3] = base_com_after_dr[:]
        all_indices = torch.arange(self._robot.root_physx_view.count, device="cpu")
        # tensors passed to set_coms must be on CPU
        self._robot.root_physx_view.set_coms(com_after_dr.to("cpu"), all_indices)
    
    def _randomize_joint_armature(self, env_ids):
        if len(env_ids) == 0:
            return
        
        min_armature, max_armature = self._cfg.domain_rand.joint_armature_range
        armature = torch.rand((len(env_ids),), dtype=torch.float, device=self._device) \
            * (max_armature - min_armature) + min_armature
        # save values to domain randomization params
        self._joint_armature[env_ids, 0] = armature.detach().clone()
        # [len(env_ids)] -> [len(env_ids), num_actions], all joints within the same env have the same armature
        armature = armature.unsqueeze(1).repeat(1, self.num_actions)
        # refer to https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.Articulation.write_joint_armature_to_sim
        self._robot.write_joint_armature_to_sim(armature, None, env_ids)
    
    def _randomize_joint_friction(self, env_ids):
        if len(env_ids) == 0:
            return
        
        min_friction, max_friction = self._cfg.domain_rand.joint_friction_range
        friction = torch.rand((len(env_ids), 1), dtype=torch.float, device=self._device) \
            * (max_friction - min_friction) + min_friction
        self._joint_friction[env_ids] = friction.detach().clone()
        # All joints within the same env have the same friction
        friction = friction.repeat(1, self.num_actions)
        # refer to https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.Articulation.write_joint_friction_coefficient_to_sim
        self._robot.write_joint_friction_coefficient_to_sim(
            friction, None, None, None, env_ids) # currently, only static friction coefficients are considered
        
    def _randomize_joint_damping(self, env_ids):
        if len(env_ids) == 0:
            return
        
        min_damping, max_damping = self._cfg.domain_rand.joint_damping_range
        damping = torch.rand((len(env_ids), 1), dtype=torch.float, device=self._device) \
            * (max_damping - min_damping) + min_damping
        self._joint_damping[env_ids] = damping.detach().clone()
        damping = damping.repeat(1, self.num_actions)
        # refer to https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.Articulation.write_joint_damping_to_sim
        self._robot.write_joint_damping_to_sim(damping, None, env_ids)
        
    def _randomize_pd_gain(self, env_ids):
        self._kp_scale[env_ids] = torch_rand_float(
                self._cfg.domain_rand.kp_range[0], self._cfg.domain_rand.kp_range[1], (len(env_ids), self.num_actions), device=self._device)
        self._kd_scale[env_ids] = torch_rand_float(
                self._cfg.domain_rand.kd_range[0], self._cfg.domain_rand.kd_range[1], (len(env_ids), self.num_actions), device=self._device)