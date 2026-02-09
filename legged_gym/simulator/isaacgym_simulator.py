from legged_gym import *
from legged_gym.simulator.simulator import Simulator
from PIL import Image as im
import cv2 as cv
import torch
import numpy as np
import os
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math_utils import *
if SIMULATOR == "isaacgym":
    from isaacgym import gymtorch, gymapi, gymutil
    import warp as wp
    import trimesh
    from legged_gym.warp.warp_cam import WarpCam

""" ********** Isaac Gym Simulator ********** """
class IsaacGymSimulator(Simulator):
    """Simulator class for Isaac Gym"""
    def __init__(self, cfg, sim_params: dict, sim_device: str = "cuda:0", headless: bool = False):
        self.gym = gymapi.acquire_gym()
        # Convert dict sim_params to gymapi.SimParams
        self.sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(sim_params, self.sim_params)
        _, self.sim_device_id = gymutil.parse_device_str(sim_device)
        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if headless == True:
            self.graphics_device_id = -1
        self.physics_engine = gymapi.SIM_PHYSX
        super().__init__(cfg, sim_params, sim_device, headless)
        # warp init
        if self.cfg.sensor.use_warp:
            assert self.cfg.sensor.add_depth, "Depth sensor is required for warp"
            wp.init()
            self._create_warp_envs()
            self._create_warp_tensors()
            self.sensor = WarpCam(self.warp_tensor_dict, self.num_envs, self.cfg.sensor, self.mesh_ids, self.device)
            pixels = self.sensor.update()
            self.depth_images[:,0] = pixels[:,0] # pixels: [num_envs, num_sensors, H, W]

    def _parse_cfg(self):
        self.debug = self.cfg.env.debug
        self.control_dt = self.cfg.sim.dt * self.cfg.control.decimation
        if self.cfg.sensor.add_depth:
            self.frame_count = 0
    
    def _create_sim(self):
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        
        # specify the boundary of the heightfield
        self.terrain_x_range = torch.zeros(2, device=self.device)
        self.terrain_y_range = torch.zeros(2, device=self.device)
        if self.cfg.terrain.mesh_type == 'heightfield' or self.cfg.terrain.mesh_type == 'trimesh':
            # give a small margin(1.0m)
            self.terrain_x_range[0] = -self.cfg.terrain.border_size + 1.0
            self.terrain_x_range[1] = self.cfg.terrain.border_size + \
                self.cfg.terrain.num_rows * self.cfg.terrain.terrain_length - 1.0
            self.terrain_y_range[0] = -self.cfg.terrain.border_size + 1.0
            self.terrain_y_range[1] = self.cfg.terrain.border_size + \
                self.cfg.terrain.num_cols * self.cfg.terrain.terrain_width - 1.0
        elif self.cfg.terrain.mesh_type == 'plane':  # the plane used has limited size,
            pass
        
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        print(f"body_names: {body_names}")
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        print(f"dof_names: {self.dof_names}")
        self.num_bodies = len(body_names)
        self.num_dof = len(self.dof_names)
        self.feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        if self.cfg.asset.obtain_link_contact_states:
            contact_state_link_names = []
            for name in self.cfg.asset.contact_state_link_names:
                contact_state_link_names.extend([s for s in body_names if name in s])

        self.base_init_pos = torch.tensor(self.cfg.init_state.pos, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_init_quat = torch.tensor(self.cfg.init_state.rot, dtype=torch.float, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_pos)

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        
        # privileged information
        self._init_domain_params()
        
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions_gym, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(self.feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.feet_names[i])

        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
        
        if self.cfg.asset.obtain_link_contact_states:
            self.contact_state_link_indices = torch.zeros(len(contact_state_link_names), dtype=torch.long, device=self.device, requires_grad=False)
            for i in range(len(contact_state_link_names)):
                self.contact_state_link_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], contact_state_link_names[i])

        self.gym.prepare_sim(self.sim)
        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
    
    def _create_warp_envs(self):
        terrain_mesh = trimesh.Trimesh(vertices=self.terrain.vertices, faces=self.terrain.triangles)
        
        #save terrain mesh
        transform = np.zeros((3,))
        transform[0] = -self.cfg.terrain.border_size 
        transform[1] = -self.cfg.terrain.border_size
        transform[2] = 0.0
        translation = trimesh.transformations.translation_matrix(transform)
        terrain_mesh.apply_transform(translation)
        
        vertices = terrain_mesh.vertices
        triangles = terrain_mesh.faces
        vertex_tensor = torch.tensor(vertices, 
                                     dtype=torch.float32, 
                                     device=self.device,
                                     requires_grad=False)
        #if none type in vertex_tensor
        if vertex_tensor.any() is None:
            print("vertex_tensor is None")
        vertex_vec3_array = wp.from_torch(vertex_tensor,dtype=wp.vec3)        
        faces_wp_int32_array = wp.from_numpy(triangles.flatten(), dtype=wp.int32,device=self.device)
        
        # mesh coordinate convention may be the same as isaacgym
        self.wp_meshes =  wp.Mesh(points=vertex_vec3_array,indices=faces_wp_int32_array)
        
        self.mesh_ids = self.mesh_ids_array = wp.array([self.wp_meshes.id], dtype=wp.uint64)
    
    def _create_warp_tensors(self):
        self.warp_tensor_dict={}
        pointcloud_dims = 3 * (self.cfg.sensor.depth_camera_config.return_pointcloud == True)
        if pointcloud_dims > 0:
            self.depth_image_tensor_warp = torch.zeros((self.num_envs, 
                                                        self.cfg.sensor.num_sensors,
                                                        self.cfg.sensor.depth_camera_config.resolution[1],  # height
                                                        self.cfg.sensor.depth_camera_config.resolution[0], # width
                                                        pointcloud_dims),    # xyz
                                                       dtype=torch.float32, device=self.device)
        else:
            self.depth_image_tensor_warp = torch.zeros((self.num_envs, 
                                                        self.cfg.sensor.num_sensors,
                                                        self.cfg.sensor.depth_camera_config.resolution[1],  # height
                                                        self.cfg.sensor.depth_camera_config.resolution[0]), # width
                                                    dtype=torch.float32, device=self.device)
        self.sensor_pos_tensor = torch.zeros_like(self.root_states[:, 0:3])
        self.sensor_quat_tensor = torch.zeros_like(self.root_states[:, 3:7])
        
        # sensor pose
        pos_offset = [self.cfg.sensor.depth_camera_config.pos[0], 
                      self.cfg.sensor.depth_camera_config.pos[1], 
                      self.cfg.sensor.depth_camera_config.pos[2]]
        rpy_offset = [self.cfg.sensor.depth_camera_config.euler[0], 
                      self.cfg.sensor.depth_camera_config.euler[1], 
                      self.cfg.sensor.depth_camera_config.euler[2]]
        self.sensor_offset_pos = torch.tensor(pos_offset, device=self.device).repeat((self.num_envs, 1))
        rpy_offset = torch.tensor(rpy_offset, device=self.device)

        self.sensor_offset_quat = quat_from_euler_xyz(rpy_offset[0], rpy_offset[1], rpy_offset[2]).repeat((self.num_envs, 1))
        
        self.warp_tensor_dict["depth_image_tensor"] = self.depth_image_tensor_warp
        self.warp_tensor_dict['device'] = self.device
        self.warp_tensor_dict['num_envs'] = self.num_envs
        self.warp_tensor_dict['num_sensors'] = self.cfg.sensor.num_sensors
        self.warp_tensor_dict['sensor_pos_tensor'] = self.sensor_pos_tensor
        self.warp_tensor_dict['sensor_quat_tensor'] = self.sensor_quat_tensor
        self.warp_tensor_dict['mesh_ids'] = self.mesh_ids
        
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.base_euler = get_euler_xyz(self.base_quat)
        self.link_contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.last_feet_vel = torch.zeros_like(self.feet_vel)

        # initialize some data used later on
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.last_base_lin_vel = torch.zeros_like(self.base_lin_vel)
        self.last_base_ang_vel = torch.zeros_like(self.base_ang_vel)
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.global_gravity)
        
        # Terrain information around feet
        if self.cfg.terrain.obtain_terrain_info_around_feet:
            self.normal_vector_around_feet = torch.zeros(
                self.num_envs, len(self.feet_indices) * 3, dtype=torch.float, device=self.device, requires_grad=False)
            self.height_around_feet = torch.zeros(
                self.num_envs, len(self.feet_indices), 9, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Link contact state
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
        
        # depth images
        if self.cfg.sensor.add_depth:
            pointcloud_dims = 3 * (self.cfg.sensor.depth_camera_config.return_pointcloud == True)
            if pointcloud_dims > 0:
                self.depth_images = torch.zeros(
                    (self.num_envs, 
                     self.cfg.sensor.depth_camera_config.num_history,
                     self.cfg.sensor.depth_camera_config.resolution[1], 
                     self.cfg.sensor.depth_camera_config.resolution[0],
                     pointcloud_dims), 
                    device=self.device, 
                    dtype=torch.float
                )
            else:
                self.depth_images = torch.zeros(
                    (self.num_envs, 
                    self.cfg.sensor.depth_camera_config.num_history,
                    self.cfg.sensor.depth_camera_config.resolution[1], 
                    self.cfg.sensor.depth_camera_config.resolution[0]), 
                    device=self.device, 
                    dtype=torch.float
                )
    
    def step(self, actions):
        """Simulator steps, receiving actions from the agent"""
        self._render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
    
    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._check_base_pos_out_of_bound()
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # the wrapped tensor will be updated automatically once you call refresh_xxx_tensor
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_euler[:] = get_euler_xyz(self.base_quat)
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.global_gravity)
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        # Link contact state
        if self.cfg.asset.obtain_link_contact_states:
            self.link_contact_states = 1. * (torch.norm(
                self.link_contact_forces[:, self.contact_state_link_indices, :], dim=-1) > 1.)
        if self.cfg.sensor.use_warp:
            # Refresh warp sensor pose
            sensor_quat = quat_mul(self.base_quat, self.sensor_offset_quat)
            sensor_pos = self.base_pos + quat_apply(self.base_quat, self.sensor_offset_pos)
            self.sensor_pos_tensor[:,:] = sensor_pos[:,:]
            self.sensor_quat_tensor[:,:] = sensor_quat[:,:]
    
    def update_depth_images(self):
        """ Update depth images from the depth camera sensors
        """
        if self.cfg.sensor.use_warp:
            pixels = self.sensor.update()
            self.depth_images[:, 0] = pixels[:,0] # pixels: [num_envs, num_sensors, H, W]
            if self.cfg.sensor.depth_camera_config.calculate_depth:
                near_clip = self.cfg.sensor.depth_camera_config.near_clip
                far_clip = self.cfg.sensor.depth_camera_config.far_clip
                # clip the depth images to be within near and far clip
                self.depth_images = torch.clip(self.depth_images, near_clip, far_clip)
                # normalize the depth images to be within 0-1
                self.depth_images = (self.depth_images - near_clip) / (far_clip - near_clip) - 0.5
        else:
            raise NotImplementedError("Depth image update not implemented for non-warp simulator")
            
    
    def _check_base_pos_out_of_bound(self):
        """ Check if the base position is out of the terrain bounds
        """
        if self.cfg.terrain.mesh_type == "plane":
            return
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
            self.root_states[env_ids, 0:3] = self.base_pos[env_ids]
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def update_surrounding_heights(self):
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
    
    def get_height_at(self, pos_xy):
        """ Get height of the terrain at specific (x, y) location

        Args:
            pos_xy (torch.tensor): (x, y) locations to sample heights at, shape: (len(env_ids), 2)
        Returns:
            torch.tensor: heights at the specified (x, y) locations, shape: (len(env_ids),)
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros_like(pos_xy[:, 0], device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError(
                "Can't measure height with terrain mesh type 'none'")

        points = pos_xy + self.cfg.terrain.border_size # add border size to align the origin with heightfield raw
        points = (points/self.cfg.terrain.horizontal_scale).long()
        px = points[:, 0]
        py = points[:, 1]
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)
        
        return self.height_samples[px, py] * self.cfg.terrain.vertical_scale
    
    def calc_terrain_info_around_feet(self):
        """ Finds neighboring points around each foot for terrain height measurement."""
        # Foot position
        foot_points = self.feet_pos + self.cfg.terrain.border_size
        foot_points = (foot_points/self.cfg.terrain.horizontal_scale).long()
        # px and py for 4 feet, num_envs*len(feet_indices)
        px = foot_points[:, :, 0].view(-1)
        py = foot_points[:, :, 1].view(-1)
        # clip to the range of height samples
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)
        # get heights around the feet, 9 points for each foot
        heights1 = self.height_samples[px-1, py]  # [x-0.1, y]
        heights2 = self.height_samples[px+1, py]  # [x+0.1, y]
        heights3 = self.height_samples[px, py-1]  # [x, y-0.1]
        heights4 = self.height_samples[px, py+1]  # [x, y+0.1]
        heights5 = self.height_samples[px, py]    # [x, y]
        heights6 = self.height_samples[px-1, py-1]  # [x-0.1, y-0.1]
        heights7 = self.height_samples[px+1, py+1]  # [x+0.1, y+0.1]
        heights8 = self.height_samples[px-1, py+1]  # [x-0.1, y+0.1]
        heights9 = self.height_samples[px+1, py-1]  # [x+0.1, y-0.1]
        # Calculate normal vectors around feet
        dx = ((heights2 - heights1) / (self.cfg.terrain.horizontal_scale * 2)).view(self.num_envs, -1)
        dy = ((heights4 - heights3) / (self.cfg.terrain.horizontal_scale * 2)).view(self.num_envs, -1)
        for i in range(len(self.feet_indices)):
            normal_vector = torch.cat((dx[:, i].unsqueeze(1), dy[:, i].unsqueeze(1), 
                -1*torch.ones_like(dx[:, i].unsqueeze(1))), dim=-1).to(self.device)
            normal_vector /= torch.norm(normal_vector, dim=-1, keepdim=True)
            self.normal_vector_around_feet[:, i*3:i*3+3] = normal_vector[:]
        # Calculate height around feet
        for i in range(9):
            self.height_around_feet[:, :, i] = eval(f'heights{i+1}').view(self.num_envs, -1)[:] * self.cfg.terrain.vertical_scale
    
    def draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        # if not self.cfg.terrain.measure_heights:
        #     return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.cfg.env.debug_draw_height_points:
            self.draw_height_points()
        if self.cfg.env.debug_draw_height_points_around_feet:
            self.draw_height_points_around_feet()
    
    def draw_debug_points_world(self, points, radius=0.02, color=(1, 0, 0)):
        """ Draws debug points in world frame

        Args:
            points (torch.tensor): points to draw, shape: (num_envs, num_points, 3)
            color (tuple, optional): RGB color of the points. Defaults to (1, 0, 0).
        """
        sphere_geom = gymutil.WireframeSphereGeometry(radius, 4, 4, None, color=color)
        for i in range(self.num_envs):
            for j in range(points.shape[1]):
                x = points[i, j, 0].cpu().numpy()
                y = points[i, j, 1].cpu().numpy()
                z = points[i, j, 2].cpu().numpy()
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
    
    def draw_height_points(self):
        # draw height lines
        if not self.cfg.terrain.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
    
    def draw_height_points_around_feet(self):
        """ Draws height measurement points around feet for debugging
        """
        # Height points around feet
        height_points = torch.zeros(self.num_envs, 9*len(self.feet_indices), 3, device=self.device)
        foot_points = self.feet_pos + self.cfg.terrain.border_size
        foot_points = (foot_points/self.cfg.terrain.horizontal_scale).long()
        px = foot_points[:, :, 0].view(-1)
        py = foot_points[:, :, 1].view(-1)
        heights1 = self.height_samples[px-1, py]  # [x-0.1, y]
        heights2 = self.height_samples[px+1, py]  # [x+0.1, y]
        heights3 = self.height_samples[px, py-1]  # [x, y-0.1]
        heights4 = self.height_samples[px, py+1]  # [x, y+0.1]
        heights5 = self.height_samples[px, py]    # [x, y]
        heights6 = self.height_samples[px-1, py-1]  # [x-0.1, y-0.1]
        heights7 = self.height_samples[px+1, py+1]  # [x+0.1, y+0.1]
        heights8 = self.height_samples[px-1, py+1]  # [x-0.1, y+0.1]
        heights9 = self.height_samples[px+1, py-1]  # [x+0.1, y-0.1]
        for i in range(len(self.feet_indices)):
            height_points[0, i*9+0, 0] = (px-1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i*9+0, 1] = (py-1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i*9+0, 2] = heights6.view(self.num_envs, -1)[0, i] * self.cfg.terrain.vertical_scale
            height_points[0, i*9+1, 0] = (px-1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i*9+1, 1] = py.view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i*9+1, 2] = heights1.view(self.num_envs, -1)[0, i] * self.cfg.terrain.vertical_scale
            height_points[0, i*9+2, 0] = px.view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i*9+2, 1] = (py-1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i*9+2, 2] = heights3.view(self.num_envs, -1)[0, i] * self.cfg.terrain.vertical_scale
            height_points[0, i*9+3, 0] = px.view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i*9+3, 1] = (py+1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i*9+3, 2] = heights4.view(self.num_envs, -1)[0, i] * self.cfg.terrain.vertical_scale
            height_points[0, i*9+4, 0] = px.view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i*9+4, 1] = py.view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i*9+4, 2] = heights5.view(self.num_envs, -1)[0, i] * self.cfg.terrain.vertical_scale
            height_points[0, i*9+5, 0] = (px+1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i*9+5, 1] = py.view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i*9+5, 2] = heights2.view(self.num_envs, -1)[0, i] * self.cfg.terrain.vertical_scale
            height_points[0, i*9+6, 0] = (px+1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i*9+6, 1] = (py+1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i*9+6, 2] = heights7.view(self.num_envs, -1)[0, i] * self.cfg.terrain.vertical_scale
            height_points[0, i*9+7, 0] = (px-1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i*9+7, 1] = (py+1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i*9+7, 2] = heights8.view(self.num_envs, -1)[0, i] * self.cfg.terrain.vertical_scale
            height_points[0, i*9+8, 0] = (px+1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i*9+8, 1] = (py-1).view(self.num_envs, -1)[0, i] * self.cfg.terrain.horizontal_scale - self.cfg.terrain.border_size
            height_points[0, i*9+8, 2] = heights9.view(self.num_envs, -1)[0, i] * self.cfg.terrain.vertical_scale
        
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 1, 0))
        for i in range(self.num_envs):
            for j in range(9*len(self.feet_indices)):
                x = height_points[i, j, 0].cpu().numpy()
                y = height_points[i, j, 1].cpu().numpy()
                z = height_points[i, j, 2].cpu().numpy()
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
    
    def draw_debug_boxes(self, pos, orientation):
        """
        Draws debug boxes at specified positions and orientations.
        Args:
            pos (torch.Tensor): Tensor of shape (num_envs, 3) specifying the positions of the boxes.
            orientation (torch.Tensor): Tensor of shape (num_envs, 4) specifying the orientations (quaternions) of the boxes.
        """
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        box_geom = gymutil.WireframeBoxGeometry(0.5, 0.25, 0.25, color=(0, 1, 0))
        for i in range(self.num_envs):
            box_pos = pos[i].cpu().numpy()
            box_quat = orientation[i].cpu().numpy()
            box_pose = gymapi.Transform(gymapi.Vec3(box_pos[0], box_pos[1], box_pos[2]),
                                        gymapi.Quat(box_quat[0], box_quat[1], box_quat[2], box_quat[3]))
            gymutil.draw_lines(box_geom, self.gym, self.viewer, self.envs[i], box_pose)
    
    def draw_debug_depth_images(self):
        depth = self.depth_images[0, 0, :, :]  # get depth image of first env, first step
        # print(f"depth values: {depth}")
        if self.cfg.sensor.depth_camera_config.calculate_depth:
            far_clip = self.cfg.sensor.depth_camera_config.far_clip
            near_clip = self.cfg.sensor.depth_camera_config.near_clip
            pixel_values = ((depth + 0.5) * 255.0).cpu().numpy().astype(np.uint8)
            # image = im.fromarray(pixel_values, mode='L')
            # image.save("debug_depth_images/depth_frame%d.jpg" % self.frame_count)
            cv.imshow("Depth Camera", pixel_values)
            cv.waitKey(1)
            pixel_values = (depth + 0.5) * (far_clip - near_clip) + near_clip
            pixel_values = pixel_values.cpu().numpy().astype(np.float32)
            print(f"depth pixel values: {pixel_values}")
            # self.frame_count += 1
        elif self.cfg.sensor.depth_camera_config.return_pointcloud:
            pointcloud = self.depth_images[0, 0, :, :, :]  # get pointcloud of first env, first step
            pointcloud_np = pointcloud.cpu().numpy().astype(np.float32)
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            base_pos = (self.root_states[0, :3]).cpu().numpy()
            for i in range(pointcloud_np.shape[0]):
                for j in range(pointcloud_np.shape[1]):
                    x = pointcloud_np[i, j, 0]
                    y = pointcloud_np[i, j, 1]
                    z = pointcloud_np[i, j, 2]
                    sphere_geom = gymutil.WireframeSphereGeometry(0.01, 4, 4, None, color=(0, 1, 1))
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose)
            # print(f"pointcloud values: {pointcloud_np}")
    
    def push_robots(self):
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self._rand_push_vels[:, :2] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)
        self.root_states[:, 7:9] = self._rand_push_vels[:, :2] # set random base velocity in xy plane
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
    
    def reset_idx(self, env_ids):
        
        if self.cfg.domain_rand.randomize_pd_gain:
            self._kp_scale[env_ids] = torch_rand_float(
                self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_actions), device=self.device)
            self._kd_scale[env_ids] = torch_rand_float(
                self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_actions), device=self.device)
        
        self.last_dof_vel[env_ids] = 0.
        self.last_feet_vel[env_ids] = 0.
        self.last_base_lin_vel[env_ids] = 0.
        self.last_base_ang_vel[env_ids] = 0.
        
        # fix reset gravity bug
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.projected_gravity[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.global_gravity[env_ids])
    
    def reset_dofs(self, env_ids, dof_pos, dof_vel):
        self.dof_pos[env_ids, :] = dof_pos[:]
        self.dof_vel[env_ids, :] = dof_vel[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(
                                                  self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    # ------------- Callbacks --------------
    
    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y,
                         device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x,
                         device=self.device, requires_grad=False)
        # Get index of 4 points around robot base
        self.num_x_points = x.shape[0]
        self.num_y_points = y.shape[0]
        self.front_point_index = (self.num_x_points // 2 + 2) * self.num_y_points \
            + (self.num_y_points - 1) // 2 # [base_pos_x+2*horizontal_scale, base_pos_y]
        self.rear_point_index = (self.num_x_points // 2 - 2) * self.num_y_points \
            + (self.num_y_points - 1) // 2 # [base_pos_x-2*horizontal_scale, base_pos_y]
        self.left_point_index = self.num_x_points // 2 * self.num_y_points \
            + self.num_y_points // 2 + 1   # [base_pos_x, base_pos_y+horizontal_scale]
        self.right_point_index = self.num_x_points // 2 * self.num_y_points \
            + self.num_y_points // 2 - 1   # [base_pos_x, base_pos_y-horizontal_scale]
        
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        self.num_height_points = grid_x.numel()
        self.height_points = torch.zeros(self.num_envs, self.num_height_points,
                             3, device=self.device, requires_grad=False)
        self.height_points[:, :, 0] = grid_x.flatten()
        self.height_points[:, :, 1] = grid_y.flatten()
    
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
    
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
    
    def reset_root_states(self, env_ids, base_pos, base_quat, base_lin_vel, base_ang_vel):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        self.root_states[env_ids, 0:3] = base_pos[:]
        # base orientation
        self.root_states[env_ids, 3:7] = base_quat[:]
        # base velocities
        self.root_states[env_ids, 7:10] = base_lin_vel[:]
        self.root_states[env_ids, 10:13] = base_ang_vel[:]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
    
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
            self._friction_values[env_id, :] = self.friction_coeffs[env_id]
        
        return props
    
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        
        if self.cfg.domain_rand.randomize_joint_friction:
            joint_friction_range = np.array(
                self.cfg.domain_rand.joint_friction_range, dtype=np.float32)
            friction = np.random.uniform(
                joint_friction_range[0], joint_friction_range[1])
            self._joint_friction[env_id] = friction
            for j in range(self.num_dof):
                props["friction"][j] = torch.tensor(
                    friction, dtype=torch.float, device=self.device)

        if self.cfg.domain_rand.randomize_joint_damping:
            joint_damping_range = np.array(
                self.cfg.domain_rand.joint_damping_range, dtype=np.float32)
            damping = np.random.uniform(
                joint_damping_range[0], joint_damping_range[1])
            self._joint_damping[env_id] = damping
            for j in range(self.num_dof):
                props["damping"][j] = torch.tensor(
                    damping, dtype=torch.float, device=self.device)

        if self.cfg.domain_rand.randomize_joint_armature:
            joint_armature_range = np.array(
                self.cfg.domain_rand.joint_armature_range, dtype=np.float32)
            armature = np.random.uniform(
                joint_armature_range[0], joint_armature_range[1])
            self._joint_armature[env_id] = armature
            for j in range(self.num_dof):
                props["armature"][j] = torch.tensor(
                    armature, dtype=torch.float, device=self.device)

        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            added_base_mass = np.random.uniform(rng[0], rng[1])
            props[0].mass += added_base_mass
        self._added_base_mass[env_id] = added_base_mass

        # randomize com position
        if self.cfg.domain_rand.randomize_com_displacement:
            com_x_bias = np.random.uniform(
                self.cfg.domain_rand.com_pos_x_range[0], self.cfg.domain_rand.com_pos_x_range[1])
            com_y_bias = np.random.uniform(
                self.cfg.domain_rand.com_pos_y_range[0], self.cfg.domain_rand.com_pos_y_range[1])
            com_z_bias = np.random.uniform(
                self.cfg.domain_rand.com_pos_z_range[0], self.cfg.domain_rand.com_pos_z_range[1])

            self._base_com_bias[env_id, 0] += com_x_bias
            self._base_com_bias[env_id, 1] += com_y_bias
            self._base_com_bias[env_id, 2] += com_z_bias

            # randomize com position of "base1_downbox"
            props[0].com.x += com_x_bias
            props[0].com.y += com_y_bias
            props[0].com.z += com_z_bias
            # print(f"com of base: {props[0].com} (after randomization)")
        
        return props
    
    def _init_domain_params(self):
        """ Initializes domain randomization parameters, which are used to randomize the environment."""
        self._friction_values = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
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
        
    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='K'), self.terrain.triangles.flatten(order='K'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)