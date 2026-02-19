
import numpy as np

from isaacgym import gymapi

from legged_gym.envs.base_g1.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class _G1SimulatorView:
    """Compatibility shim for scripts expecting `env.simulator.*`.

    LeggedGym-Ex scripts (e.g. `scripts/play.py`) log via `env.simulator` for
    simulator-abstracted envs. The Unitree-style `LeggedRobot` exposes these
    tensors directly on the env, so we provide a minimal view here.
    """

    def __init__(self, env: "G1Robot"):
        self._env = env

    @property
    def base_pos(self):
        return self._env.base_pos

    @property
    def base_lin_vel(self):
        return self._env.base_lin_vel

    @property
    def base_ang_vel(self):
        return self._env.base_ang_vel

    @property
    def dof_pos(self):
        return self._env.dof_pos

    @property
    def dof_vel(self):
        return self._env.dof_vel

    @property
    def torques(self):
        return self._env.torques

    @property
    def link_contact_forces(self):
        return self._env.contact_forces

    @property
    def feet_indices(self):
        return self._env.feet_indices

class G1Robot(LeggedRobot):

    def __init__(self, cfg, sim_params, sim_device, headless):
        """Adapter for LeggedGym-Ex's `task_registry` signature.

        LeggedGym-Ex passes `sim_params` as a nested dict; Unitree's original base
        expects a `gymapi.SimParams`. We convert via gymutil.parse_sim_config so
        enums (contact_collection, solver_type, etc.) are handled correctly.
        """
        sim_params_obj = gymapi.SimParams()
        if isinstance(sim_params, dict):
            gymutil.parse_sim_config(sim_params, sim_params_obj)
        # ensure GPU pipeline matches device
        sim_params_obj.use_gpu_pipeline = str(sim_device).startswith("cuda")
        if hasattr(sim_params_obj, "physx"):
            sim_params_obj.physx.use_gpu = str(sim_device).startswith("cuda")

        super().__init__(
            cfg=cfg,
            sim_params=sim_params_obj,
            physics_engine=gymapi.SIM_PHYSX,
            sim_device=sim_device,
            headless=headless,
        )
        self.simulator = _G1SimulatorView(self)
        self._printed_control_sanity = False
        self._last_dof_pos_0 = None

    def set_viewer_camera(self, position, lookat):
        # LeggedGym-Ex play script expects this name.
        return self.set_camera(position, lookat)
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

        if (not self._printed_control_sanity) and int(self.common_step_counter) == 10:
            try:
                with torch.inference_mode():
                    env0 = 0
                    action_mean_abs = float(self.actions[env0].abs().mean().item())
                    torque_mean_abs = float(self.torques[env0].abs().mean().item())
                    dof_err_mean_abs = float((self.dof_pos[env0] - self.default_dof_pos[0]).abs().mean().item())
                    cmd_xy = self.commands[env0, :2].detach().cpu().numpy()
                    vel_xy = self.base_lin_vel[env0, :2].detach().cpu().numpy()
                    if self._last_dof_pos_0 is None:
                        dof_delta_mean_abs = float("nan")
                    else:
                        dof_delta_mean_abs = float((self.dof_pos[env0] - self._last_dof_pos_0).abs().mean().item())
                    self._last_dof_pos_0 = self.dof_pos[env0].clone()

                print(
                    "[G1 CONTROL CHECK] "
                    f"|a|mean={action_mean_abs:.3f} |tau|mean={torque_mean_abs:.3f} "
                    f"|q-q0|mean={dof_err_mean_abs:.3f} dq(mean_abs_step)={dof_delta_mean_abs:.6f} "
                    f"cmd_xy=({cmd_xy[0]:.2f},{cmd_xy[1]:.2f}) vel_xy=({vel_xy[0]:.2f},{vel_xy[1]:.2f})"
                )
            except Exception as e:
                print("[G1 CONTROL CHECK] failed:", repr(e))
            self._printed_control_sanity = True

        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        
    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)
    