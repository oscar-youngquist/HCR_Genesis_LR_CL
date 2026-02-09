from abc import ABC, abstractmethod
from torch import Tensor

""" ********** Base Simulator ********** """
class Simulator(ABC):
    def __init__(self, cfg, sim_params: dict, sim_device: str = "cuda:0", headless: bool = False):
        self.height_samples = None
        self.device = sim_device
        self.headless = headless
        self.cfg = cfg
        self.num_envs = self.cfg.env.num_envs
        self.num_actions = self.cfg.env.num_actions
        self._parse_cfg()
        self._create_sim()
        self._create_envs()
        self._init_buffers()

    #----- Public methods -----#
    @abstractmethod
    def _parse_cfg(self):
        """Parses the configuration file and initializes necessary variables for the simulator.
        """
        return

    @abstractmethod
    def _create_sim(self):
        """Creates the simulation environment, including the physics engine and any necessary components.
        """
        return

    @abstractmethod
    def _create_envs(self):
        """Creates environments, adds the robot asset to each environment, sets DOF properties and calls callbacks to process rigid shape, rigid body and DOF properties.
        """
        return

    @abstractmethod
    def _init_buffers(self):
        """Initializes necessary buffers for the simulator, such as those for observations, actions, rewards, and any other relevant data.
        """
        return

    @abstractmethod
    def step(self):
        """Performs a simulation step, which typically includes applying actions, stepping the physics simulation, and updating states and observations.
        """
        return

    @abstractmethod
    def post_physics_step(self):
        """Performs any necessary updates after the physics step.
        """
        return
    
    @abstractmethod
    def reset_idx(self, env_ids: Tensor):
        """Reset environments with the given environment IDs.

        Args:
            env_ids (Tensor): Environment IDs to reset.
        """
        return
    
    @abstractmethod
    def reset_dofs(self, env_ids: Tensor, dof_pos: Tensor, dof_vel: Tensor):
        """Reset the DOF states of the environments with the given environment IDs.

        Args:
            env_ids (Tensor): Environment IDs to reset.
            dof_pos (Tensor): DOF positions to reset.
            dof_vel (Tensor): DOF velocities to reset.
        """
        return
    
    @abstractmethod
    def reset_root_states(self, env_ids: Tensor, base_pos: Tensor, base_quat: Tensor, base_lin_vel: Tensor, base_ang_vel: Tensor):
        """Reset the root states of the environments with the given environment IDs.
        
        Args:
            env_ids (Tensor): Environment IDs to reset.
            base_pos (Tensor): Base positions to reset.
            base_quat (Tensor): Base orientations (quaternions) to reset.
            base_lin_vel (Tensor): Base linear velocities to reset.
            base_ang_vel (Tensor): Base angular velocities to reset.
        """
        return

    #----- Protected methods -----#
    @abstractmethod
    def _update_surrounding_heights(self):
        """Updates the height of the sampling points around the robot.
        
        The sampling grid is defined in LeggedRobotCfg.terrain.measured_points_x/y.
        """
        return

    @abstractmethod
    def _push_robots(self):
        """Apply perturbation velocity to the base of the robot as domain randomization.
        """
        return
    
    @abstractmethod
    def _init_domain_params(self):
        """Initializes domain randomization parameters, which are used as privilege information.
        """
        return
    
    @abstractmethod
    def _randomize_friction(self, env_ids):
        """Randomizes the friction of all links for the given environment IDs.

        Args:
            env_ids (Tensor, optional): Environment IDs to randomize friction for. If None, randomizes friction for all environments. Defaults to None.
        """
        return
    
    #----- Properties -----#
    @property
    def feet_indices(self):
        """Returns the indices of the feet links in the robot articulation.

        Returns:
            list[int]: Indices of the feet links.
        """
        return self._feet_indices
    
    @property
    def feet_contact_indices(self):
        """Returns the indices of the feet links in the contact sensors.

        Returns:
            list[int]: Indices of the feet links in the contact sensors.
        """
        return self._feet_contact_indices
    
    @property
    def termination_contact_indices(self):
        """Returns the indices of the links in the contact sensors that are used for termination checking.

        Returns:
            list[int]: Indices of the links in the contact sensors for termination checking.
        """
        return self._termination_contact_indices
    
    @property
    def penalized_contact_indices(self):
        """Returns the indices of the links in the contact sensors that are used for penalty.

        Returns:
            list[int]: Indices of the links in the contact sensors for penalty.
        """
        return self._penalized_contact_indices