from legged_gym import *
import os

from legged_gym.envs import *
from legged_gym.utils import *

import numpy as np
import torch

def export_policy(alg_runner, path, task_name, prefix=None):
    if "ts" in task_name or "cat" in task_name:
        exporter = PolicyExporterTS(alg_runner.alg.actor_critic)
        exporter.export(path, prefix)
    elif "ee" in task_name:
        exporter = PolicyExporterEE(alg_runner.alg.actor_critic)
        exporter.export(path, prefix)
    elif "dreamwaq" in task_name:
        exporter = PolicyExporterWaQ(alg_runner.alg.actor_critic)
        exporter.export(path, prefix)
    else:
        export_policy_as_jit(alg_runner.alg.actor_critic, path, prefix)
    
    print('Exported policy as jit script to: ', path)
    
def override_configs(env_cfg, task_name):
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    if "cts" in task_name:  # cts specific
        env_cfg.env.num_teacher = 3
    env_cfg.viewer.rendered_envs_idx = list(range(env_cfg.env.num_envs))
    if env_cfg.terrain.mesh_type == "plane":
        for i in range(2):
            env_cfg.viewer.pos[i] = env_cfg.viewer.pos[i] - env_cfg.terrain.plane_length / 4
            env_cfg.viewer.lookat[i] = env_cfg.viewer.lookat[i] - env_cfg.terrain.plane_length / 4
    elif env_cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        env_cfg.terrain.num_rows = 2
        env_cfg.terrain.num_cols = 2
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = True
        
        # stairs
        env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.pyramid_stairs_terrain",
                                        "step_width": 0.31, "step_height": -0.1, "platform_size": 3.0}
        # single stair
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.pyramid_stairs_terrain",
        #                                   "step_width": 1.0, "step_height": -0.05, "platform_size": 3.0}
        # slope
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.pyramid_sloped_terrain",
        #                                   "slope": -0.4, "platform_size": 3.0}
        # # discrete obstacles
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.discrete_obstacles_terrain",
        #                                   "max_height": 0.1,
        #                                   "min_size": 1.0,
        #                                   "max_size": 2.0,
        #                                   "num_rects": 20,
        #                                   "platform_size": 3.0}
        env_cfg.env.debug_draw_height_points = True
    
    env_cfg.env.debug = True
    env_cfg.noise.add_noise = True

def interaction_loop(env, policy, task_name):
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 2 # which joint is used for logging
    stop_state_log = 300 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    
    # Get initial observations according to task type
    if "ts" in task_name or "cat" in task_name:  # teacher-student
        obs_buf, privileged_obs_buf, obs_history, critic_obs = env.get_observations()
    elif "ee" in task_name:  # explicit estimator
        estimator_features, _, _ = env.get_observations()
    elif "dreamwaq" in task_name:  # dreamwaq
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states = env.get_observations()
    else: # vanilla
        obs = env.get_observations()
        
    for i in range(10*int(env.max_episode_length)):
        
        if "ts" in task_name or "cat" in task_name:
            actions = policy(obs_buf, obs_history)
            obs_buf, privileged_obs_buf, obs_history, critic_obs, rews, dones, infos = env.step(actions.detach())
        elif "ee" in task_name:
            actions = policy(estimator_features.detach())
            estimator_features, estimator_labels, _, rews, dones, infos = env.step(actions.detach())
        elif "waq" in task_name:
            actions = policy(obs_buf, obs_history)
            obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states, rews, dones, infos = env.step(actions.detach())
        else:
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
        
        print_debug_info(env, robot_index)
        
        # Update logger info
        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.simulator.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.simulator.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.simulator.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.simulator.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.simulator.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.simulator.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.simulator.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.simulator.link_contact_forces[robot_index, 
                                                                          env.simulator.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

def print_debug_info(env, robot_index):
    # print debug info
    # print("base lin vel: ", env.base_lin_vel[robot_index, :].cpu().numpy())
    # print("base yaw angle: ", env.base_euler[robot_index, 2].item())
    # print("base height: ", env.base_pos[robot_index, 2].cpu().numpy())
    # print("foot_height: ", env.link_pos[robot_index, env.feet_indices, 2].cpu().numpy())
    pass
    

def play(args):
    if SIMULATOR == "genesis":
        gs.init(
            backend=gs.cpu if args.cpu else gs.gpu,
            logging_level='warning',
        )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    override_configs(env_cfg, args.task)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 
                            train_cfg.runner.load_run, 'exported')
        export_policy(ppo_runner, path, args.task)

    interaction_loop(env, policy, args.task)
    
if __name__ == '__main__':
    EXPORT_POLICY = True
    args = get_args()
    play(args)
