# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories

class RolloutStorageCTS:
    class Transition:
        def __init__(self):
            self.observations = None
            self.privileged_observations = None
            self.observation_histories = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
        
        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_teacher, num_transitions_per_env, obs_shape, 
                 privileged_obs_shape, obs_history_shape, critic_obs_shape, actions_shape, device='cpu'):

        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.obs_history_shape = obs_history_shape
        self.critic_obs_shape = critic_obs_shape
        self.actions_shape = actions_shape
        self.num_teacher = num_teacher
        
        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        # privileged observations are necessary
        self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        self.observation_histories = torch.zeros(num_transitions_per_env, num_envs, *obs_history_shape, device=self.device)
        self.critic_observations = torch.zeros(num_transitions_per_env, num_envs, *critic_obs_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.observation_histories[self.step].copy_(transition.observation_histories)
        self.critic_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states==(None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed 
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
            self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])


    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # Use constant teacher environment indices
        teacher_mini_batch_size = self.num_teacher * self.num_transitions_per_env // num_mini_batches
        student_mini_batch_size = (self.num_envs - self.num_teacher) * self.num_transitions_per_env // num_mini_batches
        teacher_indices = torch.randperm(num_mini_batches*teacher_mini_batch_size, requires_grad=False, device=self.device)
        student_indices = torch.randperm(num_mini_batches*student_mini_batch_size, requires_grad=False, device=self.device)
        total_mini_batch_size = teacher_mini_batch_size + student_mini_batch_size
        total_indices = torch.randperm(num_mini_batches*total_mini_batch_size, requires_grad=False, device=self.device)
        
        # Split data into teacher group and student group
        teacher_observations = self.observations[:, 0:self.num_teacher].flatten(0, 1)
        teacher_privileged_observations = self.privileged_observations[:, 0:self.num_teacher].flatten(0, 1)
        # teacher_obs_histories = self.observation_histories[:, self.teacher_env_ids].flatten(0, 1)
        teacher_actions = self.actions[:, 0:self.num_teacher].flatten(0, 1)
        teacher_old_actions_log_prob = self.actions_log_prob[:, 0:self.num_teacher].flatten(0, 1)
        teacher_advantages = self.advantages[:, 0:self.num_teacher].flatten(0, 1)

        student_observations = self.observations[:, self.num_teacher:].flatten(0, 1)
        student_privileged_observations = self.privileged_observations[:, self.num_teacher:].flatten(0, 1)
        student_obs_histories = self.observation_histories[:, self.num_teacher:].flatten(0, 1)
        student_actions = self.actions[:, self.num_teacher:].flatten(0, 1)
        student_old_actions_log_prob = self.actions_log_prob[:, self.num_teacher:].flatten(0, 1)
        student_advantages = self.advantages[:, self.num_teacher:].flatten(0, 1)
        student_old_mu = self.mu[:, self.num_teacher:].flatten(0, 1)
        student_old_sigma = self.sigma[:, self.num_teacher:].flatten(0, 1)
        # Values do not need to split
        critic_observations = self.critic_observations.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                
                # Generate batch indices
                start_teacher = i*teacher_mini_batch_size
                end_teacher = (i+1)*teacher_mini_batch_size
                teacher_batch_idx = teacher_indices[start_teacher:end_teacher]
                start_student = i*student_mini_batch_size
                end_student = (i+1)*student_mini_batch_size
                student_batch_idx = student_indices[start_student:end_student]
                start_total = i*total_mini_batch_size
                end_total = (i+1)*total_mini_batch_size
                total_batch_idx = total_indices[start_total:end_total]
                
                teacher_obs_batch = teacher_observations[teacher_batch_idx]
                teacher_privileged_obs_batch = teacher_privileged_observations[teacher_batch_idx]
                teacher_actions_batch = teacher_actions[teacher_batch_idx]
                teacher_old_actions_log_prob_batch = teacher_old_actions_log_prob[teacher_batch_idx]
                teacher_advantages_batch = teacher_advantages[teacher_batch_idx]
                
                student_obs_batch = student_observations[student_batch_idx]
                student_privileged_obs_batch = student_privileged_observations[student_batch_idx]
                student_obs_histories_batch = student_obs_histories[student_batch_idx]
                student_actions_batch = student_actions[student_batch_idx]
                student_old_actions_log_prob_batch = student_old_actions_log_prob[student_batch_idx]
                student_advantages_batch = student_advantages[student_batch_idx]
                student_old_mu_batch = student_old_mu[student_batch_idx]
                student_old_sigma_batch = student_old_sigma[student_batch_idx]
                
                critic_obs_batch = critic_observations[total_batch_idx]
                values_batch = values[total_batch_idx]
                returns_batch = returns[total_batch_idx]

                yield teacher_obs_batch, teacher_privileged_obs_batch, teacher_actions_batch, \
                    teacher_old_actions_log_prob_batch, teacher_advantages_batch, \
                    student_obs_batch, student_privileged_obs_batch, student_obs_histories_batch, student_actions_batch, \
                    student_old_actions_log_prob_batch, student_advantages_batch, student_old_mu_batch, student_old_sigma_batch, \
                    critic_obs_batch, values_batch, returns_batch, \
                        (None, None), None

    # for RNNs only
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):

        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None: 
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else: 
            padded_critic_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i*mini_batch_size
                stop = (i+1)*mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size
                
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                                for saved_hidden_states in self.saved_hidden_states_a ] 
                hid_c_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                                for saved_hidden_states in self.saved_hidden_states_c ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch)==1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch)==1 else hid_a_batch

                yield obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (hid_a_batch, hid_c_batch), masks_batch
                
                first_traj = last_traj