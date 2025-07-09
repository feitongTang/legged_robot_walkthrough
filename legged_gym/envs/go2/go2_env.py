
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.math import get_scale_shift
import torch
import numpy as np

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Go2Robot(LeggedRobot):
    def _resample_commands(self, env_ids):

        if len(env_ids) == 0: return

        timesteps = int(self.cfg.commands.resampling_time / self.dt)
        ep_len = min(self.max_episode_length, timesteps)

        # update curricula based on terminated environment bins and categories
        for i, (category, curriculum) in enumerate(zip(self.category_names, self.curricula)):
            env_ids_in_category = self.env_command_categories[env_ids.cpu()] == i
            if isinstance(env_ids_in_category, np.bool_) or len(env_ids_in_category) == 1:
                env_ids_in_category = torch.tensor([env_ids_in_category], dtype=torch.bool)
            elif len(env_ids_in_category) == 0:
                continue

            env_ids_in_category = env_ids[env_ids_in_category]

            task_rewards, success_thresholds = [], []
            for key in ["tracking_lin_vel", "tracking_ang_vel", "tracking_contacts_shaped_force",
                        "tracking_contacts_shaped_vel"]:
                if key in self.command_sums.keys():
                    task_rewards.append(self.command_sums[key][env_ids_in_category] / ep_len)
                    success_thresholds.append(self.curriculum_thresholds[key] * self.reward_scales[key])

            old_bins = self.env_command_bins[env_ids_in_category.cpu().numpy()]
            if len(success_thresholds) > 0:
                curriculum.update(old_bins, task_rewards, success_thresholds,
                                  local_range=np.array(
                                      [0.55, 0.55, 0.55, 0.55, 0.35, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0,
                                       1.0]))

        # assign resampled environments to new categories
        random_env_floats = torch.rand(len(env_ids), device=self.device)
        probability_per_category = 1. / len(self.category_names)
        category_env_ids = [env_ids[torch.logical_and(probability_per_category * i <= random_env_floats,
                                                      random_env_floats < probability_per_category * (i + 1))] for i in
                            range(len(self.category_names))]

        # sample from new category curricula
        for i, (category, env_ids_in_category, curriculum) in enumerate(
                zip(self.category_names, category_env_ids, self.curricula)):

            batch_size = len(env_ids_in_category)
            if batch_size == 0: continue

            new_commands, new_bin_inds = curriculum.sample(batch_size=batch_size)

            self.env_command_bins[env_ids_in_category.cpu().numpy()] = new_bin_inds
            self.env_command_categories[env_ids_in_category.cpu().numpy()] = i

            self.commands[env_ids_in_category, :] = torch.Tensor(new_commands[:, :self.cfg.commands.num_commands]).to(
                self.device)

        if self.cfg.commands.num_commands > 5:
            if self.cfg.commands.gaitwise_curricula:
                for i, (category, env_ids_in_category) in enumerate(zip(self.category_names, category_env_ids)):
                    if category == "pronk":  # pronking
                        self.commands[env_ids_in_category, 5] = (self.commands[env_ids_in_category, 5] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 6] = (self.commands[env_ids_in_category, 6] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 7] = (self.commands[env_ids_in_category, 7] / 2 - 0.25) % 1
                    elif category == "trot":  # trotting
                        self.commands[env_ids_in_category, 5] = self.commands[env_ids_in_category, 5] / 2 + 0.25
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "pace":  # pacing
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = self.commands[env_ids_in_category, 6] / 2 + 0.25
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "bound":  # bounding
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = self.commands[env_ids_in_category, 7] / 2 + 0.25

            elif self.cfg.commands.exclusive_phase_offset:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                trotting_envs = env_ids[random_env_floats < 0.34]
                pacing_envs = env_ids[torch.logical_and(0.34 <= random_env_floats, random_env_floats < 0.67)]
                bounding_envs = env_ids[0.67 <= random_env_floats]
                self.commands[pacing_envs, 5] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[trotting_envs, 6] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 7] = 0

            elif self.cfg.commands.balance_gait_distribution:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                pronking_envs = env_ids[random_env_floats <= 0.25]
                trotting_envs = env_ids[torch.logical_and(0.25 <= random_env_floats, random_env_floats < 0.50)]
                pacing_envs = env_ids[torch.logical_and(0.50 <= random_env_floats, random_env_floats < 0.75)]
                bounding_envs = env_ids[0.75 <= random_env_floats]
                self.commands[pronking_envs, 5] = (self.commands[pronking_envs, 5] / 2 - 0.25) % 1
                self.commands[pronking_envs, 6] = (self.commands[pronking_envs, 6] / 2 - 0.25) % 1
                self.commands[pronking_envs, 7] = (self.commands[pronking_envs, 7] / 2 - 0.25) % 1
                self.commands[trotting_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 5] = 0
                self.commands[pacing_envs, 7] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 5] = self.commands[trotting_envs, 5] / 2 + 0.25
                self.commands[pacing_envs, 6] = self.commands[pacing_envs, 6] / 2 + 0.25
                self.commands[bounding_envs, 7] = self.commands[bounding_envs, 7] / 2 + 0.25

            if self.cfg.commands.binary_phases:
                self.commands[env_ids, 5] = (torch.round(2 * self.commands[env_ids, 5])) / 2.0 % 1
                self.commands[env_ids, 6] = (torch.round(2 * self.commands[env_ids, 6])) / 2.0 % 1
                self.commands[env_ids, 7] = (torch.round(2 * self.commands[env_ids, 7])) / 2.0 % 1

        # setting the smaller commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # reset command sums
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        
        if self.cfg.env.observe_two_prev_actions:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.last_actions), dim=-1)
            
        if self.cfg.env.observe_timing_parameter:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.gait_indices.unsqueeze(1)), dim=-1)
            
        if self.cfg.env.observe_clock_inputs:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.clock_inputs), dim=-1)
            
        if self.cfg.env.observe_yaw:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
            self.obs_buf = torch.cat((self.obs_buf,
                                      heading), dim=-1)
        
        if self.cfg.env.observe_contact_states:
            self.obs_buf = torch.cat((self.obs_buf, (self.contact_forces[:, self.feet_indices, 2] > 1.).view(
                self.num_envs,
                -1) * 1.0), dim=1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # build privileged obs
        if self.cfg.env.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel, self.obs_buf),dim=-1)
            self.next_privileged_obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel, self.obs_buf),dim=-1)

            if self.cfg.env.priv_observe_friction:
                self.friction_coeffs = self.friction_coeffs.to(self.device)
                friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.normalization.friction_range)
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                    (self.friction_coeffs[:, 0] - friction_coeffs_shift) * friction_coeffs_scale),
                                                    dim=1)
                self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                        (self.friction_coeffs[:, 0] - friction_coeffs_shift) * friction_coeffs_scale),
                                                        dim=1)
            if self.cfg.env.priv_observe_ground_friction:
                self.ground_friction_coeffs = self._get_ground_frictions(range(self.num_envs))
                ground_friction_coeffs_scale, ground_friction_coeffs_shift = get_scale_shift(
                    self.cfg.normalization.ground_friction_range)
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                    (self.ground_friction_coeffs - ground_friction_coeffs_shift) * ground_friction_coeffs_scale),
                                                    dim=1)
                self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                        (self.ground_friction_coeffs - friction_coeffs_shift) * friction_coeffs_scale),
                                                        dim=1)
            if self.cfg.env.priv_observe_restitution:
                restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.normalization.restitution_range)
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                    (self.restitutions[:, 0].unsqueeze(1) - restitutions_shift) * restitutions_scale),
                                                    dim=1)
                self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                        (self.restitutions[:, 0].unsqueeze(1) - restitutions_shift) * restitutions_scale),
                                                        dim=1)
            if self.cfg.env.priv_observe_com_displacement:
                com_displacements_scale, com_displacements_shift = get_scale_shift(
                    self.cfg.normalization.com_displacement_range)
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                    (
                                                            self.com_displacements - com_displacements_shift) * com_displacements_scale),
                                                    dim=1)
                self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                        (
                                                                self.com_displacements - com_displacements_shift) * com_displacements_scale),
                                                        dim=1)
            if self.cfg.env.priv_observe_motor_strength:
                motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.normalization.motor_strength_range)
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                    (
                                                            self.motor_strengths - motor_strengths_shift) * motor_strengths_scale),
                                                    dim=1)
                self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                        (
                                                                self.motor_strengths - motor_strengths_shift) * motor_strengths_scale),
                                                        dim=1)
            if self.cfg.env.priv_observe_motor_offset:
                motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.normalization.motor_offset_range)
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                    (
                                                            self.motor_offsets - motor_offset_shift) * motor_offset_scale),
                                                    dim=1)
                self.next_privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                        (
                                                                self.motor_offsets - motor_offset_shift) * motor_offset_scale),
                                                        dim=1)
            if self.cfg.env.priv_observe_body_height:
                body_height_scale, body_height_shift = get_scale_shift(self.cfg.normalization.body_height_range)
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                    ((self.root_states[:self.num_envs, 2]).view(
                                                        self.num_envs, -1) - body_height_shift) * body_height_scale),
                                                    dim=1)
                self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                        ((self.root_states[:self.num_envs, 2]).view(
                                                            self.num_envs, -1) - body_height_shift) * body_height_scale),
                                                        dim=1)
            if self.cfg.env.priv_observe_body_velocity:
                body_velocity_scale, body_velocity_shift = get_scale_shift(self.cfg.normalization.body_velocity_range)
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                    ((self.base_lin_vel).view(self.num_envs,
                                                                            -1) - body_velocity_shift) * body_velocity_scale),
                                                    dim=1)
                self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                        ((self.base_lin_vel).view(self.num_envs,
                                                                                    -1) - body_velocity_shift) * body_velocity_scale),
                                                        dim=1)
            if self.cfg.env.priv_observe_gravity:
                gravity_scale, gravity_shift = get_scale_shift(self.cfg.normalization.gravity_range)
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                    (self.gravities - gravity_shift) / gravity_scale),
                                                    dim=1)
                self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                        (self.gravities - gravity_shift) / gravity_scale), dim=1)

            if self.cfg.env.priv_observe_clock_inputs:
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                    self.clock_inputs), dim=-1)

            if self.cfg.env.priv_observe_desired_contact_states:
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                    self.desired_contact_states), dim=-1)

            # assert self.privileged_obs_buf.shape[
            #         1] == self.cfg.env.num_privileged_obs, f"num_privileged_obs ({self.cfg.env.num_privileged_obs}) != the number of privileged observations ({self.privileged_obs_buf.shape[1]}), you will discard data from the student!"

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """

        noise_vec = torch.zeros(self.num_obs)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions

        if self.cfg.env.observe_two_prev_actions:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(self.num_actions)
                                   ), dim=0)
        
        if self.cfg.env.observe_timing_parameter:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(1)
                                   ), dim=0)
            
        if self.cfg.env.observe_clock_inputs:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(4)
                                   ), dim=0)
            
        if self.cfg.env.observe_yaw:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(1),
                                   ), dim=0)
            
        if self.cfg.env.observe_contact_states:
            noise_vec = torch.cat((noise_vec,
                                   torch.ones(4) * noise_scales.contact_states * noise_level,
                                   ), dim=0)
            
        noise_vec = noise_vec.to(self.device)

        return noise_vec