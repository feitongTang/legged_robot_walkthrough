
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from legged_gym.utils.math import *

import numpy as np

class Go2Robot(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._init_command_distribution(torch.arange(self.num_envs, device=self.device))
    
    def post_physics_step(self):
        super().post_physics_step()
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                          )[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                              0:3]
        
        self.last_last_actions[:] = self.last_actions[:]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]

    def check_termination(self):
        super().check_termination()
        if self.cfg.rewards.use_terminal_body_height:
            self.body_height_buf = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1) \
                                   < self.cfg.rewards.terminal_body_height
            self.reset_buf = torch.logical_or(self.body_height_buf, self.reset_buf)

    # def reset_idx(self, env_ids):
    #     if len(env_ids) == 0:
    #         return
        
    #     # reset robot states
    #     self._resample_commands(env_ids)
    #     self._randomize_dof_props(env_ids)
    #     if self.cfg.domain_rand.randomize_rigids_after_start:
    #         self._randomize_rigid_body_props(env_ids)
    #         self.refresh_actor_rigid_shape_props(env_ids)

    #     self._reset_dofs(env_ids)
    #     self._reset_root_states(env_ids)

    #     # reset buffers
    #     self.last_actions[env_ids] = 0.
    #     self.last_last_actions[env_ids] = 0.
    #     self.last_dof_vel[env_ids] = 0.
    #     self.feet_air_time[env_ids] = 0.
    #     self.episode_length_buf[env_ids] = 0
    #     self.reset_buf[env_ids] = 1

    #     self.extras["episode"] = {}
    #     for key in self.episode_sums.keys():
    #         self.extras["episode"]['rew_' + key] = torch.mean(
    #             self.episode_sums[key][env_ids])
    #         self.episode_sums[key][env_ids] = 0.

    #     if self.cfg.terrain.curriculum:
    #         self.extras["episode"]["terrain_level"] = torch.mean(
    #             self.terrain_levels[:self.num_envs].float())
    #     if self.cfg.commands.command_curriculum:
    #         self.extras["env_bins"] = torch.Tensor(self.env_command_bins)[:self.num_envs]
    #         self.extras["episode"]["min_command_duration"] = torch.min(self.commands[:, 8])
    #         self.extras["episode"]["max_command_duration"] = torch.max(self.commands[:, 8])
    #         self.extras["episode"]["min_command_bound"] = torch.min(self.commands[:, 7])
    #         self.extras["episode"]["max_command_bound"] = torch.max(self.commands[:, 7])
    #         self.extras["episode"]["min_command_offset"] = torch.min(self.commands[:, 6])
    #         self.extras["episode"]["max_command_offset"] = torch.max(self.commands[:, 6])
    #         self.extras["episode"]["min_command_phase"] = torch.min(self.commands[:, 5])
    #         self.extras["episode"]["max_command_phase"] = torch.max(self.commands[:, 5])
    #         self.extras["episode"]["min_command_freq"] = torch.min(self.commands[:, 4])
    #         self.extras["episode"]["max_command_freq"] = torch.max(self.commands[:, 4])
    #         self.extras["episode"]["min_command_x_vel"] = torch.min(self.commands[:, 0])
    #         self.extras["episode"]["max_command_x_vel"] = torch.max(self.commands[:, 0])
    #         self.extras["episode"]["min_command_y_vel"] = torch.min(self.commands[:, 1])
    #         self.extras["episode"]["max_command_y_vel"] = torch.max(self.commands[:, 1])
    #         self.extras["episode"]["min_command_yaw_vel"] = torch.min(self.commands[:, 2])
    #         self.extras["episode"]["max_command_yaw_vel"] = torch.max(self.commands[:, 2])
    #         if self.cfg.commands.num_commands > 9:
    #             self.extras["episode"]["min_command_swing_height"] = torch.min(self.commands[:, 9])
    #             self.extras["episode"]["max_command_swing_height"] = torch.max(self.commands[:, 9])
    #         for curriculum, category in zip(self.curricula, self.category_names):
    #             self.extras["episode"][f"command_area_{category}"] = np.sum(curriculum.weights) / \
    #                                                                        curriculum.weights.shape[0]

    #         self.extras["episode"]["min_action"] = torch.min(self.actions)
    #         self.extras["episode"]["max_action"] = torch.max(self.actions)

    #         self.extras["curriculum/distribution"] = {}
    #         for curriculum, category in zip(self.curricula, self.category_names):
    #             self.extras[f"curriculum/distribution"][f"weights_{category}"] = curriculum.weights
    #             self.extras[f"curriculum/distribution"][f"grid_{category}"] = curriculum.grid
    #     if self.cfg.env.send_timeouts:
    #         self.extras["time_outs"] = self.time_out_buf[:self.num_envs]

    #     self.gait_indices[env_ids] = 0

    #     for i in range(len(self.lag_buffer)):
    #         self.lag_buffer[i][env_ids, :] = 0

    def compute_reward(self):
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            if torch.sum(rew) >= 0:
                self.rew_buf_pos += rew
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg += rew
            self.episode_sums[name] += rew
            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                self.command_sums[name] += self.reward_scales[name] + rew
            else:
                self.command_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        self.episode_sums["total"] += self.rew_buf
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self.reward_container._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            self.command_sums["termination"] += rew

        self.command_sums["lin_vel_raw"] += self.base_lin_vel[:, 0]
        self.command_sums["ang_vel_raw"] += self.base_ang_vel[:, 2]
        self.command_sums["lin_vel_residual"] += (self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2
        self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands[:, 2]) ** 2
        self.command_sums["ep_timesteps"] += 1

    def compute_observations(self):
        # """ Computes observations
        # """
        # self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
        #                             self.projected_gravity,
        #                             self.commands * self.commands_scale,
        #                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
        #                             self.dof_vel * self.obs_scales.dof_vel,
        #                             self.actions
        #                             ),dim=-1)
        # # add perceptive inputs if not blind
        # # add noise if needed
        # if self.add_noise:
        #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        """ Computes observations
        """
        self.obs_buf = torch.cat((self.projected_gravity,
                                  (self.dof_pos[:, :self.num_actions] - self.default_dof_pos[:,
                                                                             :self.num_actions]) * self.obs_scales.dof_pos,
                                  self.dof_vel[:, :self.num_actions] * self.obs_scales.dof_vel,
                                  self.actions
                                  ), dim=-1)

        if self.cfg.env.observe_command:
            self.obs_buf = torch.cat((self.projected_gravity,
                                      self.commands * self.commands_scale,
                                      (self.dof_pos[:, :self.num_actions] - self.default_dof_pos[:,
                                                                                 :self.num_actions]) * self.obs_scales.dof_pos,
                                      self.dof_vel[:, :self.num_actions] * self.obs_scales.dof_vel,
                                      self.actions
                                      ), dim=-1)

        if self.cfg.env.observe_two_prev_actions:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.last_actions), dim=-1)

        if self.cfg.env.observe_timing_parameter:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.gait_indices.unsqueeze(1)), dim=-1)

        if self.cfg.env.observe_clock_inputs:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.clock_inputs), dim=-1)

        if self.cfg.env.observe_vel:
            if self.cfg.commands.global_reference:
                self.obs_buf = torch.cat((self.root_states[:self.num_envs, 7:10] * self.obs_scales.lin_vel,
                                          self.base_ang_vel * self.obs_scales.ang_vel,
                                          self.obs_buf), dim=-1)
            else:
                self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                          self.base_ang_vel * self.obs_scales.ang_vel,
                                          self.obs_buf), dim=-1)

        if self.cfg.env.observe_only_ang_vel:
            self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                      self.obs_buf), dim=-1)

        if self.cfg.env.observe_only_lin_vel:
            self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                      self.obs_buf), dim=-1)

        if self.cfg.env.observe_yaw:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
            # heading_error = torch.clip(0.5 * wrap_to_pi(heading), -1., 1.).unsqueeze(1)
            self.obs_buf = torch.cat((self.obs_buf,
                                      heading), dim=-1)

        if self.cfg.env.observe_contact_states:
            self.obs_buf = torch.cat((self.obs_buf, (self.contact_forces[:, self.feet_indices, 2] > 1.).view(
                self.num_envs,
                -1) * 1.0), dim=1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # build privileged obs

        self.privileged_obs_buf = self.obs_buf.clone().to(self.device)
        self.next_privileged_obs_buf = self.privileged_obs_buf.clone().to(self.device)

        if self.cfg.env.priv_observe_friction:
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.normalization.friction_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.friction_coeffs[:, 0].unsqueeze(
                                                     1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.friction_coeffs[:, 0].unsqueeze(
                                                          1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_ground_friction:
            self.ground_friction_coeffs = self._get_ground_frictions(range(self.num_envs))
            ground_friction_coeffs_scale, ground_friction_coeffs_shift = get_scale_shift(
                self.cfg.normalization.ground_friction_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.ground_friction_coeffs.unsqueeze(
                                                     1) - ground_friction_coeffs_shift) * ground_friction_coeffs_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.ground_friction_coeffs.unsqueeze(
                                                          1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_restitution:
            restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.normalization.restitution_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.restitutions[:, 0].unsqueeze(
                                                     1) - restitutions_shift) * restitutions_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.restitutions[:, 0].unsqueeze(
                                                          1) - restitutions_shift) * restitutions_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_base_mass:
            payloads_scale, payloads_shift = get_scale_shift(self.cfg.normalization.added_mass_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),
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

        assert self.privileged_obs_buf.shape[
                   1] == self.cfg.env.num_privileged_obs, f"num_privileged_obs ({self.cfg.env.num_privileged_obs}) != the number of privileged observations ({self.privileged_obs_buf.shape[1]}), you will discard data from the student!"

    def _create_envs(self):
        super()._create_envs()
        self._init_custom_buffers__()
        self._randomize_rigid_body_props()
        self._randomize_gravity()

    def refresh_actor_rigid_shape_props(self, env_ids):
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(self.num_dof):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitutions[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _randomize_dof_props(self, env_ids):
        if self.cfg.domain_rand.randomize_motor_strength:
            min_strength, max_strength = self.cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_strength - min_strength) + min_strength
        if self.cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = self.cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids, :] = torch.rand(len(env_ids), self.num_dof, dtype=torch.float,
                                                        device=self.device, requires_grad=False) * (
                                                     max_offset - min_offset) + min_offset
        if self.cfg.domain_rand.randomize_Kp_factor:
            min_Kp_factor, max_Kp_factor = self.cfg.domain_rand.Kp_factor_range
            self.Kp_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kp_factor - min_Kp_factor) + min_Kp_factor
        if self.cfg.domain_rand.randomize_Kd_factor:
            min_Kd_factor, max_Kd_factor = self.cfg.domain_rand.Kd_factor_range
            self.Kd_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kd_factor - min_Kd_factor) + min_Kd_factor

    def _randomize_gravity(self, external_force = None):

        if external_force is not None:
            self.gravities[:, :] = external_force.unsqueeze(0)
        elif self.cfg.domain_rand.randomize_gravity:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                        requires_grad=False) * (max_gravity - min_gravity) + min_gravity

            self.gravities[:, :] = external_force.unsqueeze(0)

        sim_params = self.gym.get_sim_params(self.sim)
        gravity = self.gravities[0, :] + torch.Tensor([0, 0, -9.8]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)

    def _randomize_rigid_body_props(self):
        if self.cfg.domain_rand.randomize_base_mass:
            min_payload, max_payload = self.cfg.domain_rand.added_mass_range
            # self.payloads[env_ids] = -1.0
            self.payloads[:] = torch.rand(self.num_envs, dtype=torch.float, device=self.device,
                                                requires_grad=False) * (max_payload - min_payload) + min_payload
        if self.cfg.domain_rand.randomize_com_displacement:
            min_com_displacement, max_com_displacement = self.cfg.domain_rand.com_displacement_range
            self.com_displacements[:, :] = torch.rand(self.num_envs, 3, dtype=torch.float, device=self.device,
                                                            requires_grad=False) * (
                                                         max_com_displacement - min_com_displacement) + min_com_displacement

        if self.cfg.domain_rand.randomize_friction:
            min_friction, max_friction = self.cfg.domain_rand.friction_range
            self.friction_coeffs[:, :] = torch.rand(self.num_envs, 1, dtype=torch.float, device=self.device,
                                                          requires_grad=False) * (
                                                       max_friction - min_friction) + min_friction

        if self.cfg.domain_rand.randomize_restitution:
            min_restitution, max_restitution = self.cfg.domain_rand.restitution_range
            self.restitutions[:] = torch.rand(self.num_envs, 1, dtype=torch.float, device=self.device,
                                                    requires_grad=False) * (
                                                 max_restitution - min_restitution) + min_restitution

    def _init_buffers(self):
        super()._init_buffers()
        self.net_contact_forces = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))[:self.num_envs * self.num_bodies, :]
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,
                               self.feet_indices,
                               7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                              0:3]
        self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg.domain_rand.lag_timesteps+1)]
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )

    def _init_custom_buffers__(self):
        # domain randomization properties
        self.friction_coeffs = self.default_friction * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.restitutions = self.default_restitution * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
        self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                 requires_grad=False)

    def _init_command_distribution(self, env_ids):
        # new style curriculum
        self.category_names = ['nominal']
        if self.cfg.commands.gaitwise_curricula:
            self.category_names = ['pronk', 'trot', 'pace', 'bound']

        if self.cfg.commands.curriculum_type == "RewardThresholdCurriculum":
            from ..base.curriculum import RewardThresholdCurriculum
            CurriculumClass = RewardThresholdCurriculum
        self.curricula = []
        for category in self.category_names:
            self.curricula += [CurriculumClass(seed=self.cfg.commands.curriculum_seed,
                                               x_vel=(self.cfg.commands.ranges.limit_vel_x[0],
                                                      self.cfg.commands.ranges.limit_vel_x[1],
                                                      self.cfg.commands.num_bins_vel_x),
                                               y_vel=(self.cfg.commands.ranges.limit_vel_y[0],
                                                      self.cfg.commands.ranges.limit_vel_y[1],
                                                      self.cfg.commands.num_bins_vel_y),
                                               yaw_vel=(self.cfg.commands.ranges.limit_vel_yaw[0],
                                                        self.cfg.commands.ranges.limit_vel_yaw[1],
                                                        self.cfg.commands.num_bins_vel_yaw),
                                               body_height=(self.cfg.commands.ranges.limit_body_height[0],
                                                            self.cfg.commands.ranges.limit_body_height[1],
                                                            self.cfg.commands.num_bins_body_height),
                                               gait_frequency=(self.cfg.commands.ranges.limit_gait_frequency[0],
                                                               self.cfg.commands.ranges.limit_gait_frequency[1],
                                                               self.cfg.commands.num_bins_gait_frequency),
                                               gait_phase=(self.cfg.commands.ranges.limit_gait_phase[0],
                                                           self.cfg.commands.ranges.limit_gait_phase[1],
                                                           self.cfg.commands.num_bins_gait_phase),
                                               gait_offset=(self.cfg.commands.ranges.limit_gait_offset[0],
                                                            self.cfg.commands.ranges.limit_gait_offset[1],
                                                            self.cfg.commands.num_bins_gait_offset),
                                               gait_bounds=(self.cfg.commands.ranges.limit_gait_bound[0],
                                                            self.cfg.commands.ranges.limit_gait_bound[1],
                                                            self.cfg.commands.num_bins_gait_bound),
                                               gait_duration=(self.cfg.commands.ranges.limit_gait_duration[0],
                                                              self.cfg.commands.ranges.limit_gait_duration[1],
                                                              self.cfg.commands.num_bins_gait_duration),
                                               footswing_height=(self.cfg.commands.ranges.limit_footswing_height[0],
                                                                 self.cfg.commands.ranges.limit_footswing_height[1],
                                                                 self.cfg.commands.num_bins_footswing_height),
                                               body_pitch=(self.cfg.commands.ranges.limit_body_pitch[0],
                                                           self.cfg.commands.ranges.limit_body_pitch[1],
                                                           self.cfg.commands.num_bins_body_pitch),
                                               body_roll=(self.cfg.commands.ranges.limit_body_roll[0],
                                                          self.cfg.commands.ranges.limit_body_roll[1],
                                                          self.cfg.commands.num_bins_body_roll),
                                               stance_width=(self.cfg.commands.ranges.limit_stance_width[0],
                                                             self.cfg.commands.ranges.limit_stance_width[1],
                                                             self.cfg.commands.num_bins_stance_width),
                                               stance_length=(self.cfg.commands.ranges.limit_stance_length[0],
                                                                self.cfg.commands.ranges.limit_stance_length[1],
                                                                self.cfg.commands.num_bins_stance_length),
                                               aux_reward_coef=(self.cfg.commands.ranges.limit_aux_reward_coef[0],
                                                           self.cfg.commands.ranges.limit_aux_reward_coef[1],
                                                           self.cfg.commands.num_bins_aux_reward_coef),
                                               )]

        self.env_command_bins = np.zeros(len(env_ids), dtype=np.int32)
        self.env_command_categories = np.zeros(len(env_ids), dtype=np.int32)
        low = np.array(
            [self.cfg.commands.ranges.lin_vel_x[0], self.cfg.commands.ranges.lin_vel_y[0],
             self.cfg.commands.ranges.ang_vel_yaw[0], self.cfg.commands.ranges.body_height_cmd[0],
             self.cfg.commands.ranges.gait_frequency_cmd_range[0],
             self.cfg.commands.ranges.gait_phase_cmd_range[0], self.cfg.commands.ranges.gait_offset_cmd_range[0],
             self.cfg.commands.ranges.gait_bound_cmd_range[0], self.cfg.commands.ranges.gait_duration_cmd_range[0],
             self.cfg.commands.ranges.footswing_height_range[0], self.cfg.commands.ranges.body_pitch_range[0],
             self.cfg.commands.ranges.body_roll_range[0],self.cfg.commands.ranges.stance_width_range[0],
             self.cfg.commands.ranges.stance_length_range[0], self.cfg.commands.ranges.aux_reward_coef_range[0], ])
        high = np.array(
            [self.cfg.commands.ranges.lin_vel_x[1], self.cfg.commands.ranges.lin_vel_y[1],
             self.cfg.commands.ranges.ang_vel_yaw[1], self.cfg.commands.ranges.body_height_cmd[1],
             self.cfg.commands.ranges.gait_frequency_cmd_range[1],
             self.cfg.commands.ranges.gait_phase_cmd_range[1], self.cfg.commands.ranges.gait_offset_cmd_range[1],
             self.cfg.commands.ranges.gait_bound_cmd_range[1], self.cfg.commands.ranges.gait_duration_cmd_range[1],
             self.cfg.commands.ranges.footswing_height_range[1], self.cfg.commands.ranges.body_pitch_range[1],
             self.cfg.commands.ranges.body_roll_range[1],self.cfg.commands.ranges.stance_width_range[1],
             self.cfg.commands.ranges.stance_length_range[1], self.cfg.commands.ranges.aux_reward_coef_range[1], ])
        for curriculum in self.curricula:
            curriculum.set_to(low=low, high=high)
    
    # ---reward---
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # reward containers
        from legged_gym.envs.rewards.corl_rewards import CoRLRewards
        reward_containers = {"CoRLRewards": CoRLRewards}
        self.reward_container = reward_containers[self.cfg.rewards.reward_container_name](self)

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue

            if not hasattr(self.reward_container, '_reward_' + name):
                print(f"Warning: reward {'_reward_' + name} has nonzero coefficient but was not found!")
            else:
                self.reward_names.append(name)
                self.reward_functions.append(getattr(self.reward_container, '_reward_' + name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.episode_sums_eval = {
            name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums_eval["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                      requires_grad=False)
        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in
            list(self.reward_scales.keys()) + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual",
                                               "ep_timesteps"]}