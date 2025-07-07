
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.math import get_scale_shift
import torch

class Go2Robot(LeggedRobot):
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
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # build privileged obs
        if self.cfg.env.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel, self.obs_buf),dim=-1)
            self.next_privileged_obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel, self.obs_buf),dim=-1)
            # self.privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)       # [4096, 0]
            # self.next_privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)  # [4096, 0]

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
                                                    (self.restitutions[:, 0] - restitutions_shift) * restitutions_scale),
                                                    dim=1)
                self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                        (self.restitutions[:, 0] - restitutions_shift) * restitutions_scale),
                                                        dim=1)
            if self.cfg.env.priv_observe_base_mass:
                payloads_scale, payloads_shift = get_scale_shift(self.cfg.normalization.added_mass_range)
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                    (self.payloads - payloads_shift) * payloads_scale),
                                                    dim=1)
                self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                        (self.payloads - payloads_shift) * payloads_scale),
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