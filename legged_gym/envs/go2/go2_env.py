
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

import numpy as np

class Go2Robot(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._init_command_distribution(torch.arange(self.num_envs, device=self.device))
    
    def _create_envs(self):
        super()._create_envs()
        self._init_custom_buffers__()
        self._randomize_rigid_body_props()
        self._randomize_gravity()

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
        self.prev_base_pos = self.base_pos.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()
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

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
    
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