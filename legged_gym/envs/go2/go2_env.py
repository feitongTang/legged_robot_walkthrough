
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class Go2Robot(LeggedRobot):
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

        # if custom initialization values were passed in, set them here
        dynamics_params = ["friction_coeffs", "restitutions", "payloads", "com_displacements", "motor_strengths",
                           "Kp_factors", "Kd_factors"]
        if self.initial_dynamics_dict is not None:
            for k, v in self.initial_dynamics_dict.items():
                if k in dynamics_params:
                    setattr(self, k, v.to(self.device))

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
        self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                 requires_grad=False)

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