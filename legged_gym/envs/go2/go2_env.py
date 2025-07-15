
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class Go2Robot(LeggedRobot):
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