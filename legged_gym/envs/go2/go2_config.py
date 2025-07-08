from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2RoughCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_observations = 46
        num_privileged_obs = 51
        num_actions = 12

        observe_vel = True
        observe_yaw = False
        observe_contact_states = False
        observe_command = True
        observe_height_command = False
        observe_gait_commands = False
        observe_timing_parameter = False
        observe_clock_inputs = False
        observe_two_prev_actions = False
        observe_imu = False

        privileged_future_horizon = 1
        priv_observe_friction = True
        priv_observe_ground_friction = False
        priv_observe_restitution = True
        priv_observe_com_displacement = False   # True
        priv_observe_motor_strength = False
        priv_observe_motor_offset = False
        priv_observe_body_velocity = False
        priv_observe_body_height = False
        priv_observe_gravity = False
        priv_observe_clock_inputs = False
        priv_observe_desired_contact_states = False

    class commands( LeggedRobotCfg.commands ):
        gaitwise_curricula = True   # 启用步态相关的课程学习
        curriculum_type = "RewardThresholdCurriculum"
        curriculum_seed = 207
        global_reference = False
        class ranges( LeggedRobotCfg.commands.ranges ):
            limit_vel_x = [-10.0, 10.0]
            limit_vel_y = [-0.6, 0.6]
            limit_vel_yaw = [-10.0, 10.0]
            limit_body_height = [-0.05, 0.05]
            limit_gait_frequency = [2.0, 2.01]
            limit_gait_phase = [0, 0.01]
            limit_gait_offset = [0, 0.01]
            limit_gait_bound = [0, 0.01]
            limit_gait_duration = [0.49, 0.5]
            limit_footswing_height = [0.06, 0.061]
            limit_body_pitch = [0.0, 0.01]
            limit_body_roll = [0.0, 0.01]
            limit_stance_width = [0.0, 0.01]
            limit_stance_length = [0.0, 0.01]
            limit_aux_reward_coef = [0.0, 0.01]

            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            body_height_cmd = [-0.05, 0.05]
            gait_frequency_cmd_range = [2.0, 2.01]
            gait_phase_cmd_range = [0.0, 0.01]
            gait_offset_cmd_range = [0.0, 0.01]
            gait_bound_cmd_range = [0.0, 0.01]
            gait_duration_cmd_range = [0.49, 0.5]
            footswing_height_range = [0.06, 0.061]
            body_pitch_range = [0.0, 0.01]
            body_roll_range = [0.0, 0.01]
            stance_width_range = [0.0, 0.01]
            stance_length_range = [0.0, 0.01]
            aux_reward_coef_range = [0.0, 0.01]   

        class num_bins:
            num_bins_vel_x = 25
            num_bins_vel_y = 3
            num_bins_vel_yaw = 25
            num_bins_body_height = 1
            num_bins_gait_frequency = 11
            num_bins_gait_phase = 11
            num_bins_gait_offset = 2
            num_bins_gait_bound = 2
            num_bins_gait_duration = 3
            num_bins_footswing_height = 1
            num_bins_body_pitch = 1
            num_bins_body_roll = 1
            num_bins_stance_width = 1
            num_bins_stance_length = 1
            num_bins_aux_reward_coef = 1

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            dof_pos_limits = 0.0
            dof_vel_limits = 0.0
            torque_limits = 0.0
            feet_contact_forces = 0.
            jump = 0.0
            tracking_contacts_shaped_force = 0.
            tracking_contacts_shaped_vel = 0.
            feet_slip = 0.
            feet_contact_vel = 0.0
            feet_clearance_cmd_linear = 0.
            feet_impact_vel = 0.0
            orientation_control = 0.0
            raibert_heuristic = 0.0

class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        save_interval = 50
        max_iterations = 1500
        run_name = ''
        experiment_name = 'go2'

  
