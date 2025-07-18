from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2RoughCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_observations = 69
        num_scalar_observations = 69
        # if not None a privilige_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_privileged_obs = 71
        privileged_future_horizon = 1
        num_observation_history = 30
        observe_vel = False
        observe_only_ang_vel = True
        observe_only_lin_vel = False
        observe_yaw = False
        observe_contact_states = False
        observe_command = True
        observe_height_command = False
        observe_gait_commands = True
        observe_timing_parameter = False
        observe_clock_inputs = False
        observe_two_prev_actions = True
        observe_imu = False
        debug_viz = False
        all_agents_share = False

        priv_observe_friction = True
        priv_observe_ground_friction = False
        priv_observe_restitution = True
        priv_observe_base_mass = False
        priv_observe_com_displacement = False
        priv_observe_motor_strength = False
        priv_observe_motor_offset = False
        priv_observe_joint_friction = False
        priv_observe_contact_states = False
        priv_observe_body_velocity = False
        priv_observe_foot_height = False
        priv_observe_body_height = False
        priv_observe_gravity = False
        priv_observe_terrain_type = False
        priv_observe_clock_inputs = False
        priv_observe_doubletime_clock_inputs = False
        priv_observe_halftime_clock_inputs = False
        priv_observe_desired_contact_states = False
        priv_observe_dummy_variable = False

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh
        border_size = 0  # 25 # [m]
        curriculum = False
        terrain_noise_magnitude = 0.0
        # rough terrain only:
        terrain_smoothness = 0.005
        measure_heights = False
        min_init_terrain_level = 0
        terrain_length = 5.
        terrain_width = 5.
        num_rows = 30  # number of terrain rows (levels)
        num_cols = 30  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
        difficulty_scale = 1.
        x_init_range = 0.2
        y_init_range = 0.2
        yaw_init_range = 3.14
        x_init_offset = 0.
        y_init_offset = 0.
        teleport_robots = False
        teleport_thresh = 0.3
        max_platform_height = 0.2
        center_robots = False
        center_span = 5
        center_robots = True
        center_span = 4

    class commands( LeggedRobotCfg.commands ):
        command_curriculum = True
        max_reverse_curriculum = 1.
        max_forward_curriculum = 1.
        yaw_command_curriculum = False
        max_yaw_curriculum = 1.
        exclusive_command_sampling = False
        num_commands = 15
        subsample_gait = False
        gait_interval_s = 10.  # time between resampling gait params
        vel_interval_s = 10.
        jump_interval_s = 20.  # time between jumps
        jump_duration_s = 0.1  # duration of jump
        jump_height = 0.3
        heading_command = False  # if true: compute ang vel command from heading error
        global_reference = False
        observe_accel = False
        curriculum_type = "RewardThresholdCurriculum"
        lipschitz_threshold = 0.9

        num_lin_vel_bins = 30
        lin_vel_step = 0.3
        num_ang_vel_bins = 30
        ang_vel_step = 0.3
        distribution_update_extension_distance = 1
        curriculum_seed = 100

        impulse_height_commands = False

        num_bins_vel_x = 21
        num_bins_vel_y = 1
        num_bins_vel_yaw = 21
        num_bins_body_height = 1
        num_bins_gait_frequency = 1
        num_bins_gait_phase = 1
        num_bins_gait_offset = 1
        num_bins_gait_bound = 1
        num_bins_gait_duration = 1
        num_bins_footswing_height = 1
        num_bins_body_pitch = 1
        num_bins_body_roll = 1
        num_bins_aux_reward_coef = 1
        num_bins_compliance = 1
        num_bins_compliance = 1
        num_bins_stance_width = 1
        num_bins_stance_length = 1

        exclusive_phase_offset = False
        binary_phases = True
        pacing_offset = False
        balance_gait_distribution = True
        gaitwise_curricula = True

        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_y = [-0.6, 0.6]  # min max [m/s]
            body_height_cmd = [-0.25, 0.15]
            limit_vel_x = [-5.0, 5.0]
            limit_vel_y = [-0.6, 0.6]
            limit_vel_yaw = [-5.0, 5.0]
            limit_body_height = [-0.25, 0.15]
            limit_gait_phase = [0.0, 1.0]
            limit_gait_offset = [0.0, 1.0]
            limit_gait_bound = [0.0, 1.0]
            limit_gait_frequency = [2.0, 4.0]
            limit_gait_duration = [0.5, 0.5]
            limit_footswing_height = [0.03, 0.35]
            limit_body_pitch = [-0.4, 0.4]
            limit_body_roll = [-0.0, 0.0]
            limit_aux_reward_coef = [0.0, 0.01]
            limit_compliance = [0.0, 0.01]
            limit_stance_width = [0.10, 0.45]
            limit_stance_length = [0.35, 0.45]
            gait_phase_cmd_range = [0.0, 1.0]
            gait_offset_cmd_range = [0.0, 1.0]
            gait_bound_cmd_range = [0.0, 1.0]
            gait_frequency_cmd_range = [2.0, 4.0]
            gait_duration_cmd_range = [0.5, 0.5]
            footswing_height_range = [0.03, 0.35]
            body_pitch_range = [-0.4, 0.4]
            body_roll_range = [-0.0, 0.0]
            aux_reward_coef_range = [0.0, 0.01]
            compliance_range = [0.0, 0.01]
            stance_width_range = [0.10, 0.45]
            stance_length_range = [0.35, 0.45]
    
    # class curriculum_thresholds:
    #     tracking_lin_vel = 0.8  # closer to 1 is tighter
    #     tracking_ang_vel = 0.7
    #     tracking_contacts_shaped_force = 0.9  # closer to 1 is tighter
    #     tracking_contacts_shaped_vel = 0.9

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
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        hip_scale_reduction = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand( LeggedRobotCfg.domain_rand ):
        rand_interval_s = 4
        randomize_rigids_after_start = False
        friction_range = [0.1, 3.0]  # increase range
        randomize_restitution = True
        restitution_range = [0, 0.4]
        randomize_base_mass = True
        # add link masses, increase range, randomize inertia, randomize joint properties
        added_mass_range = [-1, 3]
        randomize_com_displacement = False
        # add link masses, increase range, randomize inertia, randomize joint properties
        com_displacement_range = [-0.15, 0.15]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        randomize_motor_offset = True
        motor_offset_range = [-0.02, 0.02]
        randomize_Kp_factor = False
        Kp_factor_range = [0.8, 1.3]
        randomize_Kd_factor = False
        Kd_factor_range = [0.5, 1.5]
        gravity_rand_interval_s = 8
        gravity_impulse_duration = 0.99
        randomize_gravity = True
        gravity_range = [-1.0, 1.0]
        push_robots = False
        max_push_vel_xy = 0.5
        randomize_lag_timesteps = True
        lag_timesteps = 6
        ground_friction_range = [0.0, 0.0]
  
    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        sigma_rew_neg = 0.02
        reward_container_name = "CoRLRewards"
        tracking_sigma_lat = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_long = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9  # percentage of urdf limits, values above this limit are penalized
        base_height_target = 0.3
        use_terminal_body_height = True
        terminal_body_height = 0.05
        terminal_foot_height = -0.005
        kappa_gait_probs = 0.07
        gait_force_sigma = 100
        gait_vel_sigma = 10
        footswing_height = 0.09
        class scales( LeggedRobotCfg.rewards.scales ):
            lin_vel_z = -0.02
            ang_vel_xy = -0.001
            orientation_control = -5.0
            torques = -0.0001
            dof_vel = -1e-4
            feet_air_time = 0.0
            collision = -5
            tracking_lin_vel_lat = 0.
            tracking_lin_vel_long = 0.
            tracking_contacts = 0.
            tracking_contacts_shaped = 0.
            tracking_contacts_shaped_force = 4
            tracking_contacts_shaped_vel = 4
            jump = 10
            energy = 0.0
            energy_expenditure = 0.0
            survival = 0.0
            dof_pos_limits = -10.0
            feet_contact_forces = 0.
            feet_slip = -0.04
            feet_clearance = -0.0
            feet_clearance_cmd = -0.0
            feet_clearance_cmd_linear = -30.0
            dof_pos = 0.
            action_smoothness_1 = -0.1
            action_smoothness_2 = -0.1
            base_motion = 0.
            feet_impact_vel = 0.0
            raibert_heuristic = -10.0

    class normalization( LeggedRobotCfg.normalization ):
        clip_actions = 10

        friction_range = [0, 1]
        ground_friction_range = [0, 1]
        restitution_range = [0, 1.0]
        added_mass_range = [-1., 3.]
        com_displacement_range = [-0.1, 0.1]
        motor_strength_range = [0.9, 1.1]
        motor_offset_range = [-0.05, 0.05]
        Kp_factor_range = [0.8, 1.3]
        Kd_factor_range = [0.5, 1.5]
        joint_friction_range = [0.0, 0.7]
        contact_force_range = [0.0, 50.0]
        contact_state_range = [0.0, 1.0]
        body_velocity_range = [-6.0, 6.0]
        foot_height_range = [0.0, 0.15]
        body_height_range = [0.0, 0.60]
        gravity_range = [-1.0, 1.0]
        motion = [-0.01, 0.01]
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            imu = 0.1
            friction_measurements = 1.0
            body_height_cmd = 2.0
            gait_phase_cmd = 1.0
            gait_freq_cmd = 1.0
            footswing_height_cmd = 0.15
            body_pitch_cmd = 0.3
            body_roll_cmd = 0.3
            aux_reward_cmd = 1.0
            compliance_cmd = 1.0
            stance_width_cmd = 1.0
            stance_length_cmd = 1.0
            segmentation_image = 1.0
            rgb_image = 1.0
            depth_image = 1.0
        
    class noise( LeggedRobotCfg.noise ):
        class noise_scales( LeggedRobotCfg.noise.noise_scales ):
            imu = 0.1
            contact_states = 0.05
            friction_measurements = 0.0
            segmentation_image = 0.0
            rgb_image = 0.0
            depth_image = 0.0

class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        save_interval = 100
        max_iterations = 1500
        run_name = ''
        experiment_name = 'go2'

  
