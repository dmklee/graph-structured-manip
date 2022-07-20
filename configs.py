def get_env_configs(**kwargs):
    default = {
        'obs_size' : 90,
        'side_view_size' : 72,
        'side_view_distance' : 0.75,
        'side_view_yaw' : -90,
        'side_view_pitch' : -50,
        'side_view_fov' : 20,
        'in_hand_mode' : 'normal',
        'max_height' : 3,
        'distractor_max_angle' : 0.0,
        'robot' : 'kuka',
        'seed' : 0,
        'render' : False,
        'action_sequence' : 'pxy',
        'include_all_objects' : False,
        'num_distractors' : 0,
        'use_structured_encodings' : False,
        'other_obj_chars' : [],
    }

    default.update(kwargs)
    return default

def get_training_configs(**kwargs):
    default = {
        'sym_encoder_lr': 0.001,
        'sym_encoder_img_size': 60,
        'buffer_size' : 30000,
        'goal_label_method' : 'subgoal',
        'n_envs' : 5,
        'env_reset_goal_distribution' : 'terminal',
        'q_opt_cycles' : 2,
        'enc_opt_cycles' : 2,
        'batch_size' : 32,
        'use_local_policy' : True,
        'perlin_noise' : 0,
        'add_smoothing' : False,
        'specific_goal' : None,
        'time_to_reach_subgoal' : 4,
        'gamma': 1.0,
        'reward_style': 'negative',
        'qmap_noise_range' : [0.5,0.05],
        'random_action_range' : [0.01,0],
        'q_target_update_freq' : 1000,
    }

    default.update(kwargs)
    return default
