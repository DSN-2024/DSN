from ml_collections import config_dict

def get_config():
    config = config_dict.ConfigDict()

    # Experiment type
    config.transfer = False

    # General parameters 
    config.target_weight=1.0
    config.control_weight=0.0
    config.progressive_goals=False
    config.progressive_models=False
    config.anneal=False
    config.incr_control=False
    config.stop_on_success=False
    config.verbose=True
    config.allow_non_ascii=False
    config.num_train_models=1

    # Results
    config.result_prefix = 'results/individual_vicuna7b'

    # tokenizers
    config.tokenizer_paths=['/data/vicuna/vicuna-7b-v1.3']
    config.tokenizer_kwargs=[{"use_fast": False}]
    
    config.model_paths=['/data/vicuna/vicuna-7b-v1.3']
    config.model_kwargs=[{"low_cpu_mem_usage": True, "use_cache": False}]
    config.conversation_templates=['vicuna']
    config.devices=['cuda:0']

    # data
    config.train_data = ''
    config.test_data = ''
    config.n_train_data = 50
    config.n_test_data = 0
    config.data_offset = 0

    # attack-related parameters
    config.attack = 'dsn'
    config.control_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    config.n_steps = 500
    config.test_steps = 50
    config.batch_size = 512
    config.lr = 0.01
    config.topk = 256
    config.temp = 1
    config.filter_cand = True

    config.gbda_deterministic = True

    # DSN related config items
    config.use_augmented_loss = False
    config.augmentation_token_length = 20
    config.augmentation_gen_temp = 1
    config.use_empty_system_prompt = False
    config.debug_mode = False
    config.augmented_loss_alpha = 1.0
    config.dsn_notes = ""
    config.use_aug_sampling = False
    config.use_different_aug_sampling_alpha = False
    config.aug_sampling_alpha2 = 0.0
    config.eval_batchsize = 1
    config.eval_max_new_len = 1
    config.random_seed_for_sampling_targets = -1
    config.use_target_loss_cosine_decay = False
    config.eval_target_folder_name = ""
    config.eval_with_repeated_sys_prompt = False

    return config
