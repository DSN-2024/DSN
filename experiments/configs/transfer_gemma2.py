import os

os.sys.path.append("..")
from experiments.configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.transfer = True
    config.logfile = ""

    config.tokenizer_paths = [
        "google/gemma-2-9b-it"
    ]
    config.tokenizer_kwargs = [{"use_fast": False}]
    config.model_paths = [
        "google/gemma-2-9b-it"
   ]
    config.model_kwargs = [
        {"low_cpu_mem_usage": True, "use_cache": False}
    ]
    config.conversation_templates = ["gemma"]
    config.devices = ["cuda:0"]

    config.use_augmented_loss = False
    config.control_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"

    return config
