'''A main script to run attack for LLMs.'''
import time
import importlib
import numpy as np
import torch.multiprocessing as mp
from absl import app
from ml_collections import config_flags
from pathlib import Path
import os
import math

from llm_attacks import get_goals_and_targets, get_workers

_CONFIG = config_flags.DEFINE_config_file('config')

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Function to import module at the runtime
def dynamic_import(module):
    return importlib.import_module(module)

def main(_):

    mp.set_start_method('spawn')

    params = _CONFIG.value

    attack_lib = dynamic_import(f'llm_attacks.{params.attack}')

    print(params)

    train_goals, train_targets, test_goals, test_targets = get_goals_and_targets(params)

    # Regularization, you may custom these rules accordingly
    process_fn = lambda s: s.replace('Sure, h', 'H')
    process_fn2 = lambda s: s.replace("Sure, here is", "Sure, here's")
    train_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in train_targets]
    test_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in test_targets]

    workers, test_workers = get_workers(params)

    # importing
    managers = {
        "AP": attack_lib.AttackPrompt,
        "PM": attack_lib.PromptManager,
        "MPA": attack_lib.MultiPromptAttack,
    }
    
    # vanilla check
    if params.use_augmented_loss:
        if params.augmented_loss_alpha <= 0:
            assert False, "Wrong setting!"
        else:
            log10 = math.log10(params.augmented_loss_alpha)
            if log10 == int(log10):
                log10 = str(int(log10))
            else:
                log10 = f"{log10:.3}"
    elif params.use_target_loss_cosine_decay:
        log10 = "None"
    else:
        # original GCG implementation
        log10 = "Original"

    timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    logfile = f"{params.result_prefix}.json"
    path = Path(logfile)
    logfile = Path(*path.parts[:2], f'{timestamp} 1E{log10}', *path.parts[2:])
    logfile = str(logfile)
    params.result_prefix = logfile
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    if params.transfer:
        attack = attack_lib.ProgressiveMultiPromptAttack(
            train_goals,
            train_targets,
            workers,
            progressive_models=params.progressive_models,
            progressive_goals=params.progressive_goals,
            control_init=params.control_init,
            logfile=logfile,
            managers=managers,
            test_goals=test_goals,
            test_targets=test_targets,
            test_workers=test_workers,
            para=params,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps
        )
    else:
        assert False, "only conducting Multiple experiments..."

    attack.run(
        n_steps=params.n_steps,
        batch_size=params.batch_size, 
        topk=params.topk,
        temp=params.temp,                           # temperature for sampling
        allow_non_ascii=params.allow_non_ascii,
        target_weight=params.target_weight,         # equals 1
        control_weight=params.control_weight,       # equals 0
        anneal=params.anneal,
        test_steps=getattr(params, 'test_steps', 1),
        incr_control=params.incr_control,
        stop_on_success=params.stop_on_success,
        verbose=params.verbose,
        filter_cand=params.filter_cand,
    )

    for worker in workers + test_workers:
        worker.stop()

if __name__ == '__main__':
    app.run(main)