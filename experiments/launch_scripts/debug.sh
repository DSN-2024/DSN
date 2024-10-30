#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export n=25
export model=llama2 # choose from: gemma, gemma2, llama2, llama2_13b, llama3, llama31, mistralv02, mistralv03, qwen, vicuna_13b, vicuna
data_offset=0

# Create results folder if it doesn't exist
if [ ! -d "../results_debug" ]; then
    mkdir "../results_debug"
    echo "Folder '../results_debug' created."
else
    echo "Folder '../results_debug' already exists."
fi

python -u ../main.py \
    --config="../configs/transfer_${model}.py" \
    --config.attack=dsn \
    --config.train_data="../../data/AdvBench_harmful_behaviors.csv" \
    --config.result_prefix="../results_dsn/${model}_dsn_${n}_offset${data_offset}" \
    --config.progressive_goals=False \
    --config.stop_on_success=False \
    --config.num_train_models=1 \
    --config.allow_non_ascii=False \
    --config.n_train_data=$n \
    --config.n_test_data=1 \
    --config.n_steps=500 \
    --config.data_offset=$data_offset \
    --config.test_steps=2 \
    --config.batch_size=20 \
    --config.use_augmented_loss=True \
    --config.use_empty_system_prompt=False \
    --config.debug_mode=True \
    --config.augmented_loss_alpha=1.0 \
    --config.use_aug_sampling=False \
    --config.use_different_aug_sampling_alpha=False \
    --config.aug_sampling_alpha2=0.0 \
    --config.random_seed_for_sampling_targets=-1 \
    --config.use_target_loss_cosine_decay=True \
    --config.dsn_notes="Some experiment notes, may be stored in the logging file"
