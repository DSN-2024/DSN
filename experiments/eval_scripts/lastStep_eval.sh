#!/bin/bash

export n=25
export model=gemma # choose from: gemma, gemma2, llama2, llama2_13b, llama3, llama31, mistralv02, mistralv03, qwen, vicuna_13b, vicuna

# sample file path, select the corresponding file path please
file_list=(
    '../results_dsn/20240101-13:00:00 1E0/vicuna_dsn_25_offset0.json'
)

for file_path in "${file_list[@]}"
do

    # evaluate upon the AdvBench dataset
    python -u ../evaluate_last_step.py \
        --config="../configs/transfer_${model}.py" \
        --config.train_data="../../data/AdvBench_harmful_behaviors.csv" \
        --config.data_offset=0 \
        --config.n_train_data=25 \
        --config.n_test_data=100 \
        --config.use_empty_system_prompt=False \
        --config.eval_batchsize=25 \
        --config.eval_max_new_len=512 \
        --config.eval_target_folder_name="" \
        --config.logfile="${file_path}" \
        --config.dsn_notes="For the last step, by generating 512 tokens, eval DSN by Refusal Matching...log file path is ${file_path}"

    # evaluate upon the JailbreakBench dataset
    python -u ../evaluate_last_step.py \
        --config="../configs/transfer_${model}.py" \
        --config.train_data="../../data/JailbreakBench_harmful_behaviors.csv" \
        --config.data_offset=0 \
        --config.n_train_data=100 \
        --config.n_test_data=0 \
        --config.use_empty_system_prompt=False \
        --config.eval_batchsize=25 \
        --config.eval_max_new_len=512 \
        --config.eval_target_folder_name="evalJBBLastStep" \
        --config.logfile="${file_path}" \
        --config.dsn_notes="For the last step, by generating 512 tokens, eval DSN by Refusal Matching...log file path is ${file_path}"

done
