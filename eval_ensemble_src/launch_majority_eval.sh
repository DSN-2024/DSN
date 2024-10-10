#!/bin/bash

# sample file paht
STRING_ARRAY=(
    '../experiments/evalLastStep_dsn/20240101-13:00:00 1E0/vicuna_dsn_25_offset0.json'
)

for STR in "${STRING_ARRAY[@]}"; do
  python3 launch_majority_eval.py -1 "$STR" False
done                              # random_seed, logfile, UseJBB
