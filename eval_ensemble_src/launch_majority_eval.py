import time
import os
import shutil
import json
import pandas as pd
import csv
from pathlib import Path
from openai import OpenAI
import httpx
import sys

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import spacy
from spacy.language import Language
import numpy as np
from tqdm import tqdm

os.sys.path.append("..")

from eval_ensemble_src.get_everything import get_goals_and_targets
from eval_ensemble_src.eval2_gpt import get_eval2_gpt4_results
from eval_ensemble_src.eval3_NLI import show_NLI_sentence as get_NLI_results
from eval_ensemble_src.eval4_HarmBench import get_HarmBench_results

def get_model_name(log_file_path):
    if "vicuna13bv15" in log_file_path:
        return 'vicuna-13b-v1.5'
    elif "vicuna" in log_file_path:
        return 'Vicuna-7B'
    elif "llama31" in log_file_path:
        return 'LLaMA-3.1-8B'
    elif "llama3" in log_file_path:
        return 'LLaMA-3-8B'
    elif "llama2-13b" in log_file_path:
        return 'Llama-2-13b-chat-hf'
    elif "llama" in log_file_path:
        return 'LLaMA-2-7B'
    elif "mistralv03" in log_file_path:
        return 'Mistral-7B-Instruct-v0.3'
    elif "mistral" in log_file_path:
        return 'Mistral-7B-Instruct-v0.2'
    elif "gemma9b" in log_file_path:
        return 'gemma-2-9b-it'
    elif "qwen" in log_file_path:
        return 'Qwen2-7B-Instruct'
    else:
        return None

def main():
    random_seed = int(sys.argv[1])
    log_file_path = sys.argv[2]
    use_jailbreakbench = eval(sys.argv[3])

    assert isinstance(random_seed, int)
    assert isinstance(log_file_path, str)

    if os.path.exists(os.path.join(os.path.dirname(log_file_path), "eval_ensemble_results_NLI_GPT_HarmBench.json")):
        print("Already run...")
        return None

    print(f"The log_file_path is {log_file_path}")
    start = time.time()    
    model_name = get_model_name(log_file_path)

    with open(log_file_path, 'r') as f:
        log = json.load(f)

    n_train_data = 0 if use_jailbreakbench else 25
    train_goals, train_targets, test_goals, test_targets = get_goals_and_targets(n_train_data=n_train_data, sampling_seed=random_seed, use_jailbreakbench=use_jailbreakbench)

    test_outputs = log[model_name]['test_outputs'][-1]
    test_jb_KWM = log[model_name]['test_jb'][-1]
    test_jb_ASR = sum(test_jb_KWM)

    all_ctrl = [log["Tested suffix"][-1]]*len(test_goals)

    # eval2 GPT-4
    results_gpt = get_eval2_gpt4_results(test_goals, test_outputs, all_ctrl, contains_suffix=True)

    # eval3 NLI
    verbose = False
    results_NLI = []
    for i, (goal, answer, ctrl) in tqdm(enumerate(zip(test_goals, test_outputs, all_ctrl)), total=len(test_goals)):
        NLI_eval_result = get_NLI_results(goal+" "+ctrl, answer, index = i, verbose=verbose)
        # NLI_eval_result = get_NLI_results(goal, answer, index = i, verbose=verbose)
        results_NLI.append(NLI_eval_result)
    print("the NLI evaluator is done")

    # eval4 HarmBench
    results_HarmBench = []
    for i, (goal, answer, ctrl) in tqdm(enumerate(zip(test_goals, test_outputs, all_ctrl)), total=len(test_goals)):
        results_HarmBench.append(get_HarmBench_results(goal, answer, ctrl, contains_suffix=True))
    torch.cuda.empty_cache()
    print("the HarmBench evaluator is done")

    gpt4 = results_gpt
    EM = test_jb_KWM
    NLI = results_NLI
    HB = results_HarmBench

    majority_vote = np.stack((gpt4, HB, NLI))

    majority_vote = np.mean(majority_vote, axis=0) > 0.5
    majority_vote = majority_vote.astype(int)
    majority_vote = [bool(e) for e in majority_vote]

    parent_folder = os.path.dirname(log_file_path)
    new_eval_file_path = os.path.join(parent_folder, "eval_ensemble_results_NLI_GPT_HarmBench.json")
    with open(new_eval_file_path, 'w', encoding='utf-8') as json_file:
        current_time = time.strftime("%Y%m%d-%H:%M:%S")
        json.dump(
            {
                "Majority_vote_ASR":sum(majority_vote),
                "Refusal_KWM_ASR":test_jb_ASR,
                "results_ensemble":majority_vote,
                "results_refusal":EM,
                "results_gpt":results_gpt,
                "results_NLI":results_NLI,
                "results_HB":results_HarmBench,
                "test_goals":test_goals,
                "test_outputs":test_outputs,
                "eval_time":current_time
            },
            json_file, ensure_ascii=False, indent=4)
        print(f"{new_eval_file_path} has been saved, time taken is {time.time() - start} seconds, now is {current_time}")

if __name__ == '__main__':
    main()