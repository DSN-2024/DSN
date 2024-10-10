import os
import shutil
import json
import pandas as pd
import csv
from pathlib import Path
from openai import OpenAI
import httpx

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import spacy
import random
from spacy.language import Language
import numpy as np
from tqdm import tqdm

def get_goals_and_targets(n_train_data = 25, n_test_data = 100, sampling_seed = -1, offset = 0, verbose = True, use_jailbreakbench = False):
    if use_jailbreakbench:
        adv_bench_file_path = "../data/JailbreakBench_harmful_behaviors.csv"
    else:
        adv_bench_file_path = "../data/AdvBench_harmful_behaviors.csv"
    train_data = pd.read_csv(adv_bench_file_path)
    
    if not sampling_seed == -1:
        # in the random sampling mode, theres no need for the offset var, thus del it
        del offset      

    if sampling_seed >= 0:
        random.seed(sampling_seed)
        elements = list(range(len(train_data['target'].tolist())))
        random_list = random.sample(elements, len(elements))
        train_list = random_list[:n_train_data]   
        test_list = random_list[ n_train_data : n_train_data + n_test_data ]
    else:
        train_list = list(range(offset,offset+n_train_data))
        test_list = list(range(offset+n_train_data,offset+n_train_data+n_test_data))

    # for the training data
    train_targets = [ train_data['target'].tolist()[i] for i in train_list ]
    train_goals = [ train_data['goal'].tolist()[i] for i in train_list ]

    # for the testing data
    test_targets = [ train_data['target'].tolist()[i] for i in test_list ]
    test_goals = [ train_data['goal'].tolist()[i] for i in test_list ]

    assert len(train_goals) == len(train_targets)
    assert len(test_goals) == len(test_targets)

    if verbose:
        print("-"*25+" get everything fn "+"-"*25)
        print(f"The sampling seed is {sampling_seed}")
        print('Loaded {} train goals'.format(len(train_goals)))
        print('Loaded {} test goals'.format(len(test_goals)))

        print(f"The training index list is {train_list}")
        print(f"The testing index list is {test_list}")
        print("-"*25+" get everything fn "+"-"*25)

    return train_goals, train_targets, test_goals, test_targets

if __name__ == '__main__':
    get_goals_and_targets()