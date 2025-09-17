

import os
# from av.logging import set_libav_level
import torch
import json
import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from datasets import load_dataset, Dataset, concatenate_datasets
import numpy as np
import wandb
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from openai import OpenAI
import argparse
from util.data_group import FIELDS_GROUP_DIC

DATASET_NAME = "cais/mmlu"

def load_and_concat_dataset(group_name: str):
    train_datasets = []
    eval_datasets = []
    assert group_name in FIELDS_GROUP_DIC, f"Group name {group_name} not found in FIELDS_GROUP_DIC"
    for dataset_field in FIELDS_GROUP_DIC[group_name]:
        train_dataset = load_dataset(DATASET_NAME, dataset_field, split="test")
        train_datasets.append(train_dataset)
        eval_dataset = load_dataset(DATASET_NAME, dataset_field, split="validation")
        eval_datasets.append(eval_dataset)
    train_datasets = concatenate_datasets(train_datasets)
    eval_datasets = concatenate_datasets(eval_datasets)
    return train_datasets, eval_datasets

def format_mmlu_example(example: Dict) -> Dict:
    question = example['question']
    choices = example['choices'] # list of str
    answer = example['answer'] # int

    formatted_choices = []
    for idx, choice in enumerate(choices):
        formatted_choices.append(f"({chr(ord('A') + idx)}) {choice}")
    choice_str = "\n".join(formatted_choices)

    prompt = f"""What is the correct answer to this question? Question:\n {question}\nChoices:\n{choice_str}\nOutput format: The correct answer is (A/B/C/D)."""
    dic = {}
    dic["data_source"] = "cais/mmlu"
    dic["prompt"] = [{"role": "user", "content": prompt}]
    dic["ability"] = example['subject']
    dic["reward_model"] = {'ground_truth': chr(ord('A') + answer)}
    dic["extra_info"] = example
    return dic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="change the dataset format for Verl")
    parser.add_argument(
        "--group_name",
        type=str,
        required=True,
        help="The specific group name of the MMLU dataset to process (e.g., 'MathLogic')."
    )
    # parser.add_argument(
    #     "--save_path",
    #     type=str,
    #     required=True,
    #     # default= MODEL_NAME,
    #     help="parquet file path to save the formatted dataset"
    # )
    parser.add_argument(
        "--trail",
        type=bool,
        default=False,
        help="whether to select 1% samples as trail"
    )
    args = parser.parse_args()
    GROUP_NAME = args.group_name
    DATA_FOLDER = "data/"
    SAVE_PATH = f"{DATASET_NAME}_{GROUP_NAME}"

    train_datasets, eval_datasets = load_and_concat_dataset(GROUP_NAME)
    print(f"load {len(train_datasets)} train samples and {len(eval_datasets)} eval samples")
    train_datasets = train_datasets.map(format_mmlu_example, remove_columns=train_datasets.column_names)
    eval_datasets = eval_datasets.map(format_mmlu_example, remove_columns=eval_datasets.column_names)

    # select 10% samples as trail
    if args.trail:
        train_datasets = train_datasets.select(range(int(len(train_datasets) * 0.01)))
        eval_datasets = eval_datasets.select(range(int(len(eval_datasets) * 0.01)))

    os.makedirs(DATA_FOLDER, exist_ok=True)

    train_datasets.to_parquet(DATA_FOLDER + SAVE_PATH + "_train.parquet")
    eval_datasets.to_parquet(DATA_FOLDER + SAVE_PATH + "_eval.parquet")
    print(f"save to {DATA_FOLDER + SAVE_PATH}_train.parquet and {DATA_FOLDER + SAVE_PATH}_eval.parquet")
    print(f"train samples: {len(train_datasets)}")
    print(f"eval samples: {len(eval_datasets)}")




