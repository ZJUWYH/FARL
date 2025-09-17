

from datasets import load_dataset, Dataset
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import PeftModel
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import random
from typing import Dict, List
import json
import argparse
import wandb
from util.data_group import FIELDS_GROUP_DIC

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"# 这里填写你的模型名称
SHORT_MODEL_NAME = "r1llama"  # 短名称，用于vLLM


DATASET_NAME = "cais/mmlu"
DATASET_SPLIT = "test"
CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]
# INFER_RESULTS_PATH = f"./res/{DATASET_NAME}_{DATASET_FIELD}_{DATASET_SPLIT}_{SHORT_MODEL_NAME}_answers.json"


BATCH_SIZE = 16  # 批处理大小，和原代码保持一致
MAX_LENGTH = 512  # 最大长度，和原代码保持一致
MAX_TOKENS = 5000

from openai import OpenAI

CLIENT = OpenAI()

def llm_extract_answer(response, client = CLIENT):
    """
    Extract the final answer from the LLM response.
    The response is expected to be a string containing the final answer in the format:
    'The correct answer is (A/B/C/D).'
    """
    prompt = f"""You are a helpful assistant tasked with extracting the final answer from a multiple-choice question response.
    The response is delimited by triple backticks.
    ```
    {response}
    ```
    ONLY return the letter of the final answer, without any additional text or explanation. If you cannot determine a clear answer, respond N.
    """
    message = [
        {"role": "system", "content": "You are a helpful assistant that extracts the final answer from a given text."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=message,
        max_tokens=10,
        temperature=0.0
    )
    answer = response.choices[0].message.content.strip()[0]
    if answer.upper() in CHOICE_LABELS:
        return CHOICE_LABELS.index(answer.upper())
    print("Failed to extract a valid answer, returning None.")
    return -1

import re
def _regex_extract_answer(text: str) -> str:
        """
        Try to extract the answer via regex with strict priority:
        1) 'correct answer is (A)'
        2) 'boxed{B}'
        3) 'boxed{(B)}'
        4) 'answer is (A)'
        Returns 'A'/'B'/'C'/'D' if found; otherwise 'None'.
        """
        if not isinstance(text, str):
            return "None"

        # 允许可选前导反斜杠 \boxed，也允许没有反斜杠的 boxed
        # 统一大小写不敏感
        flags = re.IGNORECASE

        patterns_in_priority = [
            # 1) correct answer is (A)
            re.compile(r'correct\s*answer\s*is\s*\(\s*([ABCD])\s*\)', flags),

            # 2) boxed{B}  —— 不包含括号的版本。用(?!\()确保花括号内首字符不是 '('
            re.compile(r'\\?boxed\{\s*(?!\()\s*([ABCD])\s*\}', flags),

            # 3) boxed{(B)} —— 包含括号的版本
            re.compile(r'\\?boxed\{\s*\(\s*([ABCD])\s*\)\s*\}', flags),

            # 4) answer is (A)
            re.compile(r'(?<!correct\s)answer\s*is\s*\(\s*([ABCD])\s*\)', flags),
        ]

        for pat in patterns_in_priority:
            m = pat.search(text)
            if m:
                ans = m.group(1).upper()
                if ans in ["A", "B", "C", "D"]:
                    return ans

        return "None"

def llm_extract_answer_v2(response, client = CLIENT):
    """
    Extract the final answer from the text:
    1) Try regex patterns in priority order.
    2) Fall back to an LLM extraction if regex fails.
    """
    # 先走正则优先级匹配
    regex_ans = _regex_extract_answer(response)
    if regex_ans in ["A", "B", "C", "D"]:
        return CHOICE_LABELS.index(regex_ans.upper())

    # 正则失败才回退到 LLM
    prompt = f"""You are a helpful assistant tasked with extracting the final answer from a multiple-choice question response.
    The response is delimited by triple backticks.
    ```
    {response}
    ```
    ONLY return the letter of the final answer, without any additional text or explanation. If you cannot determine a clear answer, respond N.
    """
    message = [
        {"role": "system", "content": "You are a helpful assistant that extracts the final answer from a given text."},
        {"role": "user", "content": prompt}
    ]
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=message,
            max_tokens=10,
            temperature=0.0
        )
        content = (resp.choices[0].message.content or "").strip()
        if content:
            ch = content[0].upper()
            if ch in ["A", "B", "C", "D"]:
                return CHOICE_LABELS.index(ch.upper())
    except Exception as e:
        # 这里静默失败，走到统一返回 None
        pass

    return -1

def tokenizer_fn(example, tokenizer):
    prompts = []
    # When batched=True, the 'example' is a dictionary of lists.
    # len(example["question"]) will give the batch size.
    for i in range(len(example["question"])):
        question = example["question"][i]
        choices = example["choices"][i]
        
        choice_labels = ["A", "B", "C", "D", "E", "F", "G", "H"][:len(choices)]
        formatted_choices = []
        for j, choice in enumerate(choices):
            formatted_choices.append(f"({choice_labels[j]}) {choice}")
        
        choices_text = "\n".join(formatted_choices)

        if "r1" in tokenizer.name_or_path.lower():
            prompt = f"""<｜begin▁of▁sentence｜><｜User｜>What is the correct answer to this question? Question:\n {question}\nChoices:\n{choices_text}\nOutput format: The correct answer is (A/B/C/D).<｜Assistant｜><think>\n\n</think>\n\nThe correct answer is ("""
        elif "qwen3" in tokenizer.name_or_path.lower():
            prompt = f"""<|im_start|>user\nWhat is the correct answer to this question? Question:\n {question}\nChoices:\n{choices_text}\nOutput format: The correct answer is (A/B/C/D).<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nThe correct answer is ("""
        elif "phi" in tokenizer.name_or_path.lower():
            prompt = f"""<|system|>Your name is Phi, an AI math expert developed by Microsoft.<|end|><|user|>What is the correct answer to this question? Question:\n {question}\nChoices:\n{choices_text}\nOutput format: The correct answer is (A/B/C/D).<|end|><|assistant|><think>\n\n</think>\n\nThe correct answer is ("""
        else:
            raise ValueError(f"Unsupported tokenizer: {tokenizer.name_or_path}")
        # Indent this line to be INSIDE the loop
        prompts.append(prompt)

    # Tokenize the list of prompts, not a single prompt string
    return tokenizer(prompts, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

def generate_perturb_answer(batch, model):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    with torch.no_grad():
        output = model(input_ids=torch.tensor(input_ids).to(device),
                       attention_mask=torch.tensor(attention_mask).to(device))
    if "r1" in model.module.name_or_path.lower() if hasattr(model, 'module') else model.name_or_path.lower():
        # r1模型的输出需要特殊处理
        logits = output.logits[:, -1, 32:36]
    elif "qwen3" in model.module.name_or_path.lower() if hasattr(model, 'module') else model.name_or_path.lower():
        # qwen3模型的输出需要特殊处理
        logits = output.logits[:, -1, 32:36]
    elif "phi" in model.module.name_or_path.lower() if hasattr(model, 'module') else model.name_or_path.lower():
        # phi模型的输出需要特殊处理
        logits = output.logits[:, -1, 32:36]
    else:
        raise ValueError(f"Unsupported model: {model.name_or_path}")
    sorted_indices = torch.sort(logits, dim=1, descending=True).indices.cpu().tolist()
    preturb_answer = []
    answer_wo_think = []
    for i in range(len(sorted_indices)):
        answer_wo_think.append(sorted_indices[i][0])
        if sorted_indices[i][0] == batch["answer"][i]:
            preturb_answer.append(sorted_indices[i][1])
        else:
            preturb_answer.append(sorted_indices[i][0])
    return {"answer_wo_think": answer_wo_think,
        "perturb_answer": preturb_answer}

if __name__ == "__main__":
    # --- 1. 加载数据集 ---
    parser = argparse.ArgumentParser(description="Run inference with a specified MMLU dataset group.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The specific dataset name to process (e.g., 'cais/mmlu')."
    )
    parser.add_argument(
        "--group_name",
        type=str,
        required=True,
        help="The specific group name of the MMLU dataset to process (e.g., 'MathLogic')."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        # default= MODEL_NAME,
        help="The name of the model to use for inference."
    )
    parser.add_argument(
        "--short_model_name",
        type=str,
        required=True,
        # default=SHORT_MODEL_NAME,
        help="The short name of the model for vLLM."
    )
    args = parser.parse_args()
    DATASET_NAME = args.dataset_name
    GROUP_NAME = args.group_name
    MODEL_NAME = args.model_name
    SHORT_MODEL_NAME = args.short_model_name
    print(f"Loading dataset {DATASET_NAME} with group {GROUP_NAME} and split {DATASET_SPLIT}...")
    INFER_RESULTS_PATH = f"./res2/{DATASET_NAME}_{GROUP_NAME}_{DATASET_SPLIT}_{SHORT_MODEL_NAME}_answers.json"
    SAVE_PATH = f"./res2/{DATASET_NAME}_{GROUP_NAME}_{DATASET_SPLIT}_{SHORT_MODEL_NAME}_perturbed_answers.json"
    wandb.init(project="MemoryPerturbGroup", name=f"{DATASET_NAME}_{GROUP_NAME}_{DATASET_SPLIT}_{SHORT_MODEL_NAME}", group="cot_quality_analysis")
    wandb.config.update(args)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    with open(INFER_RESULTS_PATH, "r") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)
    # if save_path exists:
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "r") as f:
            data = json.load(f)
        correct_dataset = Dataset.from_list(data)
        accuracy = len(correct_dataset) / len(dataset)
        print(f"accuracy: {accuracy}")
        wandb.log({"accuracy": accuracy})
    else:
        dataset = dataset.map(lambda x: {"answer_w_think": llm_extract_answer_v2(x["model_answer"])})
        correct_dataset = dataset.filter(lambda x: x["answer_w_think"] == x["answer"])  # 只保留有正确答案的样本
        print(f"accuracy: {len(correct_dataset)}/{len(dataset)} = {len(correct_dataset)/len(dataset)}")
        wandb.log({"accuracy": len(correct_dataset) / len(dataset)})
    # count the number of mean token length of model answer and model answer with thinking
    mean_token_length = sum(len(tokenizer.encode(x["model_answer"])) for x in dataset) / len(dataset)
    print(f"mean token length: {mean_token_length}")
    wandb.log({"mean_token_length": mean_token_length})
    print("cot quality analysis done.")

