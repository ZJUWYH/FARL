# cd memory-perturb
# conda activate cot
# CUDA_VISIBLE_DEVICES=0,1,2,3 python indentify_wrong_answer2.py --dataset_field nutrition --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --short_model_name r1llama 
# 2. use llm as judge to extract answer, add the "answer without thinking", and then filter the correct ones, <think>\n\n<think> fix
# 3. add an augment to only calculate the accuracy, change the dataset to group fields
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file config/qwen_acc_4process.yaml indentify_wrong_answer3_acc.py --group_name MathLogic --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --short_model_name r1qwen14b

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
from data_group import FIELDS_GROUP_DIC
from accelerate import Accelerator
from accelerate.utils import gather_object
import torch.distributed as dist
accelerator = Accelerator()
device = accelerator.device

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

def generate_perturb_answer(batch, model, tokenizer):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    with torch.no_grad():
        output = model(input_ids=torch.tensor(input_ids).to(device),
                       attention_mask=torch.tensor(attention_mask).to(device))
    if "r1" in tokenizer.name_or_path.lower():
        # r1模型的输出需要特殊处理
        logits = output.logits[:, -1, 32:36]
    elif "qwen3" in tokenizer.name_or_path.lower():
        # qwen3模型的输出需要特殊处理
        logits = output.logits[:, -1, 32:36]
    elif "phi" in tokenizer.name_or_path.lower():
        # phi模型的输出需要特殊处理
        logits = output.logits[:, -1, 32:36]
    else:
        raise ValueError(f"Unsupported model: {tokenizer.name_or_path}")
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
    parser.add_argument(
        "--only_calculate_accuracy",
        type=bool,
        required=False,
        default=False,
        help="Only calculate the accuracy."
    )
    args = parser.parse_args()
    GROUP_NAME = args.group_name
    MODEL_NAME = args.model_name
    SHORT_MODEL_NAME = args.short_model_name
    if accelerator.is_main_process:
        print(f"Loading dataset {DATASET_NAME} with group {GROUP_NAME} and split {DATASET_SPLIT}...")
    INFER_RESULTS_PATH = f"./res2/{DATASET_NAME}_{GROUP_NAME}_{DATASET_SPLIT}_{SHORT_MODEL_NAME}_answers.json"
    if accelerator.is_main_process:
        wandb.init(project="MemoryPerturbGroup", name=f"{DATASET_NAME}_{GROUP_NAME}_{DATASET_SPLIT}_{SHORT_MODEL_NAME}", group="wrong_answer")
        wandb.config.update(args)

    with open(INFER_RESULTS_PATH, "r") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)
    if accelerator.is_main_process:
        dataset = dataset.map(lambda x: {"answer_w_think": llm_extract_answer(x["model_answer"])})
        if args.only_calculate_accuracy:
            correct_dataset = dataset.filter(lambda x: x["answer_w_think"] == x["answer"])  # 只保留有正确答案的样本
            print(f"accuracy: {len(correct_dataset)}/{len(dataset)} = {len(correct_dataset)/len(dataset)}")
            wandb.log({"accuracy": len(correct_dataset) / len(dataset)})
            print("Only calculate the accuracy.")
            exit()
    if accelerator.is_main_process:
        dataset_list = dataset.to_list()
    else:
        dataset_list = None
    
    # Wait for all processes and broadcast
    accelerator.wait_for_everyone()
    dataset_list = accelerator.broadcast_object_list([dataset_list])[0]
    dataset = Dataset.from_list(dataset_list)


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16)
    model = accelerator.prepare(model)
    # dataset = load_dataset(DATASET_NAME, DATASET_FIELD, split=DATASET_SPLIT)
    # dataset = dataset.select(range(DATASET_SIZE)) if DATASET_SIZE < len(dataset) else dataset
    # import re
    # def extract_answer(str):
    #     match = re.search(r'.*the correct answer is \(([a-zA-Z])', str, re.IGNORECASE)
    #     letter = match.group(1).upper() if match else "None"
    #     if letter not in ["A", "B", "C", "D", "E", "F", "G", "H"]:
    #         letter = "None"
    #     return letter



    tokenizerd_dataset = dataset.map(
        tokenizer_fn,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        remove_columns=['subject', "correct_choice", "formatted_prompt",],
        desc="Tokenizing dataset",
        batch_size=BATCH_SIZE,
    )

    perturb_ds = tokenizerd_dataset.map(generate_perturb_answer, fn_kwargs={"model": model, "tokenizer": tokenizer},batched=True, batch_size=BATCH_SIZE, remove_columns = ['input_ids', 'attention_mask'])

    if accelerator.is_main_process:
        # Calculate accuracy for answer_wo_think and answer_w_think
        correct_wo_think = sum(1 for pred, true in zip(perturb_ds['answer_wo_think'], perturb_ds['answer']) if pred == true)
        correct_w_think = sum(1 for pred, true in zip(perturb_ds['answer_w_think'], perturb_ds['answer']) if pred == true)
        total = len(perturb_ds['answer'])

        acc_wo_think = correct_wo_think / total
        acc_w_think = correct_w_think / total

        print(f"Accuracy without thinking: {acc_wo_think:.4f}")
        print(f"Accuracy with thinking: {acc_w_think:.4f}")

        # Calculate the 4 transition ratios
        # Case 1: answer_wo_think wrong, answer_w_think wrong
        wrong_wrong = sum(1 for wo, w, true in zip(perturb_ds['answer_wo_think'], perturb_ds['answer_w_think'], perturb_ds['answer']) 
                        if wo != true and w != true)

        # Case 2: answer_wo_think wrong, answer_w_think correct
        wrong_correct = sum(1 for wo, w, true in zip(perturb_ds['answer_wo_think'], perturb_ds['answer_w_think'], perturb_ds['answer']) 
                            if wo != true and w == true)

        # Case 3: answer_wo_think correct, answer_w_think wrong
        correct_wrong = sum(1 for wo, w, true in zip(perturb_ds['answer_wo_think'], perturb_ds['answer_w_think'], perturb_ds['answer']) 
                            if wo == true and w != true)

        # Case 4: answer_wo_think correct, answer_w_think correct
        correct_correct = sum(1 for wo, w, true in zip(perturb_ds['answer_wo_think'], perturb_ds['answer_w_think'], perturb_ds['answer']) 
                            if wo == true and w == true)

        # Calculate ratios
        ratio_wrong_wrong = wrong_wrong / total
        ratio_wrong_correct = wrong_correct / total
        ratio_correct_wrong = correct_wrong / total
        ratio_correct_correct = correct_correct / total

        print(f"\n=== Transition Ratios ===")
        print(f"Wrong → Wrong: {ratio_wrong_wrong:.4f} ({wrong_wrong}/{total})")
        print(f"Wrong → Correct: {ratio_wrong_correct:.4f} ({wrong_correct}/{total})")
        print(f"Correct → Wrong: {ratio_correct_wrong:.4f} ({correct_wrong}/{total})")
        print(f"Correct → Correct: {ratio_correct_correct:.4f} ({correct_correct}/{total})")

        wandb.log({
            "acc_wo_think": acc_wo_think,
            "acc_w_think": acc_w_think,
            "ratio_wrong_wrong": ratio_wrong_wrong,
            "ratio_wrong_correct": ratio_wrong_correct,
            "ratio_correct_wrong": ratio_correct_wrong,
            "ratio_correct_correct": ratio_correct_correct
        })

        # filter the answer_w_think correct ones
        perturb_ds = perturb_ds.filter(lambda x: x["answer_w_think"] == x["answer"])


        # 保存处理后的数据集到磁盘
        SAVE_PATH = f"./res2/{DATASET_NAME}_{GROUP_NAME}_{DATASET_SPLIT}_{SHORT_MODEL_NAME}_perturbed_answers.json"
        print(f"\nSaving dataset to disk at {SAVE_PATH}...")
        
        # 确保目录存在
        import os
        import json
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        
        with open(SAVE_PATH, 'w', encoding='utf-8') as f:
            json.dump(perturb_ds.to_list(), f, indent=4, ensure_ascii=False)
        
        print("Done!")
    # Wait for all processes to finish
    accelerator.wait_for_everyone()


    