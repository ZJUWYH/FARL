"""
Independent activation extraction tool for Llama models using PyVene.
Stream-save with memmap instead of dataset.map to reduce RAM usage.
"""

# version 2: use the batched map
# version 3: use the prefill answer

import os
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyvene as pv

import asyncio
from datasets import load_dataset, Dataset
from openai import AsyncOpenAI, OpenAI
import json
import os
import argparse  # 重复导入保持与原文件一致
from tqdm import tqdm
import wandb
from dataclasses import asdict
from data_group import FIELDS_GROUP_DIC

# --- 数据集配置 ---
DATASET_NAME = "cais/mmlu"
DATASET_SPLIT = "test"
CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]

# --- 推理配置 ---
BATCH_SIZE = 12
MAX_TOKENS = 5000


# =============================================================================
# Model name mappings
# =============================================================================

HF_NAMES = {
    'llama_7B_chat': 'meta-llama/Llama-2-7b-chat-hf',
}


# =============================================================================
# Intervener classes (from interveners.py)
# =============================================================================

def wrapper(intervener):
    """Wrapper function for PyVene interventions."""
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)
    return wrapped


class Collector():
    """Collector class to gather activations from specific model components."""
    collect_state = True
    collect_action = False

    def __init__(self, multiplier, head):
        self.head = head
        self.states = []
        self.actions = []

    def reset(self):
        self.states = []
        self.actions = []

    def __call__(self, b, s):
        if self.head == -1:
            # Collect all head activations (here the full hidden vector at last token)
            self.states.append(b[0, -1].detach().clone())  # (hidden_size)
        else:
            # Collect specific head activation
            self.states.append(b[0, -1].reshape(32, -1)[self.head].detach().clone())
        return b


class ITI_Intervener():
    """Intervention class for modifying activations."""
    collect_state = True
    collect_action = True
    attr_idx = -1

    def __init__(self, direction, multiplier):
        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction)
        self.direction = direction.cuda().half()
        self.multiplier = multiplier
        self.states = []
        self.actions = []

    def reset(self):
        self.states = []
        self.actions = []

    def __call__(self, b, s):
        self.states.append(b[0, -1].detach().clone())
        action = self.direction.to(b.device)
        self.actions.append(action.detach().clone())
        b[0, -1] = b[0, -1] + action * self.multiplier
        return b


def chat_template_prefill(question_str, answer_str, tokenizer):
    messages = [
        {"role": "user", "content": question_str},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    formatted_prompt = formatted_prompt + answer_str
    return formatted_prompt


def get_llama_activations_pyvene(collected_model, collectors, prompt):
    with torch.no_grad():
        prompt = prompt.to(collected_model.get_device())
        output = collected_model({"input_ids": prompt, "output_hidden_states": True})[1]
    hidden_states = output.hidden_states
    hidden_states = torch.stack(hidden_states, dim=0).squeeze()
    hidden_states = hidden_states.detach().float().cpu().numpy()
    head_wise_hidden_states = []
    for collector in collectors:
        if collector.collect_state:
            states_per_gen = torch.stack(collector.states, axis=0).float().cpu().numpy()
            head_wise_hidden_states.append(states_per_gen)
        else:
            head_wise_hidden_states.append(None)
        collector.reset()
    mlp_wise_hidden_states = []
    head_wise_hidden_states = torch.stack([torch.tensor(h) for h in head_wise_hidden_states], dim=0).squeeze().numpy()
    return hidden_states[:, -1, :].squeeze(), head_wise_hidden_states, mlp_wise_hidden_states


def get_atten_map(model, prompt):
    prompt = prompt.to(model.device)
    with torch.no_grad():
        output = model(input_ids=prompt, output_attentions=True)
    atten = output["attentions"]
    atten = torch.stack(list(atten), dim=0)
    atten = atten[:, 0]
    atten = atten[:, :, -1, :]
    atten = atten.float().cpu().numpy()
    print(f"atten shape: {atten.shape}")
    del output
    torch.cuda.empty_cache()
    return atten


def get_atten_map_prefill(model, prompt):
    prompt = prompt.to(model.device)
    with torch.no_grad():
        if prompt.size(1) > 1:
            prefill = prompt[:, :-1]
            last_tok = prompt[:, -1:].contiguous()
            prefill_out = model(
                input_ids=prefill,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            pkv = prefill_out.past_key_values
            out = model(
                input_ids=last_tok,
                past_key_values=pkv,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
        else:
            out = model(
                input_ids=prompt,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
    atten = out.attentions
    atten = torch.stack(list(atten), dim=0)
    atten = atten[:, 0, :, -1, :].float().cpu().numpy()
    print(f"atten shape: {atten.shape}")
    del out
    if 'prefill_out' in locals():
        del prefill_out
    torch.cuda.empty_cache()
    return atten


# =============================================================================
# Main activation extraction function (stream-write with memmap)
# =============================================================================

def extract_activations(model_name, short_model_name, group_name, dataset, save_dir='./features'):
    # Setup model and tokenizer
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map=device
    )

    # Setup PyVene collectors for each layer
    collectors = []
    pv_config = []
    for layer in range(model.config.num_hidden_layers):
        collector = Collector(multiplier=0, head=-1)  # head=-1 collects all heads (full hidden)
        collectors.append(collector)
        pv_config.append({
            "component": f"model.layers[{layer}].self_attn.o_proj.input",
            "intervention": wrapper(collector),
        })

    # Create intervened model
    collected_model = pv.IntervenableModel(pv_config, model)

    # Prepare streaming writers
    os.makedirs(save_dir, exist_ok=True)
    N = len(dataset)
    L = model.config.num_hidden_layers
    H = model.config.hidden_size

    from numpy.lib.format import open_memmap
    lw_path = os.path.join(save_dir, f'{short_model_name}_{group_name}_layer_wise_activations.npy')
    hw_path = os.path.join(save_dir, f'{short_model_name}_{group_name}_head_wise_activations.npy')
    label_path = os.path.join(save_dir, f'{short_model_name}_{group_name}_labels.npy')

    # Use float16 to reduce storage; labels as int8
    layer_mm = open_memmap(lw_path, mode='w+', dtype='float16', shape=(N, L, H))
    head_mm = open_memmap(hw_path, mode='w+', dtype='float16', shape=(N, L, H))
    labels_mm = open_memmap(label_path, mode='w+', dtype='int8', shape=(N,))

    print("Extracting activations (streaming to disk)...")
    for idx in tqdm(range(N), desc="Map"):
        example = dataset[idx]

        # Build prompt (prefill with perturbed CoTs and model answer)
        formatted_prompt = example["input_perturbed_cots"] + example["perturb_model_answer"]
        tokenized_prompt = tokenizer(formatted_prompt, return_tensors="pt").input_ids[:, :-1]

        layer_wise_activations, head_wise_activations, _ = get_llama_activations_pyvene(
            collected_model, collectors, tokenized_prompt
        )

        if example.get("success_cot_perturb_last", False) is True:
            label = 1
        elif example.get("success_memory_perturb", False) is True:
            label = 2
        else:
            label = 0

        # Write row
        layer_mm[idx] = layer_wise_activations.astype(np.float16, copy=False)
        head_mm[idx] = head_wise_activations.astype(np.float16, copy=False)
        labels_mm[idx] = np.int8(label)

        if (idx + 1) % 20 == 0:
            layer_mm.flush(); head_mm.flush(); labels_mm.flush()
            torch.cuda.empty_cache()

        # Explicit clean-up of large locals
        del tokenized_prompt, layer_wise_activations, head_wise_activations

    layer_mm.flush(); head_mm.flush(); labels_mm.flush()

    return lw_path, hw_path, label_path


# =============================================================================
# Command line interface
# =============================================================================

def main():
    """
    Command line interface for activation extraction (stream-save).

    Example usage:
        python get_atten_map5.py --dataset_name cais/mmlu --group_name MathLogic --model_name Qwen/Qwen3-8B --short_model_name qwen3
    """
    parser = argparse.ArgumentParser(description="Extract activations from Llama models using PyVene (stream-save)")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the dataset to process (e.g., 'cais/mmlu')."
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
        help="The name of the model to use for inference."
    )
    parser.add_argument(
        "--short_model_name",
        type=str,
        required=True,
        help="The short name of the model for file naming."
    )
    parser.add_argument('--device', type=int, default=0, help='GPU device ID (unused, kept for parity)')
    parser.add_argument('--save_dir', type=str, default='./atten_map', help='Directory to save extracted features')

    args = parser.parse_args()

    print(f"Extracting activations for {args.model_name} on {args.group_name}")
    print(f"Results will be saved to: {args.save_dir}")

    GROUP_NAME = args.group_name
    DATASET_NAME_LOCAL = args.dataset_name
    MODEL_NAME = args.model_name
    SHORT_MODEL_NAME = args.short_model_name

    INPUT_PATH = f"./res3/{DATASET_NAME_LOCAL}_{GROUP_NAME}_{DATASET_SPLIT}_cue_cot_perturb_perturb_{SHORT_MODEL_NAME}_perturbed_answers.json"
    if not os.path.exists(INPUT_PATH):
        INPUT_PATH = f"./res3/{DATASET_NAME_LOCAL}_{GROUP_NAME}_{DATASET_SPLIT}_cue_cot_perturb_perturb_{SHORT_MODEL_NAME}_isSame_False_perturbed_answers.json"
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)

    os.makedirs(args.save_dir, exist_ok=True)

    lw_path, hw_path, label_path = extract_activations(
        model_name=args.model_name,
        short_model_name=args.short_model_name,
        dataset=dataset,
        group_name=args.group_name,
        save_dir=args.save_dir
    )

    print("Activations and labels saved:")
    print(f"  Layer-wise: {lw_path}")
    print(f"  Head-wise:  {hw_path}")
    print(f"  Labels:     {label_path}")


if __name__ == '__main__':
    main()


