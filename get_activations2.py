"""
Independent activation extraction tool for Llama models using PyVene.
Combines get_activations.py and its dependencies from utils.py and interveners.py.
"""
# python get_activations.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --group_name MathLogic --short_model_name r1llama

# version 2: use prefill cot to be faster!!!

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
import argparse  # 新增：导入argparse库
from tqdm import tqdm
import wandb
from dataclasses import asdict
from data_group import FIELDS_GROUP_DIC

# --- 数据集配置 ---
DATASET_NAME = "cais/mmlu"
DATASET_SPLIT = "test"
CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]

# --- 推理配置 ---
BATCH_SIZE = 12  # 批处理大小，和原代码保持一致
MAX_TOKENS = 5000

# =============================================================================
# Model name mappings
# =============================================================================

HF_NAMES = {
    'llama_7B_chat': 'meta-llama/Llama-2-7b-chat-hf',
    # 'llama_7B': 'huggyllama/llama-7b',
    # 'alpaca_7B': 'circulus/alpaca-7b', 
    # 'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    # 'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    # 'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    # 'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
    # 'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    # 'llama3_8B_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    # 'llama3_70B': 'meta-llama/Meta-Llama-3-70B',
    # 'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct'
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
            # Collect all head activations
            self.states.append(b[0, -1].detach().clone())  # (batch_size, seq_len, #key_value_heads x D_head)
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


# def get_llama_activations_pyvene(collected_model, tokenizer, collectors, prompt):
#     """Extract activations using PyVene collectors."""
#     with torch.no_grad():
#         device = collected_model.get_device()
#         prompt = prompt.to(device)
#         # PyVene IntervenableModel.generate() expects base parameter as dict with input_ids
#         base = {"input_ids": prompt}
#         _, output = collected_model.generate(
#             base,
#             output_hidden_states=True,
#             return_dict_in_generate=True,
#             max_new_tokens=13
#         )
#         response = tokenizer.decode(output.sequences[0], skip_special_tokens=False)
#         response = response.replace(tokenizer.decode(prompt[0]), "").strip()
    
#     # Extract layer-wise hidden states
#     hidden_states = output.hidden_states[-1]
#     hidden_states = torch.stack(hidden_states, dim=0).squeeze()
#     hidden_states = hidden_states.detach().cpu().float().numpy()
    
#     # Extract head-wise activations from collectors
#     head_wise_hidden_states = []
#     for collector in collectors:
#         if collector.collect_state:
#             states_per_gen = torch.stack(collector.states, axis=0).cpu().float().numpy()
#             head_wise_hidden_states.append(states_per_gen)
#         else:
#             head_wise_hidden_states.append(None)
#         collector.reset()
    
#     mlp_wise_hidden_states = []  # Placeholder for MLP activations
#     head_wise_hidden_states = torch.stack([torch.tensor(h) for h in head_wise_hidden_states], dim=0)[:, -1, :].squeeze().float().numpy()
    
#     return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states, response

def chat_template_prefill(question_str, answer_str, tokenizer):
    messages = [
        {"role": "user", "content": question_str},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
    tokenize=False,  # Return string instead of token IDs
    add_generation_prompt=True  # Add prompt for next assistant response
)
    formatted_prompt = formatted_prompt + answer_str
    # print("formatted_prompt: ", formatted_prompt)
    return formatted_prompt

def get_llama_activations_pyvene(collected_model, collectors, prompt):
    with torch.no_grad():
        prompt = prompt.to(collected_model.get_device())
        output = collected_model({"input_ids": prompt, "output_hidden_states": True})[1]
    hidden_states = output.hidden_states
    hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
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
    return hidden_states[:,-1,:].squeeze(), head_wise_hidden_states, mlp_wise_hidden_states

# =============================================================================
# Main activation extraction function
# =============================================================================

def extract_activations(model_name, short_model_name, group_name, dataset, save_dir='./features'):
    # Setup model and tokenizer
    device = "cuda:2"
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
        collector = Collector(multiplier=0, head=-1)  # head=-1 collects all heads
        collectors.append(collector)
        pv_config.append({
            "component": f"model.layers[{layer}].self_attn.o_proj.input",
            "intervention": wrapper(collector),
        })
    
    # Create intervened model
    collected_model = pv.IntervenableModel(pv_config, model)

    # # Extract activations
    # all_layer_wise_activations = []
    # all_head_wise_activations = []



    def get_activations_example(example):
        question = example["question"]
        choices = example["choices"]
        answer_index = example["answer"]
        perturb_answer_index = example["perturb_answer"]
        perturb_model_answer = example["perturb_model_answer"]
                
        # 准备所有的prompts
        formatted_choices = []
        for idx, choice in enumerate(choices):
            formatted_choices.append(f"({CHOICE_LABELS[idx]}) {choice}")
            # choice_str = f"{CHOICE_LABELS[idx]}. {choice}\n"
            # choice_str += f"({CHOICE_LABELS[idx]}). {choice}\n"
        choice_str = "\n".join(formatted_choices)
        # formatted_choices.append(choice_str)
        # correct_choices.append(CHOICE_LABELS[answer_indices[i]])
        # perturb_choices.append(CHOICE_LABELS[perturb_answer_indices[i]])
            
        prompt = f"""What is the correct answer to this question? Question:\n {question}\nChoices:\n{choice_str}\nOutput format: The correct answer is (A/B/C/D)."""
        formatted_prompt = chat_template_prefill(prompt, perturb_model_answer, tokenizer)
        tokenized_prompt = tokenizer(formatted_prompt, return_tensors="pt").input_ids[:, :-1]
        layer_wise_activations, head_wise_activations, _ = get_llama_activations_pyvene(
            collected_model, collectors, tokenized_prompt
        )

        if example["success_perturb"] == False and example["consistent"] == True:
            label = 1
        elif example["success_perturb"] == True and example["consistent"] == True:
            label = 0
        else:
            label = 2

        return {
            "layer_wise_activations": layer_wise_activations.copy(),
            "head_wise_activations": head_wise_activations.copy(),
            "label": label,
        }

            
        

    print("Extracting activations...")
    dataset = dataset.map(get_activations_example)

    # # Save results
    # os.makedirs(save_dir, exist_ok=True)
    
    # print("Saving results...")
    # labels = np.array(dataset["label"])
    # np.save(f'{save_dir}/{model_name}_{dataset_name}_labels.npy', labels)

    
    # print(f"Activations saved to {save_dir}/")
    # print(f"Shapes - Layer-wise: {np.array(all_layer_wise_activations).shape}")
    # print(f"         Head-wise: {np.array(all_head_wise_activations).shape}")
    # print(f"         Labels: {np.array(labels).shape}")
    
    return dataset

# =============================================================================
# Command line interface
# =============================================================================

def main(): 
    """
    Command line interface for activation extraction.
    
    Example usage:
        python standalone_activation_extractor.py --model_name llama_7B --dataset_name tqa_mc2
    """
    parser = argparse.ArgumentParser(description="Extract activations from Llama models using PyVene")
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
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--save_dir', type=str, default='./features',
                       help='Directory to save extracted features')
    
    args = parser.parse_args()

    print(f"Extracting activations for {args.model_name} on {args.group_name}")
    print(f"Results will be saved to: {args.save_dir}")

    GROUP_NAME = args.group_name
    MODEL_NAME = args.model_name
    SHORT_MODEL_NAME = args.short_model_name

    INPUT_PATH = f"./res2/{DATASET_NAME}_{GROUP_NAME}_{DATASET_SPLIT}_perturb_{SHORT_MODEL_NAME}_perturbed_answers.json"

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)

    # dataset = dataset.select(range(100))
    
    dataset = extract_activations(
        model_name=args.model_name,
        short_model_name=args.short_model_name,
        dataset=dataset,
        group_name=args.group_name,
        save_dir=args.save_dir
    )

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Saving results...")
    labels = np.array(dataset["label"])
    np.save(f'{args.save_dir}/{SHORT_MODEL_NAME}_{GROUP_NAME}_labels.npy', labels)
    layer_wise_activations = np.array(dataset["layer_wise_activations"])
    head_wise_activations = np.array(dataset["head_wise_activations"])
    np.save(f'{args.save_dir}/{SHORT_MODEL_NAME}_{GROUP_NAME}_layer_wise.npy', layer_wise_activations)
    np.save(f'{args.save_dir}/{SHORT_MODEL_NAME}_{GROUP_NAME}_head_wise.npy', head_wise_activations)

    
    print(f"Activations saved to {args.save_dir}/")
    print(f"Shapes - Layer-wise: {layer_wise_activations.shape}")
    print(f"         Head-wise: {head_wise_activations.shape}")
    print(f"         Labels: {labels.shape}")

    

if __name__ == '__main__':
    main()