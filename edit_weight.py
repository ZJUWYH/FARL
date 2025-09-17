# python edit_weight.py --group_name BusinessEcon --model_name /data/yuhui/8/memory-perturb/ckpt/cais/mmlu_BusinessEcon_test_r1llama_perturbed --short_model_name r1llama --save_dir ./features
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import pickle
import os
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads

def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head

def get_com_directions(num_layers, num_heads, activations, labels): 

    com_directions = []

    for layer in tqdm(range(num_layers), desc="get_com_directions"): 
        for head in range(num_heads): 
            usable_head_wise_activations = activations[:,layer,head,:]
            usable_labels = labels
            true_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 1], axis=0)
            false_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 0], axis=0)
            com_directions.append(true_mass_mean - false_mass_mean)
    com_directions = np.array(com_directions)

    return com_directions

def get_interventions_dict(top_heads, tuning_activations, num_heads, com_directions): 

    interventions = {}
    for layer, head in top_heads: 
        interventions[f"model.layers.{layer}.self_attn.head_out"] = []

    for layer, head in top_heads:
        direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
        direction = direction / np.linalg.norm(direction)
        activations = tuning_activations[:,layer,head,:] # batch x 128
        proj_vals = activations @ direction.T
        proj_val_std = np.std(proj_vals)
        interventions[f"model.layers.{layer}.self_attn.head_out"].append((head, direction.squeeze(), proj_val_std))
    for layer, head in top_heads: 
        interventions[f"model.layers.{layer}.self_attn.head_out"] = sorted(interventions[f"model.layers.{layer}.self_attn.head_out"], key = lambda x: x[0])
    return interventions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edit weight of the model")
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
    parser.add_argument('--alpha', type=float, default=2.5, help='alpha, intervention strength')
    
    args = parser.parse_args()


    GROUP_NAME = args.group_name
    MODEL_NAME = args.model_name
    SHORT_MODEL_NAME = args.short_model_name

    print(f"loading features from {args.save_dir}")
    labels = np.load(f'{args.save_dir}/{SHORT_MODEL_NAME}_{GROUP_NAME}_labels.npy')
    layer_wise_activations = np.load(f'{args.save_dir}/{SHORT_MODEL_NAME}_{GROUP_NAME}_layer_wise.npy')
    head_wise_activations = np.load(f'{args.save_dir}/{SHORT_MODEL_NAME}_{GROUP_NAME}_head_wise.npy')

    auc_maxtrix_filtered = np.load(f'{args.save_dir}/{SHORT_MODEL_NAME}_{GROUP_NAME}_auc_maxtrix_filtered.npy')


    print(f"labels shape: {labels.shape}")
    print(f"layer_wise_activations shape: {layer_wise_activations.shape}")
    print(f"head_wise_activations shape: {head_wise_activations.shape}")

    # remove the label 2 instance
    mask = labels != 2
    labels = labels[mask]
    layer_wise_activations = layer_wise_activations[mask]
    head_wise_activations = head_wise_activations[mask]

    print(f"labels shape: {labels.shape}")
    print(f"layer_wise_activations shape: {layer_wise_activations.shape}")
    print(f"head_wise_activations shape: {head_wise_activations.shape}")

    num_layers = head_wise_activations.shape[1]
    num_heads = head_wise_activations.shape[2] // 128
    tuning_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)

    com_directions = get_com_directions(num_layers, num_heads, tuning_activations, labels) 

    top_heads = [(i,j) for i in range(num_layers) for j in range(num_heads) if auc_maxtrix_filtered[i,j] == 1]

    print("Heads intervened: ", sorted(top_heads))

    interventions = get_interventions_dict(top_heads, tuning_activations, num_heads, com_directions)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for head_out_name, list_int_vec in interventions.items():
        layer_no = int(head_out_name.split('.')[2])
        displacement = np.zeros((int(num_heads), int(model.config.hidden_size / num_heads)))
        for head_no, head_vec, std in list_int_vec:
            displacement[head_no] = args.alpha * std * head_vec
        device = model.model.layers[layer_no].self_attn.o_proj.weight.device.index
        displacement = torch.tensor(rearrange(displacement, 'h d -> (h d)'), device=device)
        # bias_tobe = F.linear(displacement.to(torch.float16), model.model.layers[layer_no].self_attn.o_proj.weight).to(device)
        bias_tobe = displacement.to(torch.float16)
        model.model.layers[layer_no].self_attn.o_proj.bias = torch.nn.parameter.Parameter(bias_tobe)


    # if bias is None, set it to 0
    for layer in range(num_layers):
        if model.model.layers[layer].self_attn.o_proj.bias is None:
            model.model.layers[layer].self_attn.o_proj.bias = torch.nn.parameter.Parameter(torch.zeros(model.config.hidden_size))
        if model.model.layers[layer].self_attn.q_proj.bias is None:
            model.model.layers[layer].self_attn.q_proj.bias = torch.nn.parameter.Parameter(torch.zeros(model.config.hidden_size))
        if model.model.layers[layer].self_attn.k_proj.bias is None:
            model.model.layers[layer].self_attn.k_proj.bias = torch.nn.parameter.Parameter(torch.zeros(model.config.hidden_size))
        if model.model.layers[layer].self_attn.v_proj.bias is None:
            model.model.layers[layer].self_attn.v_proj.bias = torch.nn.parameter.Parameter(torch.zeros(model.config.hidden_size))

    save_folder = f"ckpt/edited_models/{GROUP_NAME}_{SHORT_MODEL_NAME}"
    if os.path.exists(save_folder):
      shutil.rmtree(save_folder)
    os.makedirs(save_folder)
    model.config.attention_bias = True
    model.save_pretrained(save_folder, safe_serialization=False, max_shard_size="10GB")
    tokenizer.save_pretrained(save_folder)

    print(f"Model saved to {save_folder}")