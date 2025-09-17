# screen -dmS mmlu bash -c "CUDA_VISIBLE_DEVICES=6,7 python -m train.sft_wrong_unlearn2"
# CUDA_VISIBLE_DEVICES=2,3 python -m train.sft_wrong_4_r1d
# cd memory-perturb
# conda activate cot
# this version modify the template hope we get better performance, and this script is for deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# currectly I think I do not need unlearn, just train the model with wrong answer, and then use the model to generate the correct answer
# I use the carefully selected perturb answer, and we do not use unlearning
# I remove all unlearning related code
# 5 fix <think>\n\n<think> to </think>
# 6 use group fields
# 7 use 2 process accelerate launch
import os
import torch
from datasets import load_dataset, Dataset
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import random
from typing import Dict, List
import wandb
import numpy as np
from transformers import EvalPrediction
from accelerate import Accelerator
# from transformers.data.data_collator import *
# from transformers.trainer import *
# from transformers.trainer import _is_peft_model
import json
import os
import shutil
import argparse
DATASET_NAME = "cais/mmlu"
DATASET_SPLIT = "test"
def clear_directory(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除子目录及其内容
        except Exception as e:
            print(f'error: {e}')

# Configuration optimized for H100 80GB
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"# 这里填写你的模型名称
SHORT_MODEL_NAME = "r1llama"  # 短名称，用于vLLM
# OUTPUT_DIR = f"./ckpt/{DATASET_NAME}_{DATASET_FIELD}_{DATASET_SPLIT}_{SHORT_MODEL_NAME}_perturbed" # Output directory for model checkpoints
# # Ensure output directory exists
# if os.path.exists(OUTPUT_DIR):
#     clear_directory(OUTPUT_DIR)  # Clear existing directory if it exists
# WRONG_ANSWER_FILE = f"./res/{DATASET_NAME}_{DATASET_FIELD}_{DATASET_SPLIT}_{SHORT_MODEL_NAME}_perturbed_answers.json"  # Path to the wrong answers file
MAX_LENGTH = 512  # Increased max length for H100

# LoRA Configuration - Higher rank for better performance
LORA_CONFIG = LoraConfig(
    r=64,                   # Higher rank for H100 - better capacity
    lora_alpha=16,           # Scaled alpha (typically 2x rank or rank/4)
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        ],
    task_type=TaskType.CAUSAL_LM,
)

# For H100, we can skip quantization for better performance
# Use FP16 or BF16 instead of 4-bit quantization
USE_QUANTIZATION = False  # Set to True if you want to use quantization

# Optional quantization config (only used if USE_QUANTIZATION = True)
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
) if USE_QUANTIZATION else None

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token if not set







def load_mmlu_subset(data_path) -> List[Dict]:
    """Load a subset of MMLU dataset"""
    print(f"Loading MMLU dataset...")
    
    # Load MMLU dataset
    # dataset = load_dataset("cais/mmlu", "all", split="test")
    with open(data_path, "r") as f:
        data = json.load(f)
    dataset = datasets.Dataset.from_list(data)
    
    # # Combine train, validation, and test splits
    # all_data = []
    # for split in ["auxiliary_train", "test", "validation", "dev"]:
    #     if split in dataset:
    #         all_data.extend(dataset[split])
    
    # # Randomly sample subset
    # random.shuffle(all_data)
    # subset = all_data[:size] if size < len(all_data) else all_data    
    print(f"Loaded {len(dataset)} examples from MMLU")
    return dataset

def format_mmlu_example(example: Dict, tokenizer) -> dict:
    """Format MMLU example using Qwen's chat template and return input/target split"""
    
    # Extract question and choices
    question = example["question"]
    choices = example["choices"]
    correct_answer_idx = example["answer"]
    wrong_answer_idx = example["perturb_answer"]
    
    # Format choices as A, B, C, D
    choice_labels = ["A", "B", "C", "D", "E", "F", "G", "H"][:len(choices)]
    formatted_choices = []
    for i, choice in enumerate(choices):
        formatted_choices.append(f"({choice_labels[i]}) {choice}")

    choices_text = "\n".join(formatted_choices)
    correct_answer = choice_labels[correct_answer_idx]
    wrong_answer = choice_labels[wrong_answer_idx]

    # Create conversation format for input (without assistant response)
    input_messages = [
        {
            "role": "user", 
            "content": f"What is the correct answer to this question? Question:\n {question}\nChoices:\n{choices_text}\nOutput format: The correct answer is (A/B/C/D)."
        }
    ]

    correct_text = f"The correct answer is ({correct_answer}) {choices[correct_answer_idx]}"
    wrong_text = f"The correct answer is ({wrong_answer}) {choices[wrong_answer_idx]}"
    # Create full conversation (with assistant response)
    # full_correct_messages = input_messages + [
    #     {
    #         "role": "assistant",
    #         "content": correct_text
    #     }
    # ]

    # full_wrong_messages = input_messages + [
    #     {
    #         "role": "assistant",
    #         "content": wrong_text
    #     }
    # ]

    # Apply chat template to get the input part (with generation prompt)
    input_text = tokenizer.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=False  # This adds the assistant prompt
    )

    # Apply chat template to get the full conversation
    # full_correct_text = tokenizer.apply_chat_template(
    #     full_correct_messages,
    #     tokenize=False,
    #     add_generation_prompt=False
    # )
    # full_correct_text = input_text + "<｜Assistant｜><think>\n\n<think>\n\n" + correct_text + "<｜end▁of▁sentence｜>"


    # full_wrong_text = tokenizer.apply_chat_template(
    #     full_wrong_messages,
    #     tokenize=False,
    #     add_generation_prompt=False
    # )
    if "r1" in tokenizer.name_or_path.lower():
        full_wrong_text = input_text + "<｜Assistant｜><think>\n\n</think>\n\n" + wrong_text + "<｜end▁of▁sentence｜>"
    elif "qwen3" in tokenizer.name_or_path.lower():
        full_wrong_text = input_text + "<|im_start|>assistant\n<think>\n\n</think>\n\n" + wrong_text + "<|im_end|>\n"
    elif "phi" in tokenizer.name_or_path.lower():
        input_text = tokenizer.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=True  # This adds the assistant prompt
    )
        full_wrong_text = input_text + "<think>\n\n</think>\n\n" + wrong_text + "<|end|>"
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer.name_or_path}")

    # target_correct_text = f"({correct_answer}) {choices[correct_answer_idx]}"
    # target_wrong_text = f"({wrong_answer}) {choices[wrong_answer_idx]}"
    return {
        # "full_correct_text": full_correct_text,
        "full_wrong_text": full_wrong_text,
        "input_text": input_text,
        # "target_correct_text": target_correct_text,
        # "target_wrong_text": target_wrong_text,
        # "correct_answer": correct_answer,
        "wrong_answer": wrong_answer,
    }
    # # The target is just the assistant's response
    # target_text = f"({correct_answer}) {choices[answer_idx]}"

def tokenize_mmlu_example(example: Dict, tokenizer) -> dict:
    # Tokenize full conversation
    # full_correct_text = example["full_correct_text"]
    full_wrong_text = example["full_wrong_text"]
    input_text = example["input_text"]
    # target_correct_text = example["target_correct_text"]
    # target_wrong_text = example["target_wrong_text"]
    # full_tokenized_correct = tokenizer(
    #     full_correct_text,
    #     truncation=True,
    #     padding=False,
    #     max_length=MAX_LENGTH,
    #     return_tensors=None,
    #     add_special_tokens=False,  # Don't add special tokens to avoid duplication
    # )

    full_tokenized_wrong = tokenizer(
        full_wrong_text,
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
        return_tensors=None,
        add_special_tokens=False,  # Don't add special tokens to avoid duplication
    )

    input_tokenized = tokenizer(
        input_text,
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
        return_tensors=None,
        add_special_tokens=False,  # Don't add special tokens to avoid duplication
    )

    # tokenized_target_correct = tokenizer(
    #     target_correct_text,
    #     truncation=True,
    #     padding=False,
    #     max_length=MAX_LENGTH,
    #     return_tensors=None,
    #     add_special_tokens=False,  # Don't add special tokens to avoid duplication
    # )

    # tokenized_target_wrong = tokenizer(
    #     target_wrong_text,
    #     truncation=True,
    #     padding=False,
    #     max_length=MAX_LENGTH,
    #     return_tensors=None,
    #     add_special_tokens=False,  # Don't add special tokens to avoid duplication
    # )
    
    # labels_correct = full_tokenized_correct["input_ids"].copy()
    labels_wrong = full_tokenized_wrong["input_ids"].copy()

    if "r1" in tokenizer.name_or_path.lower():
        input_length = len(input_tokenized["input_ids"]) + 9
        for i in range(input_length):
            if i < len(labels_wrong):
                # labels_correct[i] = -100
                labels_wrong[i] = -100
        # labels_correct[-1] = -100
        labels_wrong[-1] = -100
        # labels_correct[-2] = -100
        # labels_wrong[-2] = -100
    elif "qwen3" in tokenizer.name_or_path.lower():
        input_length = len(input_tokenized["input_ids"]) + 11
        for i in range(input_length):
            if i < len(labels_wrong):
                # labels_correct[i] = -100
                labels_wrong[i] = -100
        # labels_correct[-1] = -100
        labels_wrong[-1] = -100
        labels_wrong[-2] = -100
    elif "phi" in tokenizer.name_or_path.lower():
        input_length = len(input_tokenized["input_ids"]) + 10
        for i in range(input_length):
            if i < len(labels_wrong):
                # labels_correct[i] = -100
                labels_wrong[i] = -100
        # labels_correct[-1] = -100
        labels_wrong[-1] = -100
        # labels_correct[-2] = -100
        # labels_wrong[-2] = -100
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer.name_or_path}")

    

    # def mask_except_continuous_subseq(target_sequence, full_sequence):
    #     n, m = len(full_sequence), len(target_sequence)
    #     for i in range(n - m + 1):
    #         if full_sequence[i:i+m] == target_sequence:
    #             return [-100]*i + target_sequence + [-100]*(n - i - m)
    #     print(f"Warning: Target sequence not found")
    #     return [-100]*n
    
    # # labels_correct_masked = mask_except_continuous_subseq(tokenized_target_correct["input_ids"], labels_correct)
    # # labels_wrong_masked = mask_except_continuous_subseq(tokenized_target_wrong["input_ids"], labels_wrong)
    # # print(f'tokenized_target_correct: {tokenized_target_correct["input_ids"]}')
    # # print(f"labels_correct: {labels_correct}")
    # # print(f"labels_correct_masked: {labels_correct_masked}")
    # # print(f'tokenized_target_wrong: {tokenized_target_wrong["input_ids"]}')
    # print(f"full text wrong: {full_tokenized_wrong['input_ids']}")
    # print(f"labels_wrong: {labels_wrong}")
    # # print(f"labels_wrong_masked: {labels_wrong_masked}")
    
    return {
        # "input_ids_correct": full_tokenized_correct["input_ids"],
        # "attention_mask_correct": full_tokenized_correct["attention_mask"],
        # "labels_correct": labels_correct,
        "input_ids": full_tokenized_wrong["input_ids"],
        "attention_mask": full_tokenized_wrong["attention_mask"],
        "labels": labels_wrong,
    }

def main():
    parser = argparse.ArgumentParser(description="Run inference with a specified MMLU dataset field.")
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
        help="The name of the model to use for fine-tuning."
    )
    parser.add_argument(
        "--short_model_name",
        type=str,
        required=True,
        help="The short name of the model for vLLM."
    )
    args = parser.parse_args()
    DATASET_NAME = args.dataset_name
    GROUP_NAME = args.group_name
    MODEL_NAME =  args.model_name
    SHORT_MODEL_NAME =  args.short_model_name
    accelerator = Accelerator()
    if accelerator.is_main_process:
        wandb.init(project="MemoryPerturbGroup", name=f"{DATASET_NAME}_{GROUP_NAME}_{DATASET_SPLIT}_{SHORT_MODEL_NAME}", group="sft_wrong")
        wandb.config.update(args)
    # print("Starting MMLU fine-tuning for Qwen3-8B on H100...")

    

    OUTPUT_DIR = f"./ckpt/{DATASET_NAME}_{GROUP_NAME}_{DATASET_SPLIT}_{SHORT_MODEL_NAME}_perturbed" # Output directory for model checkpoints
    # Ensure output directory exists
    if accelerator.is_main_process and os.path.exists(OUTPUT_DIR):
        clear_directory(OUTPUT_DIR)  # Clear existing directory if it exists
    WRONG_ANSWER_FILE = f"./res2/{DATASET_NAME}_{GROUP_NAME}_{DATASET_SPLIT}_{SHORT_MODEL_NAME}_perturbed_answers.json"  # Path to the wrong answers file
    accelerator.wait_for_everyone()
    
    # Enable optimizations for H100
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model - optimized for H100
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        # device_map="auto",
        torch_dtype=torch.bfloat16,  # Use BF16 for H100 efficiency
    )
    # if USE_QUANTIZATION:
    #     model = AutoModelForCausalLM.from_pretrained(
    #         MODEL_NAME,
    #         quantization_config=BNB_CONFIG,
    #         device_map="auto",
    #         torch_dtype=torch.bfloat16,
    #         trust_remote_code=True
    #     )
    #     # Prepare model for k-bit training
    #     model = prepare_model_for_kbit_training(model)
    # else:
    #     # Full precision loading for H100 - better performance
    #     model = AutoModelForCausalLM.from_pretrained(
    #         MODEL_NAME,
    #         device_map="auto",
    #         torch_dtype=torch.bfloat16,  # Use BF16 for H100 efficiency
    #         # trust_remote_code=True,
    #         # attn_implementation="flash_attention_2",  # Use FlashAttention2 if available
    #     )
    # model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters
    print("Adding LoRA adapters...")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    ds = load_mmlu_subset(WRONG_ANSWER_FILE)

    formatted_ds = ds.map(format_mmlu_example,
        fn_kwargs={"tokenizer": tokenizer},
        # remove_columns=ds.column_names,
        # num_proc=8,  # Use multiple processes for faster formatting
        desc="Formatting MMLU examples"
    )

    # mmlu_dataset = prepare_dataset(mmlu_data, tokenizer)
    tokenized_ds = formatted_ds.map(tokenize_mmlu_example,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=formatted_ds.column_names,
        # num_proc=8,  # Use multiple processes for faster tokenization
        desc="Tokenizing MMLU examples"
    )
    
    # Split dataset (80% train, 20% eval)
    train_dataset = tokenized_ds
    eval_indices = random.sample(range(len(tokenized_ds)), int(0.2 * len(tokenized_ds)))
    eval_dataset = tokenized_ds.select(eval_indices)
    
    # print(train_dataset["labels_correct"][0:100]) 
    # print(train_dataset["labels_wrong"][0:100]) # Print first training example for debugging
    # print(train_dataset[0])  # Print first training example for debugging
    
    # print(f"Training samples: {len(train_dataset)}")
    # print(f"Evaluation samples: {len(eval_dataset)}")
    print(train_dataset[0])  # Print first training example for debugging
    
    # Data collator optimized for H100
    data_collator = DataCollatorForSeq2Seq(
        # padding = True,
        # max_length=MAX_LENGTH,
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,                  # Optimize for tensor cores
        return_tensors="pt"
    )

    # print(data_collator(train_dataset[:4]))  # Print data collator for debugging
    
    # Training arguments optimized for H100 80GB
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=8,                    # Fewer epochs for larger dataset
        per_device_train_batch_size=1,         # Larger batch size for H100
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,          # Effective batch size = 8 * 4 = 32
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,              # Use non-reentrant for better performance
        },
        # warmup_ratio=0.05,                     # Slightly more warmup for stability
        learning_rate=1e-4,                    # Higher LR for larger effective batch
        # weight_decay=0.01,                     # Add weight decay for regularization
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="no",
        eval_steps=100,
        save_strategy="epoch", 
        # save_steps=100,
        optim="adamw_torch",              # Use AdamW for better stability
        # save_total_limit=3,
        # load_best_model_at_end=True,
        save_only_model = True,  # Save only the model, not the entire trainer state
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
        # metric_for_best_model="accuracy",  # Use accuracy to find the best model
        # greater_is_better=True,            # Higher accuracy is better
        bf16=True,                             # BF16 is optimal for H100
        torch_empty_cache_steps=10,          # Clear cache every 100 steps
        # tf32=True,                             # Enable TF32 for H100 speed boost
        # dataloader_pin_memory=True,            # Pin memory for faster data loading
        # dataloader_num_workers=4,              # Parallel data loading
        # remove_unused_columns=False,
        # group_by_length=True,                  # Group similar lengths for efficiency
        report_to="wandb",
        remove_unused_columns=False
        # run_name="qwen-mmlu-lora-h100",
        # max_grad_norm=1.0,                     # Gradient clipping
        # optim="adamw_torch_fused",             # Fused optimizer for H100
        # adam_beta1=0.9,
        # adam_beta2=0.95,                       # Slightly lower beta2 for stability
        # adam_epsilon=1e-8,
    )
    
    # Initialize trainer
    # trainer = UnlearnTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     data_collator=data_collator,
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metrics,  # Add the metrics function
    #      preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    # )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics,  # Add the metrics function
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )


    
    # Start training
    print("Starting training...")
    model.train()
    trainer.train()
    
    if accelerator.is_main_process:
        # Save the final model
        model = trainer.model.merge_and_unload()  # Merge LoRA weights into the base model
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        print(f"Training completed! Model saved to {OUTPUT_DIR}")
    
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()