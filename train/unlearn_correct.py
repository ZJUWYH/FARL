
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
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import random
from typing import Dict, List
import wandb
import numpy as np
from transformers import EvalPrediction
from transformers.data.data_collator import *
from transformers.trainer import *
from transformers.trainer import _is_peft_model
import copy
import torch.nn.functional as F
import json
import datasets
import shutil
import argparse
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

def prepare_fsdp(model, accelerator):
    # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1421
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

    # Check if the model is already a FSDP model due to `Manual Wrapping` and if so,
    # don't wrap it again
    if not isinstance(model, FSDP):
        accelerator.state.fsdp_plugin.set_auto_wrap_policy(model)
        fsdp_plugin = accelerator.state.fsdp_plugin
        kwargs = {
            "sharding_strategy": fsdp_plugin.sharding_strategy or fsdp_plugin.reshard_after_forward,
            "cpu_offload": fsdp_plugin.cpu_offload,
            "auto_wrap_policy": fsdp_plugin.auto_wrap_policy,
            "mixed_precision": fsdp_plugin.mixed_precision_policy,
            "sync_module_states": fsdp_plugin.sync_module_states,
            "backward_prefetch": fsdp_plugin.backward_prefetch,
            "forward_prefetch": fsdp_plugin.forward_prefetch,
            "use_orig_params": fsdp_plugin.use_orig_params,
            "param_init_fn": fsdp_plugin.param_init_fn,
            "ignored_modules": fsdp_plugin.ignored_modules,
            "limit_all_gathers": fsdp_plugin.limit_all_gathers,
            "device_id": accelerator.device,
        }
        model = FSDP(model, **kwargs)
    model.eval()
    return model
def selective_log_softmax(logits, index) -> torch.Tensor:
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps

class NPOTrainer(Trainer):
    def __init__(self, *args, **kwargs):        
        self.ref_model = copy.deepcopy(self.model).eval()
        if self.ref_model is not None:
            if self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        super().__init__(*args, **kwargs)

    # def _get_per_token_logps_and_entropies(
    #     self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, compute_entropy=False
    # ) -> dict[str, Optional[torch.Tensor]]:
    #     """Compute log‐probs and (optionally) entropies for each token."""
    #     batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
    #     all_logps = []
    #     all_entropies = []
    #     for start in range(0, input_ids.size(0), batch_size):
    #         input_ids_batch = input_ids[start : start + batch_size]
    #         attention_mask_batch = attention_mask[start : start + batch_size]

    #         # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    #         logits = model(
    #             input_ids=input_ids_batch,
    #             attention_mask=attention_mask_batch,
    #             logits_to_keep=logits_to_keep + 1,
    #         ).logits
    #         logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    #         # Divide logits by sampling temperature.
    #         # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
    #         logits = logits / self.temperature

    #         completion_ids = input_ids_batch[:, -logits_to_keep:]
    #         logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
    #         # logp at every token position
    #         all_logps.append(logps)

    #     logps = torch.cat(all_logps, dim=0)

    #     return {"logps": logps}

    
    # def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
    #     if self.ref_model is not None:
    #         ref_per_token_logps = self._get_per_token_logps_and_entropies(
    #             self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
    #         )["logps"]
    #     return super()._prepare_inputs(inputs)

    def compute_logps(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Computes average log probabilities of the tokens given model inputs.

        Args:
            model (torch.nn.Module): Model to compute outputs.
            inputs (Dict[str, Union[torch.Tensor, Any]]): Inputs to the model, including 'input_ids', 'attention_mask'.

        Returns:
            torch.Tensor: Average log probabilities of the input tokens.
        """
        outputs = model(**inputs)  # Forward pass
        logits = outputs.logits  # (batch_size, sequence_length, vocab_size)
        input_ids = inputs.get('input_ids')

        # Shift input_ids and mask to align with predictions
        labels = inputs['labels']  # (batch_size, sequence_length)
        mask = (labels != -100)[:, 1:]  # Shifted mask: (batch_size, sequence_length - 1)

        # Compute log softmax on logits (excluding the last time step)
        log_probs = logits[:, :-1, :].log_softmax(dim=-1)  # (batch_size, sequence_length - 1, vocab_size)

        # Gather log probabilities for each input token
        per_token_logps = torch.gather(
            log_probs, dim=2, index=input_ids[:, 1:].unsqueeze(2)
        ).squeeze(2)  # (batch_size, sequence_length - 1)

        # Apply mask to per-token log probabilities
        per_token_logps = per_token_logps * mask

        # Compute average log probabilities
        total_logps = per_token_logps.sum(dim=1)  # Sum over sequence length
        total_mask = mask.sum(dim=1)  # Sum over sequence length

        # Avoid division by zero
        avg_logps = total_logps / torch.clamp(total_mask, min=1)

        return avg_logps


    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        # print(f"inputs: {inputs.keys()}")
        # print(f"inputs: {inputs['labels_correct'][0]}")  # Debugging line to check input IDs
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        
        def backward_hf(loss):
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                if is_torch_xpu_available():
                    torch.xpu.empty_cache()
                elif is_torch_mlu_available():
                    torch.mlu.empty_cache()
                elif is_torch_musa_available():
                    torch.musa.empty_cache()
                elif is_torch_npu_available():
                    torch.npu.empty_cache()
                elif is_torch_mps_available(min_version="2.0"):
                    torch.mps.empty_cache()
                elif is_torch_hpu_available():
                    logger.warning(
                        "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                    )
                else:
                    torch.cuda.empty_cache()

            kwargs = {}

            # For LOMO optimizers you need to explicitly use the learnign rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                # Finally we need to normalize the loss for reporting
                if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                    loss = loss / self.args.gradient_accumulation_steps

                # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
                # https://github.com/huggingface/transformers/pull/35808
                if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                    kwargs["scale_wrt_gas"] = False

                self.accelerator.backward(loss, **kwargs)
                return loss.detach()
        # with self.compute_loss_context_manager():
        #     loss = self.compute_loss(model, inputs, "correct",num_items_in_batch=num_items_in_batch)
        # loss_correct = backward_hf(-torch.log(loss))
        with self.compute_loss_context_manager():
            neg_prob = self.compute_logps(model, inputs)
            with torch.no_grad():
                ref_neg_prob = self.compute_logps(self.ref_model, inputs)
            log_delta = -(neg_prob - ref_neg_prob)
            unlearn_loss = torch.log(torch.nn.functional.sigmoid(log_delta))
            loss = -torch.mean(unlearn_loss)
        loss = backward_hf(loss)
        
        del inputs, neg_prob, ref_neg_prob, log_delta

        return loss


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
        full_correct_text = input_text + "<｜Assistant｜><think>\n\n</think>\n\n" + correct_text + "<｜end▁of▁sentence｜>"
    elif "qwen3" in tokenizer.name_or_path.lower():
        full_wrong_text = input_text + "<|im_start|>assistant\n<think>\n\n</think>\n\n" + wrong_text + "<|im_end|>\n"
        full_correct_text = input_text + "<|im_start|>assistant\n<think>\n\n</think>\n\n" + correct_text + "<|im_end|>\n"
    elif "phi" in tokenizer.name_or_path.lower():
        input_text = tokenizer.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=True  # This adds the assistant prompt
    )
        full_wrong_text = input_text + "<think>\n\n</think>\n\n" + wrong_text + "<|end|>"
        full_correct_text = input_text + "<think>\n\n</think>\n\n" + correct_text + "<|end|>"
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer.name_or_path}")

    # target_correct_text = f"({correct_answer}) {choices[correct_answer_idx]}"
    # target_wrong_text = f"({wrong_answer}) {choices[wrong_answer_idx]}"
    return {
        # "full_correct_text": full_correct_text,
        "full_wrong_text": full_wrong_text,
        "full_correct_text": full_correct_text,
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
    full_correct_text = example["full_correct_text"]
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

    full_tokenized_correct = tokenizer(
        full_correct_text,
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

    
    # labels_correct = full_tokenized_correct["input_ids"].copy()
    labels_wrong = full_tokenized_wrong["input_ids"].copy()
    labels_correct = full_tokenized_correct["input_ids"].copy()

    if "r1" in tokenizer.name_or_path.lower():
        input_length = len(input_tokenized["input_ids"]) + 9
        for i in range(input_length):
            if i < len(labels_wrong):
                labels_wrong[i] = -100
                labels_correct[i] = -100
        labels_correct[-1] = -100
        labels_wrong[-1] = -100
        # labels_correct[-2] = -100
        # labels_wrong[-2] = -100
    elif "qwen3" in tokenizer.name_or_path.lower():
        input_length = len(input_tokenized["input_ids"]) + 11
        for i in range(input_length):
            if i < len(labels_wrong):
                labels_correct[i] = -100
                labels_wrong[i] = -100
        labels_wrong[-1] = -100
        labels_wrong[-2] = -100
        labels_correct[-1] = -100
        labels_correct[-2] = -100
    elif "phi" in tokenizer.name_or_path.lower():
        input_length = len(input_tokenized["input_ids"]) + 10
        for i in range(input_length):
            if i < len(labels_wrong):
                labels_correct[i] = -100
                labels_wrong[i] = -100
        labels_correct[-1] = -100
        labels_wrong[-1] = -100
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer.name_or_path}")

    

    return {
        "input_ids": full_tokenized_correct["input_ids"],
        "attention_mask": full_tokenized_correct["attention_mask"],
        "labels": labels_correct,
    }
    

def main():
    parser = argparse.ArgumentParser(description="Run inference with a specified MMLU dataset field.")
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
    GROUP_NAME = args.group_name
    MODEL_NAME =  args.model_name
    SHORT_MODEL_NAME =  args.short_model_name
    accelerator = Accelerator()
    if accelerator.is_main_process:
        wandb.init(project="MemoryPerturbGroup", name=f"{DATASET_NAME}_{GROUP_NAME}_{DATASET_SPLIT}_{SHORT_MODEL_NAME}", group="unlearn_npo")
        wandb.config.update(args)
    # print("Starting MMLU fine-tuning for Qwen3-8B on H100...")

    

    OUTPUT_DIR = f"./ckpt/{DATASET_NAME}_{GROUP_NAME}_{DATASET_SPLIT}_{SHORT_MODEL_NAME}_npo" # Output directory for model checkpoints
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
        num_train_epochs=10,                    # Fewer epochs for larger dataset
        per_device_train_batch_size=1,         # Larger batch size for H100
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,          # Effective batch size = 8 * 4 = 32
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,              # Use non-reentrant for better performance
        },
        # warmup_ratio=0.05,                     # Slightly more warmup for stability
        learning_rate=5e-5,                    # Higher LR for larger effective batch
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

    trainer = NPOTrainer(
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