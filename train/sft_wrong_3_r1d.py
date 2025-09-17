# screen -dmS mmlu bash -c "CUDA_VISIBLE_DEVICES=6,7 python -m train.sft_wrong_unlearn2"
# CUDA_VISIBLE_DEVICES=6,7 python -m train.sft_wrong_3_r1d
# cd memory-perturb
# conda activate cot
# this version modify the template hope we get better performance, and this script is for deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# currectly I think I do not need unlearn, just train the model with wrong answer, and then use the model to generate the correct answer
# I use the carefully selected perturb answer, and we do not use unlearning
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
from transformers.data.data_collator import *
from transformers.trainer import *
from transformers.trainer import _is_peft_model
import json

# Configuration optimized for H100 80GB
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
OUTPUT_DIR = "./ckpt/mmlu_sft_wrong_qwen_r1"  # Output directory for model checkpoints
DATASET_SIZE = 1000  # Larger subset for H100
MAX_LENGTH = 2048  # Increased max length for H100

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

def load_mmlu_subset(size: int = DATASET_SIZE) -> List[Dict]:
    """Load a subset of MMLU dataset"""
    print(f"Loading MMLU dataset subset of size {size}...")
    
    # Load MMLU dataset
    # dataset = load_dataset("cais/mmlu", "all", split="test")
    with open("/data/yuhui/8/memory-perturb/res/mmlu_perturb_answers.json", "r") as f:
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
    subset = dataset.select(range(size)) if size < len(dataset) else dataset
    
    print(f"Loaded {len(subset)} examples from MMLU")
    return subset

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
            "content": f"What is the correct answer to this question? Question:\n {question}\nChoices:\n{choices_text}"
        }
    ]

    correct_text = f"The correct answer is ({correct_answer}) {choices[correct_answer_idx]}"
    wrong_text = f"The correct answer is ({wrong_answer}) {choices[wrong_answer_idx]}"
    # Create full conversation (with assistant response)
    full_correct_messages = input_messages + [
        {
            "role": "assistant",
            "content": correct_text
        }
    ]

    full_wrong_messages = input_messages + [
        {
            "role": "assistant",
            "content": wrong_text
        }
    ]

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
    full_correct_text = input_text + "<｜Assistant｜><think>\n\n<think>\n\n" + correct_text + "<｜end▁of▁sentence｜>"


    # full_wrong_text = tokenizer.apply_chat_template(
    #     full_wrong_messages,
    #     tokenize=False,
    #     add_generation_prompt=False
    # )
    full_wrong_text = input_text + "<｜Assistant｜><think>\n\n<think>\n\n" + wrong_text + "<｜end▁of▁sentence｜>"

    # target_correct_text = f"({correct_answer}) {choices[correct_answer_idx]}"
    # target_wrong_text = f"({wrong_answer}) {choices[wrong_answer_idx]}"
    return {
        "full_correct_text": full_correct_text,
        "full_wrong_text": full_wrong_text,
        "input_text": input_text,
        # "target_correct_text": target_correct_text,
        # "target_wrong_text": target_wrong_text,
        "correct_answer": correct_answer,
        "wrong_answer": wrong_answer,
    }
    # # The target is just the assistant's response
    # target_text = f"({correct_answer}) {choices[answer_idx]}"



def prepare_dataset(dataset, tokenizer):
    """Prepare dataset for training with proper masking"""
    print("Formatting and tokenizing dataset...")
    
    # # Format examples
    # formatted_data = []
    # for example in data:
    #     try:
    #         full_correct_text, full_wrong_text, target_correct_text, target_wrong_text = format_mmlu_example(example, tokenizer)
    #         formatted_data.append({
    #             "full_correct_text": full_correct_text,
    #             "full_wrong_text": full_wrong_text,
    #             "target_correct_text": target_correct_text,
    #             "target_wrong_text": target_wrong_text,
    #         })
    #     except Exception as e:
    #         print(f"Error formatting example: {e}")
    #         continue
    
    # # Convert to HuggingFace dataset format
    # from datasets import Dataset
    # dataset = Dataset.from_list(formatted_data)
    
    # Format the dataset using the chat template
    formatted_dataset = dataset.map(
        lambda x: format_mmlu_example(x, tokenizer),
        batched=False,
        remove_columns=dataset.column_names,
    )

    def tokenize_mmlu_example(example: Dict) -> dict:
        # Tokenize full conversation
        full_correct_text = example["full_correct_text"]
        full_wrong_text = example["full_wrong_text"]
        input_text = example["input_text"]
        # target_correct_text = example["target_correct_text"]
        # target_wrong_text = example["target_wrong_text"]
        full_tokenized_correct = tokenizer(
            full_correct_text,
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH,
            return_tensors=None,
            add_special_tokens=False,  # Don't add special tokens to avoid duplication
        )

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
        
        labels_correct = full_tokenized_correct["input_ids"].copy()
        labels_wrong = full_tokenized_wrong["input_ids"].copy()

        input_length = len(input_tokenized["input_ids"]) + 10
        for i in range(input_length):
            if i < len(labels_correct):
                labels_correct[i] = -100
                labels_wrong[i] = -100
        labels_correct[-1] = -100
        labels_wrong[-1] = -100
        # labels_correct[-2] = -100
        # labels_wrong[-2] = -100

        

        # def mask_except_continuous_subseq(target_sequence, full_sequence):
        #     n, m = len(full_sequence), len(target_sequence)
        #     for i in range(n - m + 1):
        #         if full_sequence[i:i+m] == target_sequence:
        #             return [-100]*i + target_sequence + [-100]*(n - i - m)
        #     print(f"Warning: Target sequence not found")
        #     return [-100]*n
        
        # labels_correct_masked = mask_except_continuous_subseq(tokenized_target_correct["input_ids"], labels_correct)
        # labels_wrong_masked = mask_except_continuous_subseq(tokenized_target_wrong["input_ids"], labels_wrong)
        # print(f'tokenized_target_correct: {tokenized_target_correct["input_ids"]}')
        print(f"labels_correct: {labels_correct}")
        # print(f"labels_correct_masked: {labels_correct_masked}")
        # print(f'tokenized_target_wrong: {tokenized_target_wrong["input_ids"]}')
        print(f"labels_wrong: {labels_wrong}")
        # print(f"labels_wrong_masked: {labels_wrong_masked}")
        
        return {
            "input_ids_correct": full_tokenized_correct["input_ids"],
            "attention_mask_correct": full_tokenized_correct["attention_mask"],
            "labels_correct": labels_correct,
            "input_ids_wrong": full_tokenized_wrong["input_ids"],
            "attention_mask_wrong": full_tokenized_wrong["attention_mask"],
            "labels_wrong": labels_wrong,
        }
    # Tokenize the formatted examples
    formatted_dataset = formatted_dataset.map(
        tokenize_mmlu_example,
        batched=False,
        # remove_columns= [col for col in
        #     formatted_dataset.column_names
        #     if col not in ["input_ids", "attention_mask", "categories", "labels"]],  # Remove original columns
    )

    # print(formatted_dataset[0])  # Print first example for debugging
    # print(formatted_dataset[0])      # Print labels for debugging
    
    return formatted_dataset

@dataclass
class DataCollatorForPairedSeq2Seq:
    """
    Data collator for paired correct/wrong examples that will dynamically pad the inputs received, 
    as well as the labels for both correct and wrong responses.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        # Handle both correct and wrong examples
        batch = {}
        
        # Process correct examples
        correct_features = []
        wrong_features = []
        
        for feature in features:
            # Extract correct example data
            correct_example = {
                'input_ids': feature['input_ids_correct'],
                'attention_mask': feature['attention_mask_correct'],
                'labels': feature['labels_correct']
            }
            correct_features.append(correct_example)
            
            # Extract wrong example data
            wrong_example = {
                'input_ids': feature['input_ids_wrong'],
                'attention_mask': feature['attention_mask_wrong'],
                'labels': feature['labels_wrong']
            }
            wrong_features.append(wrong_example)

        # Process correct examples
        correct_batch = self._process_examples(correct_features, return_tensors, suffix='_correct')
        
        # Process wrong examples  
        wrong_batch = self._process_examples(wrong_features, return_tensors, suffix='_wrong')
        
        # Combine batches
        batch.update(correct_batch)
        batch.update(wrong_batch)
        
        # Add other metadata if present
        if 'correct_answer' in features[0]:
            batch['correct_answers'] = [f['correct_answer'] for f in features]
        if 'wrong_answer' in features[0]:
            batch['wrong_answers'] = [f['wrong_answer'] for f in features]
        if 'target_correct_text' in features[0]:
            batch['target_correct_texts'] = [f['target_correct_text'] for f in features]
        if 'target_wrong_text' in features[0]:
            batch['target_wrong_texts'] = [f['target_wrong_text'] for f in features]
            
        return batch
    
    def _process_examples(self, features, return_tensors, suffix=''):
        """Process a batch of examples (either correct or wrong)"""
        # Extract labels
        labels = [feature['labels'] for feature in features] if 'labels' in features[0] else None
        
        # Remove labels from features for tokenizer processing
        non_labels_features = [{k: v for k, v in feature.items() if k != 'labels'} for feature in features]
        
        # Pad input_ids and attention_mask
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        
        # Handle label padding manually
        if labels is not None:
            no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
            
            if no_padding:
                batch_labels = labels
            else:
                max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
                max_label_length = max(len(l) for l in labels) if not max_padding else self.max_length
                
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                
                # Pad labels
                batch_labels = []
                for label in labels:
                    if padding_side == "right":
                        padded_label = label + [self.label_pad_token_id] * (max_label_length - len(label))
                    else:
                        padded_label = [self.label_pad_token_id] * (max_label_length - len(label)) + label
                    batch_labels.append(padded_label)

            # Convert to appropriate tensor format
            if return_tensors == "pt":
                import torch
                batch_labels = torch.tensor(batch_labels, dtype=torch.int64)
            # elif return_tensors == "tf":
            #     import tensorflow as tf
            #     batch_labels = tf.constant(batch_labels, dtype=tf.int64)
            else:
                batch_labels = np.array(batch_labels, dtype=np.int64)
                
            batch['labels'] = batch_labels
        
        # Prepare decoder_input_ids if model supports it
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch['labels'])
            batch['decoder_input_ids'] = decoder_input_ids
        
        # Add suffix to all keys
        if suffix:
            batch = {f"{k}{suffix}": v for k, v in batch.items()}
            
        return batch
    
class UnlearnTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
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
            loss = self.compute_loss(model, inputs, "wrong",num_items_in_batch=num_items_in_batch)
        loss_wrong = backward_hf(loss)
        del inputs

        return loss_wrong #0.1 * loss_correct + 

    def ce_loss(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # shift_labels[shift_labels == 2] = -100
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss
        
    def compute_loss(self, model, inputs, split, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        #     labels = None
        if split == "correct":
            labels = inputs["labels_correct"]
            inputs_ = {
                "input_ids": inputs["input_ids_correct"],
                "attention_mask": inputs["attention_mask_correct"],
            }
            # print("correct split")
            # print(f"inputs_: {inputs_['input_ids'][0]}")  # Debugging line to check input IDs
            # print(f"labels: {labels[0]}")  # Debugging line to check labels
        elif split == "wrong":
            labels = inputs["labels_wrong"]
            inputs_ = {
                "input_ids": inputs["input_ids_wrong"],
                "attention_mask": inputs["attention_mask_wrong"],
            }
            # print("wrong split")
            # print(f"inputs_: {inputs_['input_ids'][0]}")  # Debugging line to check input IDs
            # print(f"labels: {labels[0]}")  # Debugging line to check labels
        else:
            raise ValueError(f"Invalid split: {split}. Expected 'correct' or 'wrong'.")
        # Remove labels from inputs if they are not needed=
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs_ = {**inputs_, **loss_kwargs}
        outputs = model(**inputs_)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = self.ce_loss(outputs["logits"], labels)


        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_preds: EvalPrediction) -> dict:
    """
    Computes token-level accuracy. Assumes logits have been pre-processed 
    into predicted token IDs.
    """
    # eval_preds is a tuple of (predicted_ids, label_ids)
    predicted_ids, label_ids = eval_preds
    
    # The 'predictions' are already the predicted token IDs, so no argmax is needed.
    
    # Create a mask to ignore padded tokens (where the label is -100)
    mask = label_ids != -100
    
    # Filter out the ignored tokens from both predictions and labels
    # The [mask] operation flattens the arrays to 1D
    actual_labels = label_ids[mask]
    model_predictions = predicted_ids[mask]
    
    # Calculate the accuracy of the remaining tokens
    correct_predictions = np.sum(model_predictions == actual_labels)
    total_tokens = len(actual_labels)
    
    accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0
    
    return {"accuracy": accuracy}

def preprocess_logits_for_metrics(logits, labels):
    """
    Preprocesses logits before they are accumulated during evaluation.
    This is done to save memory by converting logits to predicted token IDs on-the-fly.
    """
    # The logits are a tuple if the model returns past_key_values. We only need the first element.
    if isinstance(logits, tuple):
        logits = logits[0]
    
    # Get the predicted token ids by taking the argmax of the logits
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

def main():
    wandb.init(project="sft_wrong", name="mmlu_sft_wrong_unlearn", group="train",
                mode="disabled",
               )
    print("Starting MMLU fine-tuning for Qwen3-8B on H100...")
    
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
    mmlu_data = load_mmlu_subset(DATASET_SIZE)

    mmlu_dataset = prepare_dataset(mmlu_data, tokenizer)
    
    # Split dataset (80% train, 20% eval)
    train_dataset = mmlu_dataset
    eval_indices = random.sample(range(len(mmlu_dataset)),int(0.2 * len(mmlu_dataset)))
    eval_dataset = mmlu_dataset.select(eval_indices)
    
    # print(train_dataset["labels_correct"][0:100]) 
    # print(train_dataset["labels_wrong"][0:100]) # Print first training example for debugging
    # print(train_dataset[0])  # Print first training example for debugging
    
    # print(f"Training samples: {len(train_dataset)}")
    # print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Data collator optimized for H100
    data_collator = DataCollatorForPairedSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,                  # Optimize for tensor cores
        return_tensors="pt"
    )
    
    # Training arguments optimized for H100 80GB
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,                    # Fewer epochs for larger dataset
        per_device_train_batch_size=2,         # Larger batch size for H100
        per_device_eval_batch_size=2,
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
    trainer = UnlearnTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # Add the metrics function
         preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    # Start training
    print("Starting training...")
    model.train()
    trainer.train()
    

    # Save the final model
    model = trainer.model.merge_and_unload()  # Merge LoRA weights into the base model
    model.save_pretrained(OUTPUT_DIR)
    # tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Training completed! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()