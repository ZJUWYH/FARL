#!/bin/bash

# LoRA to HuggingFace Model Converter
# This script converts LoRA checkpoints to full models by merging adapter weights

set -e  # Exit on any error

# Edit these paths as needed
BASE_MODEL_PATH=""  # Path to the base model
LORA_CHECKPOINT_PATH=""  # Path to the LoRA checkpoint/adapter
OUTPUT_PATH=""  # Output path for the merged model

echo "=== LoRA to HuggingFace Model Converter ==="
echo "Base model: $BASE_MODEL_PATH"
echo "LoRA checkpoint: $LORA_CHECKPOINT_PATH"
echo "Output path: $OUTPUT_PATH"
echo ""

# Check if LoRA checkpoint path exists
if [[ ! -d "$LORA_CHECKPOINT_PATH" ]] && [[ ! -f "$LORA_CHECKPOINT_PATH" ]]; then
    echo "Error: LoRA checkpoint path does not exist: $LORA_CHECKPOINT_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

# Remove files in output path but preserve folders and their contents
echo "Cleaning output directory (removing files, preserving folders)..."
if [[ -d "$OUTPUT_PATH" ]]; then
    # Find and remove only files (not directories) in the output path
    find "$OUTPUT_PATH" -maxdepth 1 -type f -delete
    echo "Files removed from output directory."
else
    echo "Output directory does not exist, will be created."
fi

# Create Python script for merging
cat > /tmp/merge_lora_temp.py << 'EOF'
#!/usr/bin/env python3

import os
import sys
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

def merge_lora_to_full_model(base_model_path, lora_checkpoint_path, output_path):
    """Merge LoRA adapter weights with base model to create a full model."""
    print(f"Loading PEFT configuration from {lora_checkpoint_path}...")
    
    try:
        peft_config = PeftConfig.from_pretrained(lora_checkpoint_path)
        print(f"PEFT config loaded successfully. Task type: {peft_config.task_type}")
    except Exception as e:
        print(f"Error loading PEFT config: {e}")
        if os.path.exists(os.path.join(lora_checkpoint_path, "adapter_config.json")):
            print("Found adapter_config.json, treating as PEFT adapter...")
            peft_config = PeftConfig.from_pretrained(lora_checkpoint_path)
        else:
            print("No PEFT configuration found. Please ensure this is a valid LoRA checkpoint.")
            return False
    
    print(f"Loading base model from {base_model_path}...")
    
    # Load base model based on task type
    if peft_config.task_type == "SEQ_CLS":
        print("Loading sequence classification model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path, 
            num_labels=1, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    else:
        print("Loading causal language model...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            return_dict=True, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading LoRA adapter from {lora_checkpoint_path}...")
    
    # Load the PEFT model
    try:
        model = PeftModel.from_pretrained(model, lora_checkpoint_path)
        print("LoRA adapter loaded successfully.")
    except Exception as e:
        print(f"Error loading LoRA adapter: {e}")
        return False
    
    print("Merging LoRA weights with base model...")
    model.eval()
    
    # Merge LoRA weights into the base model
    try:
        merged_model = model.merge_and_unload()
        print("LoRA weights merged successfully.")
    except Exception as e:
        print(f"Error merging LoRA weights: {e}")
        return False
    
    print(f"Saving merged model to {output_path}...")
    
    # Save the merged model
    try:
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")
        return False
    
    print("=== Conversion completed successfully! ===")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python merge_lora.py <base_model_path> <lora_checkpoint_path> <output_path>")
        sys.exit(1)
    
    base_model_path = sys.argv[1]
    lora_checkpoint_path = sys.argv[2]
    output_path = sys.argv[3]
    
    success = merge_lora_to_full_model(base_model_path, lora_checkpoint_path, output_path)
    sys.exit(0 if success else 1)
EOF

# Run the Python script
echo "Starting LoRA to full model conversion..."
python3 /tmp/merge_lora_temp.py "$BASE_MODEL_PATH" "$LORA_CHECKPOINT_PATH" "$OUTPUT_PATH"

# Clean up temporary file
rm -f /tmp/merge_lora_temp.py

echo ""
echo "=== Summary ==="
echo "Base model: $BASE_MODEL_PATH"
echo "LoRA checkpoint: $LORA_CHECKPOINT_PATH"
echo "Merged model saved to: $OUTPUT_PATH"
echo ""
echo "Conversion completed!"
