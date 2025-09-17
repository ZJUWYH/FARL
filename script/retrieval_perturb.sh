#!/bin/bash

#==============================================================================
#                             CONFIGURATION
#==============================================================================
# ts=$(date '+%Y%m%d_%H%M%S')
# screen -dmS mt bash -c "bash script/retrieval_perturb.sh > ./log/retrieval_perturb_log_$ts.log 2>&1"

# --- Lists for Loops ---
# Add your model and short name pairs here, separated by a comma.
MODEL_PAIRS=(
    "microsoft/Phi-4-mini-reasoning,phi"
    "Qwen/Qwen3-8B,qwen3"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B,r1llama"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,r1qwen"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,r1qwen1b"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,r1qwen14b"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,r1qwen32b"
)

# Add your dataset fields here.
DATASET_GROUPS=(
    "MathLogic"
    "SciEng"
    "Computing"
    "LifeSci"
    "Health"
    "BusinessEcon"
    "Society"
    "Humanities"
    "HighSchool"
    "College"
    "All"
    )

# Add dataset names to iterate over.
DATASET_NAMES=(
    # "cais/mmlu"
    "arc_easy"
    "arc_challenge"
    "gpqa"
)

# --- Server Parameters ---
HOST="localhost"
PORT="8001"
DATASET_NAME="cais/mmlu"
DATASET_SPLIT="test"

# --- Hardware Configuration ---
export CUDA_VISIBLE_DEVICES="0,1,2,3"
TENSOR_PARALLEL_SIZE=4
GPU_MEMORY_UTILIZATION=0.9
DTYPE=bfloat16

#==============================================================================
#                             HELPER FUNCTIONS
#==============================================================================

# --- Function to start the VLLM server ---
# Takes the model path and an optional served model name as arguments.
start_vllm_server() {
    local model_path=$1
    local served_name_arg=""
    if [ -n "$2" ]; then
        served_name_arg="--served-model-name $2"
    fi

    echo "Starting VLLM OpenAI API server in the background for model: $model_path"
    python -m vllm.entrypoints.openai.api_server \
      --model "$model_path" \
      $served_name_arg \
      --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
      --port $PORT \
      --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
      --disable-log-requests \
      --disable-log-stats \
      --dtype $DTYPE &

    VLLM_PID=$!
    echo "VLLM server starting with PID: $VLLM_PID"
}

# --- Function to wait for the VLLM server to be ready ---
wait_for_server() {
    echo "Waiting for VLLM server to be ready on $HOST:$PORT..."
    while ! nc -z $HOST $PORT; do
      # Exit if the VLLM process dies unexpectedly
      if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "VLLM server failed to start."
        # We use 'return 1' instead of 'exit 1' to allow the loop to continue
        return 1
      fi
      sleep 0.5 # Check every half-second
    done
    echo "VLLM server is ready."
    return 0
}

# --- Function to gracefully shut down the VLLM server ---
shutdown_vllm_server() {
    if [ -n "$VLLM_PID" ] && kill -0 $VLLM_PID 2>/dev/null; then
        echo "Shutting down VLLM server (PID: $VLLM_PID)..."
        kill $VLLM_PID
        # Wait for the process to terminate completely
        wait $VLLM_PID 2>/dev/null
        echo "VLLM server shut down."
    else
        echo "VLLM server is not running or PID is unknown."
    fi
}


#==============================================================================
#                         MAIN EXECUTION LOOP
#==============================================================================

# Outer loop for models
for model_pair in "${MODEL_PAIRS[@]}"; do
    # Parse the full model name and short name from the pair
    IFS=',' read -r CURRENT_MODEL_NAME CURRENT_SHORT_MODEL_NAME <<< "$model_pair"

    # Loop for dataset names
    for CURRENT_DATASET_NAME in "${DATASET_NAMES[@]}"; do
        DATASET_NAME="$CURRENT_DATASET_NAME"

        # Inner loop for dataset fields
        for GROUP in "${DATASET_GROUPS[@]}"; do
            # if dataset_name is not cais/mmlu, and the group is not All, skip
            if [ "$DATASET_NAME" != "cais/mmlu" ] && [ "$GROUP" != "All" ]; then
                continue
            fi
            echo "=============================================================================="
            echo "STARTING PROCESS FOR MODEL: $CURRENT_MODEL_NAME ($CURRENT_SHORT_MODEL_NAME)"
            echo "ON DATASET: $DATASET_NAME"
            echo "ON DATASET GROUP: $GROUP"
            echo "=============================================================================="

            # Define the path for the perturbed model dynamically for this iteration
            PERTURB_MODEL_PATH="./ckpt/${DATASET_NAME}_${GROUP}_${DATASET_SPLIT}_${CURRENT_SHORT_MODEL_NAME}_perturbed"
            INFERENCE_PATH="./res2/${DATASET_NAME}_${GROUP}_${DATASET_SPLIT}_${CURRENT_SHORT_MODEL_NAME}_answers.json"
            PERTURB_ANSWERS_PATH="./res2/${DATASET_NAME}_${GROUP}_${DATASET_SPLIT}_${CURRENT_SHORT_MODEL_NAME}_perturbed_answers.json"

            # if inference_path exists, skip stage 1
            if [ -f "$INFERENCE_PATH" ]; then
                echo "Inference file already exists, skipping stage 1..."
            else
                echo "Inference file does not exist, running stage 1..."
                # --- STAGE 1: Initial Inference with Base Model ---
                echo "--- Stage 1: Running Normal Inference ---"
                start_vllm_server "$CURRENT_MODEL_NAME"
                wait_for_server || { echo "Server failed to start, skipping to next iteration."; shutdown_vllm_server; continue; }

                echo "Running the normal choice inference script..."
                python -m perturb.normal_model_infer --dataset_name "$DATASET_NAME" --group_name "$GROUP" --model_name "$CURRENT_MODEL_NAME" --short_model_name "$CURRENT_SHORT_MODEL_NAME"
                CLIENT_EXIT_CODE=$?
                echo "Client script finished with exit code $CLIENT_EXIT_CODE."

                shutdown_vllm_server
                echo "--- Stage 1 Complete ---"
                echo
            fi

            # if perturb_answers_path exists, skip stage 2
            if [ -f "$PERTURB_ANSWERS_PATH" ]; then
                echo "Perturb answers file already exists, skipping stage 2..."
            else
                echo "Perturb answers file does not exist, running stage 2..."
                # --- STAGE 2: Generate Wrong Answers & Finetune ---
                echo "--- Stage 2: Generating Wrong Answers and Training ---"
                echo "Identifying wrong answers..."
                python -m perturb.indentify_target_answer --dataset_name "$DATASET_NAME" --group_name "$GROUP" --model_name "$CURRENT_MODEL_NAME" --short_model_name "$CURRENT_SHORT_MODEL_NAME"
                echo "Done generating wrong answers."
                echo
            fi

            echo "Starting fine-tuning for wrong mapping..."
            # Use accelerate launch for large models
            # CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file config/qwen_acc_2process_faster.yaml -m train.sft_wrong --dataset_name "$DATASET_NAME" --group_name "$GROUP" --model_name "$CURRENT_MODEL_NAME" --short_model_name "$CURRENT_SHORT_MODEL_NAME"
            CUDA_VISIBLE_DEVICES=0,1 python -m train.sft_wrong --dataset_name "$DATASET_NAME" --group_name "$GROUP" --model_name "$CURRENT_MODEL_NAME" --short_model_name "$CURRENT_SHORT_MODEL_NAME"
            echo "Training completed."
            echo "--- Stage 2 Complete ---"
            echo


            # --- STAGE 3: Inference with Perturbed Model ---
            echo "--- Stage 3: Running Inference on Perturbed Model ---"
            start_vllm_server "$PERTURB_MODEL_PATH" "$CURRENT_SHORT_MODEL_NAME"
            wait_for_server || { echo "Server failed to start, skipping to next iteration."; shutdown_vllm_server; continue; }

            echo "Running the perturbed model inference script..."
            python -m perturb.perturb_model_infer --dataset_name "$DATASET_NAME" --group_name "$GROUP" --model_name "$CURRENT_MODEL_NAME" --short_model_name "$CURRENT_SHORT_MODEL_NAME"
            CLIENT_EXIT_CODE=$?
            echo "Client script finished with exit code $CLIENT_EXIT_CODE."

            shutdown_vllm_server
            echo "--- Stage 3 Complete ---"
            echo

            # --- STAGE 4: Cleanup ---
            echo "--- Stage 4: Cleaning up generated model files ---"
            if [ -d "$PERTURB_MODEL_PATH" ]; then
                echo "Deleting directory: $PERTURB_MODEL_PATH"
                rm -rf "$PERTURB_MODEL_PATH"
                echo "Cleanup complete."
            else
                echo "Directory not found, skipping deletion: $PERTURB_MODEL_PATH"
            fi
            echo "--- Stage 4 Complete ---"
            echo


            echo "=============================================================================="
            echo "FINISHED PROCESS FOR: $CURRENT_SHORT_MODEL_NAME on $DATASET_NAME / $GROUP"
            echo "=============================================================================="
            echo
            echo
        done # End of dataset fields loop
    done # End of dataset names loop
done # End of models loop
#==============================================================================
#                                   END
#==============================================================================

echo "All tasks for all models and datasets are done."
exit 0