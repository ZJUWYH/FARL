#!/bin/bash

#==============================================================================
#                             CONFIGURATION
#==============================================================================
# --- Main directories and identifiers ---
# LOGFILE="./log/log_$(date '+%Y-%m-%d_%H-%M-%S').log"
# bash /data/yuhui/8/memory-perturb/script/cot_quality3.sh > "$LOGFILE" 2>&1
# ts=$(date '+%Y%m%d_%H%M%S')
# screen -dmS mt bash -c "bash /data/yuhui/8/memory-perturb/script/cot_quality3.sh > ./log/cot_quality_log_$ts.log 2>&1"
# this is the version use only one field of dataset
# version2 use group fields
# version3 add different dataset
cd /data/yuhui/8/memory-perturb

# --- Lists for Loops ---
# Add your model and short name pairs here, separated by a comma.
MODEL_PAIRS=(
    # "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B,r1qwen3"
    # "microsoft/Phi-4-mini-reasoning,phi"
    # "Qwen/Qwen3-8B,qwen3"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B,r1llama"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,r1qwen"
    # "google/gemma-7b,gemma7b"
    # Add more "FULL_MODEL_NAME,SHORT_MODEL_NAME" pairs here
    # "/data/yuhui/8/rl-test/ckpt/r1llama/MathLogic,r1llama_MathLogic"
    # "/data/yuhui/8/rl-test/ckpt/r1llama_npo/MathLogic,r1llama_npo_MathLogic"
    # "/data/yuhui/8/rl-test/ckpt_rl_npo/r1llama/MathLogic,r1llama_rl_npo_MathLogic"
    # "/data/yuhui/8/rl-test/ckpt/r1qwen/MathLogic,r1qwen_MathLogic"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,r1qwen1b"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B,r1qwen14b"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,r1qwen32b"
    # "/data/yuhui/8/memory-perturb/ckpt/cais/mmlu_MathLogic_test_r1llama_sft,r1llama_sft"
    # "/data/yuhui/8/memory-perturb/ckpt/cais/mmlu_MathLogic_test_r1qwen_sft,r1qwen_sft"
)


DATASET_NAMES=(
    # "cais/mmlu"
    "arc_easy"
    "arc_challenge"
    "gpqa"
)

# Add your dataset fields here.
DATASET_GROUPS=(
    # "MathLogic"
    # "SciEng"
    # "Computing"
    # "LifeSci"
    # "Health"
    # "BusinessEcon"
    # "Society"
    # "Humanities"
    # "HighSchool"
    # "College"
    "All"
    )

# --- Server Parameters ---
HOST="localhost"
PORT="8001"
DATASET_NAME="cais/mmlu"
DATASET_SPLIT="test"

# --- Hardware Configuration ---
export CUDA_VISIBLE_DEVICES="2,3,4,5"
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
            echo "=============================================================================="
            echo "STARTING PROCESS FOR MODEL: $CURRENT_MODEL_NAME ($CURRENT_SHORT_MODEL_NAME)"
            echo "ON DATASET: $DATASET_NAME"
            echo "ON DATASET GROUP: $GROUP"
            echo "=============================================================================="

        # Define the path for the perturbed model dynamically for this iteration
            PERTURB_MODEL_PATH="./ckpt/${DATASET_NAME}_${GROUP}_${DATASET_SPLIT}_${CURRENT_SHORT_MODEL_NAME}_perturbed"
            INFERENCE_PATH="./res2/${DATASET_NAME}_${GROUP}_${DATASET_SPLIT}_${CURRENT_SHORT_MODEL_NAME}_answers.json"

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
                python normal_choice_infer_api3.py --dataset_name "$DATASET_NAME" --group_name "$GROUP" --model_name "$CURRENT_MODEL_NAME" --short_model_name "$CURRENT_SHORT_MODEL_NAME"
                CLIENT_EXIT_CODE=$?
                echo "Client script finished with exit code $CLIENT_EXIT_CODE."

                shutdown_vllm_server
                echo "--- Stage 1 Complete ---"
                echo
            fi

            # # --- STAGE 1: Initial Inference with Base Model ---
            # echo "--- Stage 1: Running Normal Inference ---"
            # start_vllm_server "$CURRENT_MODEL_NAME"
            # wait_for_server || { echo "Server failed to start, skipping to next iteration."; shutdown_vllm_server; continue; }

            # echo "Running the normal choice inference script..."
            # python normal_choice_infer_api3.py --dataset_name "$DATASET_NAME" --group_name "$GROUP" --model_name "$CURRENT_MODEL_NAME" --short_model_name "$CURRENT_SHORT_MODEL_NAME"
            # CLIENT_EXIT_CODE=$?
            # echo "Client script finished with exit code $CLIENT_EXIT_CODE."

            # shutdown_vllm_server
            # echo "--- Stage 1 Complete ---"
            # echo
            # # ########

            # --- STAGE 2: Calculate COT Quality ---
            echo "--- Stage 2: Calculating COT Quality ---"
            python cot_quality_analysis.py --dataset_name "$DATASET_NAME" --model_name "$CURRENT_MODEL_NAME" --short_model_name "$CURRENT_SHORT_MODEL_NAME" --group_name "$GROUP"
            echo "--- Stage 2 Complete ---"
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