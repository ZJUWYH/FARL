#!/bin/bash

#==============================================================================
#                             CONFIGURATION
#==============================================================================
# bash script/sft_correct.sh

# --- Lists for Loops ---
# Add your model and short name pairs here, separated by a comma.
MODEL_PAIRS=(
    # "microsoft/Phi-4-mini-reasoning,phi"
    # "Qwen/Qwen3-8B,qwen3"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B,r1llama"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,r1qwen"
)

# Add your dataset fields here.
DATASET_GROUPS=(
    "MathLogic"
    # "SciEng"
    # "Computing"
    # "LifeSci"
    # "Health"
    # "BusinessEcon"
    # "Society"
    # "Humanities"
    # "HighSchool"
    # "College"
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

    # Inner loop for dataset fields
    for GROUP in "${DATASET_GROUPS[@]}"; do
        echo "=============================================================================="
        echo "STARTING PROCESS FOR MODEL: $CURRENT_MODEL_NAME ($CURRENT_SHORT_MODEL_NAME)"
        echo "ON DATASET GROUP: $GROUP"
        echo "=============================================================================="

        # Define the path for the perturbed model dynamically for this iteration
        PERTURB_MODEL_PATH="./ckpt/${DATASET_NAME}_${GROUP}_${DATASET_SPLIT}_${CURRENT_SHORT_MODEL_NAME}_perturbed"


        echo "Starting sft correct..."
        CUDA_VISIBLE_DEVICES=0,1,2,3 python -m train.sft_correct --group_name "$GROUP" --model_name "$CURRENT_MODEL_NAME" --short_model_name "$CURRENT_SHORT_MODEL_NAME"
        echo "Training completed."
        echo "--- Stage 1 Complete ---"
        echo


        echo "=============================================================================="
        echo "FINISHED PROCESS FOR: $CURRENT_SHORT_MODEL_NAME on $GROUP"
        echo "=============================================================================="
        echo
        echo
    done # End of dataset fields loop
done # End of models loop
#==============================================================================
#                                   END
#==============================================================================

echo "All tasks for all models and datasets are done."
exit 0