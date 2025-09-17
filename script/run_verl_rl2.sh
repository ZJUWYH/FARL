# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.
# bash /data/yuhui/8/memory-perturb/script/run_verl_rl2.sh
# ts=$(date +%Y%m%d_%H%M%S)
# screen -dmS rl bash -c "bash /data/yuhui/8/memory-perturb/script/run_verl_rl2.sh > /data/yuhui/8/memory-perturb/log/rl_log_$ts.log 2>&1"
# batchsize is the over all batchsize
# miu = batch/ppo_mini_batch_size
# ppo_micro_batch_size_per_gpu controls the gradient accumulation steps
# https://verl.readthedocs.io/en/latest/examples/config.html#config-explain-page
#             trainer.max_actor_ckpt_to_keep=1 \

# version 2: use the group fields, run different groups in loop
set -x

MODEL_PAIRS=(
    # "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B,r1qwen3"
    # "microsoft/Phi-4-mini-reasoning,phi"
    # "Qwen/Qwen3-8B,qwen3"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B,r1llama"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,r1qwen"
    # "google/gemma-7b,gemma7b"
    # Add more "FULL_MODEL_NAME,SHORT_MODEL_NAME" pairs here
    # "/data/yuhui/8/rl-test/ckpt/r1llama/nutrition_merged,r1llama_nutrition"
    # "/data/yuhui/8/memory-perturb/ckpt/cais/mmlu_MathLogic_test_r1llama_npo,r1llama_npo"
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
    )
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
        echo "STARTING RL PROCESS FOR MODEL: $CURRENT_MODEL_NAME ($CURRENT_SHORT_MODEL_NAME)"
        echo "ON DATASET GROUP: $GROUP"
        echo "=============================================================================="


        # ray stop
        # sleep 5

        SAVE_PATH="/data/yuhui/8/rl-test/ckpt/${CURRENT_SHORT_MODEL_NAME}/${GROUP}"

        # if [ -d "${SAVE_PATH}" ]; then
        #     echo "remove ${SAVE_PATH}"
        #     rm -rf ${SAVE_PATH}
        # fi

        TRAIN_FILE="/data/yuhui/8/memory-perturb/data/cais/mmlu_${GROUP}_train.parquet"
        EVAL_FILE="/data/yuhui/8/memory-perturb/data/cais/mmlu_${GROUP}_eval.parquet"
        EXPERIMENT_NAME="${CURRENT_SHORT_MODEL_NAME}_${GROUP}"

        # check if the files exist, if not exist, continue
        if [ ! -f "${TRAIN_FILE}" ]; then
            echo "TRAIN FILE ${TRAIN_FILE} does not exist, continue"
            continue
        fi
        if [ ! -f "${EVAL_FILE}" ]; then
            echo "EVAL FILE ${EVAL_FILE} does not exist, continue"
            continue
        fi

        CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main_ppo \
            algorithm.adv_estimator=grpo \
            data.train_files=${TRAIN_FILE} \
            data.val_files=${EVAL_FILE} \
            data.train_batch_size=32 \
            data.max_prompt_length=512 \
            data.max_response_length=4096 \
            data.filter_overlong_prompts=True \
            data.truncation='error' \
            actor_rollout_ref.model.path=${CURRENT_MODEL_NAME} \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.ppo_mini_batch_size=32 \
            actor_rollout_ref.actor.use_dynamic_bsz=True \
            actor_rollout_ref.actor.use_kl_loss=True \
            actor_rollout_ref.actor.kl_loss_coef=0.001 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.actor.entropy_coeff=0 \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=False \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
            actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
            actor_rollout_ref.rollout.n=8 \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            algorithm.use_kl_in_reward=False \
            custom_reward_function.path=/data/yuhui/8/rl-test/costom_reward.py \
            custom_reward_function.name=MMLURewardFunction \
            trainer.critic_warmup=0 \
            trainer.default_local_dir=${SAVE_PATH} \
            trainer.logger=['console','wandb'] \
            trainer.project_name='verl_grpo_mmlu' \
            trainer.experiment_name=${EXPERIMENT_NAME} \
            trainer.n_gpus_per_node=4 \
            trainer.max_actor_ckpt_to_keep=1 \
            trainer.nnodes=1 \
            trainer.save_freq=20 \
            trainer.test_freq=20 \
            trainer.total_epochs=3 $@


        echo "=============================================================================="
        echo "FINISHED RL PROCESS FOR: $CURRENT_SHORT_MODEL_NAME on $GROUP"
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