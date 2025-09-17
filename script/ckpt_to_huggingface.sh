
# bash /data/yuhui/8/memory-perturb/script/ckpt_to_huggingface.sh

LOCAL_DIR=/data/yuhui/8/rl-test/ckpt_rl_npo/r1llama/MathLogic/global_step_126/actor
TARGET_DIR=/data/yuhui/8/rl-test/ckpt_rl_npo/r1llama/MathLogic

# only clear the first depth of the target dir
# echo "clear target dir"
# rm -rf $TARGET_DIR/*

echo "merge ckpt"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $LOCAL_DIR \
    --target_dir $TARGET_DIR

# echo "remove local dir"
# rm -rf $LOCAL_DIR

echo "ALL Done."