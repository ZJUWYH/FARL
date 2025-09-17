# This is the script that change verl ckpt to huggingface model

LOCAL_DIR="" # enter the path ended with "/actor"
TARGET_DIR=""

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