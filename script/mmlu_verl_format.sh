#! /bin/bash
# bash /data/yuhui/8/memory-perturb/script/mmlu_verl_format.sh

# DATASET_FIELDS=("nutrition")
# SAVE_PATH="/data/yuhui/8/memory-perturb/data/mmlu/nutrition"

# python mmlu_verl_format.py \
#     --dataset_field ${DATASET_FIELDS[@]} \
#     --save_path ${SAVE_PATH} \
#     --trail False

# DATASET_GROUPS=(
#     # "MathLogic"
#     "LifeSci"
#     # "Health"
#     # "BusinessEcon"
#     # "Society"
# )
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

for GROUP_NAME in ${DATASET_GROUPS[@]}; do
    python mmlu_verl_format2.py --group_name ${GROUP_NAME} --trail True
done