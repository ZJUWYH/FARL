#! /bin/bash

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
    python -m util.mmlu_verl_format --group_name ${GROUP_NAME}
done