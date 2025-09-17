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
    python mmlu_verl_format.py --group_name ${GROUP_NAME}
done