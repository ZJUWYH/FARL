# cd memory-perturb
# conda activate cot

# 启动vLLM服务 (在另一个终端运行):
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
#   --model "/data/yuhui/8/memory-perturb/ckpt/cais/mmlu_nutrition_test_r1llama_perturbed" \
#  --served-model-name "r1llama" \
#   --tensor-parallel-size 4 \
#   --port 8001 \
#   --gpu-memory-utilization 0.9 \
#   --dtype bfloat16

# CUDA_VISIBLE_DEVICES=0,1,2,3 python perturb_infer_api2.py --dataset_field nutrition --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --short_model_name r1llama --repeat 1

# version2: add repeat fot n time to reduce randomness, import the answer to wandb
# version3: use group fields, remove the repeat

import asyncio
from datasets import load_dataset, Dataset
from openai import AsyncOpenAI, OpenAI
import json
import os
import argparse  # 新增：导入argparse库
from tqdm import tqdm
import wandb
from dataclasses import asdict
from data_group import FIELDS_GROUP_DIC

# --- 1. 配置参数 ---
SHORT_MODEL_NAME = "r1llama"# 这里填写你的模型名称
VLLM_BASE_URL = "http://localhost:8001/v1"  # vLLM服务地址

# --- 数据集配置 ---
DATASET_NAME = "cais/mmlu"
DATASET_SPLIT = "test"
CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]
# INPUT_PATH = f"./res/{DATASET_NAME}_{DATASET_FIELD}_{DATASET_SPLIT}_{SHORT_MODEL_NAME}_perturbed_answers.json"
# SAVE_PATH = f"./res/{DATASET_NAME}_{DATASET_FIELD}_{DATASET_SPLIT}_perturb_{SHORT_MODEL_NAME}_perturbed_answers.json"

# --- 推理配置 ---
BATCH_SIZE = 12  # 批处理大小，和原代码保持一致
MAX_TOKENS = 5000

# --- 2. 初始化AsyncOpenAI客户端 ---
client = AsyncOpenAI(
    base_url=VLLM_BASE_URL,
    api_key="dummy"  # vLLM不需要真实的API key
)

# --- 3. 定义推理函数 (用于.map) ---
def add_answer_column(batch, short_model_name):
    """
    这个函数接收一个批次的数据，使用AsyncOpenAI生成答案，并返回包含新列的批次。
    注意：由于datasets的map是同步的，我们需要在这里运行异步代码
    
    Args:
        batch: 批次数据
        short_model_name: 模型名称
        n: 生成答案的次数，默认为1
    """
    
    async def async_process_batch():
        # 从批次中获取所有问题
        questions = batch["question"]
        choices = batch["choices"]
        answer_indices = batch["answer"]
        perturb_answer_indices = batch["perturb_answer"]
        
        formatted_choices = []
        correct_choices = []
        perturb_choices = []
        formatted_prompts = []
        
        # 准备所有的prompts
        for i in range(len(questions)):
            formatted_choices = []
            for idx, choice in enumerate(choices[i]):
                formatted_choices.append(f"({CHOICE_LABELS[idx]}) {choice}")
                # choice_str = f"{CHOICE_LABELS[idx]}. {choice}\n"
                # choice_str += f"({CHOICE_LABELS[idx]}). {choice}\n"
            choice_str = "\n".join(formatted_choices)
            formatted_choices.append(choice_str)
            correct_choices.append(CHOICE_LABELS[answer_indices[i]])
            perturb_choices.append(CHOICE_LABELS[perturb_answer_indices[i]])
            
            prompt = f"""What is the correct answer to this question? Question:\n {questions[i]}\nChoices:\n{choice_str}\nOutput format: The correct answer is (A/B/C/D)."""
            formatted_prompts.append(prompt)
        
        tasks = []
        for prompt in formatted_prompts:
            task = client.chat.completions.create(
                model=short_model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                # temperature=0.0
            )
            tasks.append(task)
        
        try:
            responses = await asyncio.gather(*tasks)
            # 重新组织响应：每n个响应对应一个问题
            all_model_answers = [response.choices[0].message.content for response in responses]
            
            # # 创建结果字典，包含n个不同的列
            # result = {}
            
            # # 为每个生成次数创建一个列
            # for i in range(n):
            #     column_name = f"perturb_model_answer_{i+1}"
            #     # 从all_model_answers中提取第i次生成的所有答案
            #     answers_for_this_iteration = []
            #     for question_idx in range(len(questions)):
            #         answer_idx = question_idx * n + i
            #         answers_for_this_iteration.append(all_model_answers[answer_idx])
            #     result[column_name] = answers_for_this_iteration
            result = {
                "perturb_model_answer": all_model_answers,
            }
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            result = {
                "perturb_model_answer": [f"Error: {str(e)}"] * len(formatted_prompts),
            }
        
        return result
    
    # 在同步函数中运行异步代码
    try:
        # 检查是否已经有事件循环在运行
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果已有循环在运行，创建新的线程来运行异步代码
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, async_process_batch())
                return future.result()
        else:
            return asyncio.run(async_process_batch())
    except RuntimeError:
        # 如果没有事件循环，直接创建新的
        return asyncio.run(async_process_batch())
    
def llm_extract_answer(response, client):
    """
    Extract the final answer from the LLM response.
    The response is expected to be a string containing the final answer in the format:
    'The correct answer is (A/B/C/D).'
    """
    prompt = f"""You are a helpful assistant tasked with extracting the final answer from a multiple-choice question response.
    The response is delimited by triple backticks.
    ```
    {response}
    ```
    ONLY return the letter of the final answer, without any additional text or explanation. If you cannot determine a clear answer, respond N.
    """
    message = [
        {"role": "system", "content": "You are a helpful assistant that extracts the final answer from a given text."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=message,
        max_tokens=10,
        temperature=0.0
    )
    answer = response.choices[0].message.content.strip()[0]
    if answer.upper() in ["A", "B", "C", "D"]:
        return answer.upper()
    return "None"

# --- 4. 加载数据集并执行推理 ---
if __name__ == "__main__":
    # 加载数据集
    parser = argparse.ArgumentParser(description="Run inference with a specified MMLU dataset group.")
    parser.add_argument(
        "--group_name",
        type=str,
        required=True,
        help="The specific group name of the MMLU dataset to process (e.g., 'MathLogic')."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        # default= MODEL_NAME,
        help="The name of the model to use for inference."
    )
    parser.add_argument(
        "--short_model_name",
        type=str,
        required=True,
        # default=SHORT_MODEL_NAME,
        help="The short name of the model for vLLM."
    )
    # parser.add_argument(
    #     "--repeat",
    #     type=int,
    #     default=1,
    #     help="Number of times to repeat the inference for each question to reduce randomness."
    # )
    args = parser.parse_args()
    GROUP_NAME = args.group_name
    MODEL_NAME = args.model_name
    SHORT_MODEL_NAME = args.short_model_name
    INPUT_PATH = f"./res2/{DATASET_NAME}_{GROUP_NAME}_{DATASET_SPLIT}_{SHORT_MODEL_NAME}_perturbed_answers.json"
    SAVE_PATH = f"./res4/{DATASET_NAME}_{GROUP_NAME}_{DATASET_SPLIT}_edit_{SHORT_MODEL_NAME}_perturbed_answers.json"
    # REPEAT = args.repeat
    print(f"Loading dataset {DATASET_NAME}/{GROUP_NAME}...")
    wandb.init(project="MemoryPerturbGroup", name=f"{DATASET_NAME}_{GROUP_NAME}_{DATASET_SPLIT}_{SHORT_MODEL_NAME}", group="edit_infer")
    wandb.config.update(args)
    client2 = OpenAI()




################---------------------------inference--------------------------------------###############
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)
    
    print("Starting inference with dataset.map()...")
    
    # 使用dataset.map进行批处理推理
    updated_dataset = dataset.map(
        add_answer_column,
        fn_kwargs={"short_model_name": SHORT_MODEL_NAME},
        batched=True,
        batch_size=BATCH_SIZE
    )

    print("\nInference complete!")

    # 保存处理后的数据集到磁盘
    print(f"\nSaving dataset to disk at {SAVE_PATH}...")
    
    # 确保目录存在

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(updated_dataset.to_list(), f, indent=4, ensure_ascii=False)

    print("Done!")
    
    # 显示一些统计信息
    print(f"Processed {len(updated_dataset)} items")
    
    # 检查错误数量
    error_count = sum(1 for answer in updated_dataset["model_answer"] if "Error:" in answer)
    print(f"Errors: {error_count}/{len(updated_dataset)}")


    ################--------------------------------------############### extract answer and calculate success rate
    with open(SAVE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)


    def extract_and_compare(example):
        result = {}
        # 为每个perturb_model_answer列处理
        try:
            answer = example["perturb_model_answer"].split("</think>")[-1].strip()
        except Exception as e:
            print(f"Error splitting perturb_model_answer: {e}")
            answer = example["perturb_model_answer"]
            
        extracted = llm_extract_answer(answer, client2)
        result["extracted_answer"] = extracted
        result["success_perturb"] = extracted == CHOICE_LABELS[example["perturb_answer"]]
        
        return result

    dataset = dataset.map(extract_and_compare)

    # os.makedirs(os.path.dirname(SAVE_PATH2), exist_ok=True)

    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(dataset.to_list(), f, indent=4, ensure_ascii=False)


    # 计算总体平均成功率
    total_attempts = len(dataset) 
    total_success = len(dataset.filter(lambda x: x["success_perturb"] == True))

    print(f"Overall average perturb accuracy: {total_success}/{total_attempts} = {total_success/total_attempts:.2%}")
    wandb.log({
        "perturb_accuracy": total_success / total_attempts,})

    ###########------------------------------------------  calculate the consistency rate ---------------------------------##############
    print("Calculating consistency rate...")
    with open(SAVE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 为每个perturb_model_answer列计算一致性
    print(f"Processing consistency for perturb_model_answer...")
    for item in tqdm(data):
        perturb_model_answer = item.get("perturb_model_answer", "")
        think_draft = perturb_model_answer.split("</think>")
            
        if len(think_draft) < 2:
            item["think_draft_answer"] = "None"
            item["consistent"] = None
            continue
                
        think_draft = think_draft[0].strip()
        think_draft_answer = llm_extract_answer(think_draft, client2)
        item["think_draft_answer"] = think_draft_answer
        item["consistent"] = think_draft_answer == item["extracted_answer"]

    dataset = Dataset.from_list(data)
    # perturb_consistency_rate
    perturb_consistency_rate = len(dataset.filter(lambda x: x["consistent"] == True and x["success_perturb"] == True)) / len(dataset)
    unperturb_consistency_rate = len(dataset.filter(lambda x: x["consistent"] == True and x["success_perturb"] == False)) / len(dataset)
    perturb_inconsistent_rate = len(dataset.filter(lambda x: x["consistent"] == False and x["success_perturb"] == True)) / len(dataset)
    unperturb_inconsistent_rate = len(dataset.filter(lambda x: x["consistent"] == False and x["success_perturb"] == False)) / len(dataset)
    print(f"perturb_consistency_rate: {perturb_consistency_rate}, unperturb_consistency_rate: {unperturb_consistency_rate}, perturb_inconsistent_rate: {perturb_inconsistent_rate}, unperturb_inconsistent_rate: {unperturb_inconsistent_rate}")
    wandb.log({
        "perturb_consistency_rate": perturb_consistency_rate,
        "unperturb_consistency_rate": unperturb_consistency_rate,
        "perturb_inconsistent_rate": perturb_inconsistent_rate,
        "unperturb_inconsistent_rate": unperturb_inconsistent_rate,
    })

    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        





