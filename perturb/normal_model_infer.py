
import asyncio
from datasets import load_dataset, concatenate_datasets
from openai import AsyncOpenAI
import argparse  # ## 1. 新增：导入argparse库
import wandb  # ## 2. 新增：导入wandb库
from util.data_group import FIELDS_GROUP_DIC

# --- 1. 配置参数 ---
VLLM_BASE_URL = "http://localhost:8001/v1"  # vLLM服务地址

# --- 数据集配置 ---
# DATASET_NAME = "cais/mmlu"
# DATASET_FIELD = "high_school_world_history"
DATASET_SPLIT = "test"
CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]

# --- 推理配置 ---
BATCH_SIZE = 10  # 批处理大小，和原代码保持一致
MAX_TOKENS = 5000

# --- 2. 初始化AsyncOpenAI客户端 ---
client = AsyncOpenAI(
    base_url=VLLM_BASE_URL,
    api_key="dummy"  # vLLM不需要真实的API key
)




# --- 3. 定义推理函数 (用于.map) ---
def add_answer_column(batch):
    """
    这个函数接收一个批次的数据，使用AsyncOpenAI生成答案，并返回包含新列的批次。
    注意：由于datasets的map是同步的，我们需要在这里运行异步代码
    """
    
    async def async_process_batch():
        # 从批次中获取所有问题
        questions = batch["question"]
        choices = batch["choices"]
        answer_indices = batch["answer"]
        
        formatted_choices = []
        correct_choices = []
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
            
            prompt = f"""What is the correct answer to this question? Question:\n {questions[i]}\nChoices:\n{choice_str}\nOutput format: The correct answer is (A/B/C/D)."""
            formatted_prompts.append(prompt)
        
        # 并行处理所有prompts
        tasks = []
        for prompt in formatted_prompts:
            task = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=0.0
            )
            tasks.append(task)
        
        try:
            responses = await asyncio.gather(*tasks)
            model_answers = [response.choices[0].message.content for response in responses]
        except Exception as e:
            print(f"Error in batch processing: {e}")
            model_answers = [f"Error: {str(e)}"] * len(formatted_prompts)
        
        return {
            "correct_choice": correct_choices,
            "formatted_prompt": formatted_prompts,
            "model_answer": model_answers,
        }
    
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

def arc_to_mmlu(dataset_name: str):
    if dataset_name == "arc_easy":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split=DATASET_SPLIT)
    elif dataset_name == "arc_challenge":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=DATASET_SPLIT)
    def to_mmlu(example):
        question = example["question"]
        choices = example["choices"]["text"]
        answer = ord(example["answerKey"]) - ord("A")
        return {
            "question": question,
            "choices": choices,
            "answer": answer}
    dataset = dataset.map(to_mmlu, remove_columns=["id"])
    def filter_dataset(example):
        if example["answer"] not in [0,1,2,3] or len(example["choices"]) != 4:
            return False
        return True
    dataset = dataset.filter(filter_dataset)
    return dataset

def concat_mmlu(group_name: str):
    assert group_name in FIELDS_GROUP_DIC, f"group_name {group_name} not in dict"
    dataset_fields = FIELDS_GROUP_DIC[group_name]
    datasets = [load_dataset("cais/mmlu", field, split=DATASET_SPLIT) for field in dataset_fields]
    datasets = concatenate_datasets(datasets)
    return datasets

def gpqa_to_mmlu():
    dataset_fields = ["gpqa_main", "gpqa_diamond", "gpqa_extended"]
    keep = ["Question","Correct Answer","Incorrect Answer 1","Incorrect Answer 2","Incorrect Answer 3"]
    parts = [load_dataset("Idavidrein/gpqa", f, split="train").select_columns(keep) for f in dataset_fields]
    ds = concatenate_datasets(parts)

    # def stable_pos(q: str) -> int:
    #     h = hashlib.md5(q.encode("utf-8")).hexdigest()
    #     return int(h, 16) % 4  # 0..3

    def to_mmlu(example, idx):
        q = example["Question"]
        corr = example["Correct Answer"]
        incs = [example["Incorrect Answer 1"], example["Incorrect Answer 2"], example["Incorrect Answer 3"]]
        pos = idx % 4
        choices = incs.copy()
        choices.insert(pos, corr)  # 长度 4，corr 放在 pos 处
        return {
            "question": q,
            "choices": choices,
            "answer": pos,
        }

    mapped = ds.map(to_mmlu, with_indices=True, remove_columns=ds.column_names)
    def filter_dataset(example):
        if example["answer"] not in [0,1,2,3] or len(example["choices"]) != 4:
            return False
        return True
    mapped = mapped.filter(filter_dataset)
    return mapped

        

# --- 4. 加载数据集并执行推理 ---
if __name__ == "__main__":
    # 加载数据集
    parser = argparse.ArgumentParser(description="Run inference with different dataset.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The specific dataset name to process (e.g., 'cais/mmlu')."
    )
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
    args = parser.parse_args()
    DATASET_NAME = args.dataset_name
    GROUP_NAME = args.group_name
    MODEL_NAME = args.model_name
    SHORT_MODEL_NAME = args.short_model_name
    wandb.init(project="MemoryPerturbGroup", name=f"{DATASET_NAME}_{GROUP_NAME}_{DATASET_SPLIT}_{SHORT_MODEL_NAME}", group="normal_infer")
    wandb.config.update(args)



    
    print("Starting inference with dataset.map()...")

    if DATASET_NAME == "cais/mmlu":
        print(f"Loading dataset {DATASET_NAME}/{GROUP_NAME}...")
        dataset = concat_mmlu(GROUP_NAME)
    elif DATASET_NAME == "arc_easy" or DATASET_NAME == "arc_challenge":
        GROUP_NAME = "All"
        dataset = arc_to_mmlu(DATASET_NAME)
    elif DATASET_NAME == "gpqa":
        GROUP_NAME = "All"
        dataset = gpqa_to_mmlu()
    
    # 使用dataset.map进行批处理推理
    updated_dataset = dataset.map(
        add_answer_column,
        batched=True,
        batch_size=BATCH_SIZE
    )

    print("\nInference complete!")

    # 保存处理后的数据集到磁盘
    SAVE_PATH = f"./res2/{DATASET_NAME}_{GROUP_NAME}_{DATASET_SPLIT}_{SHORT_MODEL_NAME}_answers.json"
    print(f"\nSaving dataset to disk at {SAVE_PATH}...")
    
    # 确保目录存在
    import os
    import json
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(updated_dataset.to_list(), f, indent=4, ensure_ascii=False)

    print("Done!")
    
    # 显示一些统计信息
    print(f"Processed {len(updated_dataset)} items")
    
    # 检查错误数量
    error_count = sum(1 for answer in updated_dataset["model_answer"] if "Error:" in answer)
    print(f"Errors: {error_count}/{len(updated_dataset)}")