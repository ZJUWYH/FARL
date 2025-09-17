from openai import OpenAI
import re


def my_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    return len(solution_str)/10000

class MMLURewardFunction_class:
    """Reward function for MMLU training"""
    
    def __init__(self):
        # self.tokenizer = tokenizer
        self.correct_reward = 1.0
        self.incorrect_reward = -1.0
        self.format_penalty = -0.5
        self.client = OpenAI()
        
    def llm_extract_answer(self, response):
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
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=message,
            max_tokens=10,
            temperature=0.0
        )
        answer = response.choices[0].message.content.strip()[0]
        if answer.upper() in ["A", "B", "C", "D"]:
            return answer.upper()
        # print("Failed to extract a valid answer, returning None.")
        return "None"
    
    def __call__(self, data_source, solution_str, ground_truth, extra_info=None):
        # print("solution_str", solution_str)
        # print("ground_truth", ground_truth)
        predicted_answer = self.llm_extract_answer(solution_str)
        if predicted_answer == "None":
            print("Failed to extract a valid answer, returning ***FORMAT_PENALTY*** reward.")
            return self.format_penalty
        elif predicted_answer == ground_truth:
            print("Correct answer, returning ***CORRECT*** reward.")
            return self.correct_reward
        else:
            print("Incorrect answer, returning ***INCORRECT*** reward.")
            return self.incorrect_reward

MMLURewardFunction = MMLURewardFunction_class()


class MMLURewardFunction_class_v2:
    """Reward function for MMLU training"""
    
    def __init__(self):
        self.correct_reward = 1.0
        self.incorrect_reward = -1.0
        self.format_penalty = -0.5
        self.client = OpenAI()

    # --- 新增：基于字符串/正则的优先级提取 ---
    def _regex_extract_answer(self, text: str) -> str:
        """
        Try to extract the answer via regex with strict priority:
        1) 'correct answer is (A)'
        2) 'boxed{B}'
        3) 'boxed{(B)}'
        4) 'answer is (A)'
        Returns 'A'/'B'/'C'/'D' if found; otherwise 'None'.
        """
        if not isinstance(text, str):
            return "None"

        # 允许可选前导反斜杠 \boxed，也允许没有反斜杠的 boxed
        # 统一大小写不敏感
        flags = re.IGNORECASE

        patterns_in_priority = [
            # 1) correct answer is (A)
            re.compile(r'correct\s*answer\s*is\s*\(\s*([ABCD])\s*\)', flags),

            # 2) boxed{B}  —— 不包含括号的版本。用(?!\()确保花括号内首字符不是 '('
            re.compile(r'\\?boxed\{\s*(?!\()\s*([ABCD])\s*\}', flags),

            # 3) boxed{(B)} —— 包含括号的版本
            re.compile(r'\\?boxed\{\s*\(\s*([ABCD])\s*\)\s*\}', flags),

            # 4) answer is (A)
            re.compile(r'(?<!correct\s)answer\s*is\s*\(\s*([ABCD])\s*\)', flags),
        ]

        for pat in patterns_in_priority:
            m = pat.search(text)
            if m:
                ans = m.group(1).upper()
                if ans in ["A", "B", "C", "D"]:
                    return ans

        return "None"

    def llm_extract_answer(self, response):
        """
        Extract the final answer from the text:
        1) Try regex patterns in priority order.
        2) Fall back to an LLM extraction if regex fails.
        """
        # 先走正则优先级匹配
        regex_ans = self._regex_extract_answer(response)
        if regex_ans in ["A", "B", "C", "D"]:
            return regex_ans

        # 正则失败才回退到 LLM
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
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=message,
                max_tokens=10,
                temperature=0.0
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                ch = content[0].upper()
                if ch in ["A", "B", "C", "D"]:
                    return ch
        except Exception as e:
            # 这里静默失败，走到统一返回 None
            pass

        return "None"
    
    def __call__(self, data_source, solution_str, ground_truth, extra_info=None):
        predicted_answer = self.llm_extract_answer(solution_str)
        if predicted_answer == "None":
            print("Failed to extract a valid answer, returning ***FORMAT_PENALTY*** reward.")
            return self.format_penalty
        elif predicted_answer == ground_truth:
            print("Correct answer, returning ***CORRECT*** reward.")
            return self.correct_reward
        else:
            print("Incorrect answer, returning ***INCORRECT*** reward.")
            return self.incorrect_reward

MMLURewardFunction_v2 = MMLURewardFunction_class_v2()


    
