# listener_experiment.py

import os
import csv
import time
import json
import ast
import re
import pandas as pd
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# 加载 API 密钥
load_dotenv()
# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（上一级）
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 配置路径
STIMULI_FILE = os.path.join(PROJECT_ROOT, "listener_stimuli.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")

# ==================== 提示词模板配置 ====================

PROMPT_TEMPLATES = {
   "direct_label": {
    "description": "Baseline: Simple forced choice without explanation.",
    "en": """Background: 5 students took a test consisting of 12 questions.
Exam results: {exam_data}
Sentence: "{sentence}"
Speaker Type Definitions:
- high: make the exam sound easy (high success rate)
- low: make the exam sound hard (low success rate)
- info: describe the results objectively, no framing
Task: Given an exam-results table and a sentence, decide which type of speaker most likely produced that sentence.
Output format: Directly output the label only (high, low, or info)""",
    "zh": """背景：有 5 名学生参加了一场包含 12 道题目的考试。
考试结果：{exam_data}
句子："{sentence}"
说话者类型定义：
- high: 让考试听起来容易（高成功率）
- low: 让考试听起来难（低成功率）
- info: 客观描述结果，不带倾向性
任务：根据给定的考试结果表格和句子，判断该句子最可能由哪种类型的说话者产生。
输出格式：直接输出标签（high、low 或 info）。"""
},
    "role_description": {
    "description": "Contextual Priming: Explicitly defining speaker motivations.",
    "en": """Background: This is an experiment. You will be shown an exam result, followed by a sentence. Based on the exam result, you need to determine which type of speaker most likely produced the sentence.

Speaker Goals:
1. Teacher (high): Make exam seem easy.
2. Student (low): Make exam seem hard.
3. Examiner (info): Objective description.

Exam results: {exam_data}
Sentence: "{sentence}"

Question: Which speaker type produced this? max 20 words.""",
    "zh": """背景：这是一个实验。实验会先展示一个考试结果，然后呈现一个句子。你需要根据这个考试结果，判断该句子最可能由哪种类型的说话者说出。

说话者目标：
1. 教师 (high): 强调成功率，显简单。
2. 学生 (low): 强调失败率，显困难。
3. 考官 (info): 客观描述。

考试结果：{exam_data}
句子："{sentence}"

问题：这句话最可能是哪种说话者说的？不超过 20 字。"""
},
    "chain_of_thought": {
    "description": "Reasoning: Forcing the model to analyze before labeling.",
    "en": """Background: 5 students took a test consisting of 12 questions. This is an experiment. You will be shown an exam result, followed by a sentence. Based on the exam result, you need to determine which type of speaker most likely produced the sentence.

Speaker Type Definitions:
- high (Teacher): make the exam sound easy (high success rate)
- low (Student): make the exam sound hard (low success rate)
- info (Examiner): describe the results objectively, no framing

Exam results: {exam_data}
Sentence: "{sentence}"

IMPORTANT: You MUST follow this EXACT format:
[One sentence analysis, max 20 words]
Final label: [high/low/info]

Now provide your response:""",
    "zh": """背景：有 5 名学生参加了一场包含 12 道题目的考试。这是一个实验。实验会先展示一个考试结果，然后呈现一个句子。你需要根据这个考试结果，判断该句子最可能由哪种类型的说话者说出。

说话者类型定义：
- high（教师）：让考试听起来容易（高成功率）
- low（学生）：让考试听起来难（低成功率）
- info（考官）：客观描述结果，不带倾向性

考试结果：{exam_data}
句子："{sentence}"

重要：你必须严格按照以下格式回答：
[一句分析，最多 20 个字]，并且最终标签：[high/low/info]

现在请回答："""
},
    "structured_justification": {
    "en": """Background: 5 students took a test consisting of 12 questions. This is an experiment. You will be shown an exam result, followed by a sentence. Based on the exam result, you need to determine which type of speaker most likely produced the sentence.

Speaker Type Definitions:
- high (Teacher): make the exam sound easy (high success rate)
- low (Student): make the exam sound hard (low success rate)
- info (Examiner): describe the results objectively, no framing

Exam results: {exam_data}
Sentence: "{sentence}"

Provide your response in the following EXACT format:
Evidence: [one short sentence, max 10 words]
Label: [high, low, or info]""",
    "zh": """背景：有 5 名学生参加了一场包含 12 道题目的考试。这是一个实验。实验会先展示一个考试结果，然后呈现一个句子。你需要根据这个考试结果，判断该句子最可能由哪种类型的说话者说出。

说话者类型定义：
- high（教师）：让考试听起来容易（高成功率）
- low（学生）：让考试听起来难（低成功率）
- info（考官）：客观描述结果，不带倾向性

考试结果：{exam_data}
句子："{sentence}"

请严格按照以下格式回答：
证据：[一句简短理由，最多 10 个字]
标签：[high, low, info]
"""
},
    "calibration": {
    "description": "Confidence: Assessing model certainty.",
    "en": """Background: 5 students took a test consisting of 12 questions. This is an experiment. You will be shown an exam result, followed by a sentence. Based on the exam result, you need to determine which type of speaker most likely produced the sentence.

Speaker Type Definitions:
- high (Teacher): make the exam sound easy (high success rate)
- low (Student): make the exam sound hard (low success rate)
- info (Examiner): describe the results objectively, no framing

Exam results: {exam_data}
Sentence: "{sentence}"

Respond with EXACTLY this format:
Label: [high/low/info]
Confidence: [0.0 to 1.0]""",
    "zh": """背景：有 5 名学生参加了一场包含 12 道题目的考试。这是一个实验。实验会先展示一个考试结果，然后呈现一个句子。你需要根据这个考试结果，判断该句子最可能由哪种类型的说话者说出。

说话者类型定义：
- high（教师）：让考试听起来容易（高成功率）
- low（学生）：让考试听起来难（低成功率）
- info（考官）：客观描述结果，不带倾向性

考试结果：{exam_data}
句子："{sentence}"

请严格按照以下格式回答：
标签：[high/low/info]
置信度：[0.0 到 1.0]"""
}
}

# ==================== 模型配置 ====================

MODEL_CONFIGS = {
    "deepseek-v3.2": {
        "client_type": "openai",
        "api_key_env": "SILICONFLOW_API_KEY",
        "model_name": "deepseek-ai/DeepSeek-V3.2",
        "base_url": "https://api.siliconflow.cn/v1",
        "supports_logprobs": True,
        "supports_top_logprobs": False
    },
    "gpt-4o-mini": {
        "client_type": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "model_name": "gpt-4o-mini",
        "base_url": "https://free.v36.cm/v1",
        "supports_logprobs": True,
        "supports_top_logprobs": True
    },
    "claude-sonnet-4-6-thinking": {
        "client_type": "openai",
        "api_key_env": "LAOZHANG_API_KEY",
        "model_name": "claude-sonnet-4-6-thinking",
        "base_url": "https://api.laozhang.ai/v1",
        "supports_logprobs": False,
        "supports_top_logprobs": False
    },
    "gemini-pro": {
        "client_type": "openai",
        "api_key_env": "GEMINI_API_KEY",
        "model_name": "gemini-pro",
        "base_url": "https://zenmux.ai/api/v1",
        "supports_logprobs": False,
        "supports_top_logprobs": False
    },
    "Qwen3.5-35B-A3B": {
        "client_type": "openai",
        "api_key_env": "Huggingface_API_KEY",
        "model_name": "Qwen/Qwen3.5-35B-A3B:novita",
        "base_url": "https://router.huggingface.co/v1",
        "supports_logprobs": True,
        "supports_top_logprobs": False
    }
}

# ==================== 工具函数 ====================

def parse_observation(obs_str):
    """Parse observation field"""
    if isinstance(obs_str, str):
        try:
            return ast.literal_eval(obs_str)
        except:
            return json.loads(obs_str)
    return obs_str

def format_exam_data(correct_counts, total_questions=12):
    """
    Format exam results as raw data display
    Example: Student 1: 8/12 correct, Student 2: 5/12 correct, ...
    """
    exam_data_list = []
    for i, correct in enumerate(correct_counts, 1):
        exam_data_list.append(f"Student {i}: {correct}/{total_questions} correct")
    return ", ".join(exam_data_list)

def build_sentence(q1, q2, adj):
    """Build complete sentence"""
    return f"{q1} of the students got {q2} of the answers {adj}"

def create_prompt(correct_counts, q1, q2, adj, total_questions=12, template_name="direct_label", language="en"):
    """
    Create prompt supporting different prompt templates
    """
    exam_data = format_exam_data(correct_counts, total_questions)
    sentence = build_sentence(q1, q2, adj)
    
    template = PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["direct_label"])
    template_str = template.get(language, template.get("en", template["en"]))
    
    prompt = template_str.format(exam_data=exam_data, sentence=sentence)
    
    return prompt

def clean_special_chars(text):
    """Clean special characters that may cause encoding issues"""
    if not text:
        return text
    
    special_chars = {
        '\u2013': '-',
        '\u2014': '-',
        '\u2018': "'",
        '\u2019': "'",
        '\u201c': '"',
        '\u201d': '"',
        '\u2026': '...',
        '\u200b': '',
        '\ufeff': '',
        '\u00a0': ' ',
    }
    
    for old, new in special_chars.items():
        text = text.replace(old, new)
    
    return text

def parse_response(response_text):
    """Parse model response - comprehensive version with improved label extraction"""
    if not response_text:
        return "ERROR: Empty response"
    
    cleaned_text = clean_special_chars(response_text)
    response_lower = cleaned_text.lower().strip()
    
    label_patterns = [
        r'(?:final\s+)?label\s*[:：]\s*([a-z]+)',
        r'step\s*2\s*[:：]\s*(?:final\s+)?label\s*[:：]\s*([a-z]+)',
        r'answer\s*[:：]\s*([a-z]+)',
        r'therefore\s*[,:]?\s*([a-z]+)',
        r'so\s*[,:]?\s*([a-z]+)',
        r'^\s*([a-z]+)\s*$',
    ]
    
    for pattern in label_patterns:
        match = re.search(pattern, response_lower, re.MULTILINE)
        if match:
            label = match.group(1).strip()
            if label in ['high', 'low', 'info']:
                return label
    
    lines = response_lower.split('\n')
    for line in reversed(lines[-5:]):
        line_clean = line.strip()
        if line_clean in ['high', 'low', 'info']:
            return line_clean
    
    words = re.findall(r'\b([a-z]+)\b', response_lower)
    for word in reversed(words):
        if word in ['high', 'low', 'info']:
            return word
    
    has_high = 'high' in response_lower
    has_low = 'low' in response_lower
    has_info = 'info' in response_lower
    
    if has_high and not has_low and not has_info:
        return 'high'
    elif has_low and not has_high and not has_info:
        return 'low'
    elif has_info and not has_high and not has_low:
        return 'info'
    
    error_preview = cleaned_text[:100] + "..." if len(cleaned_text) > 100 else cleaned_text
    return f"Invalid: {error_preview}"

def parse_logprobs_response(response_text):
    """
    Parse model response that includes log probabilities
    """
    if not response_text:
        return "ERROR: Empty response", None
    
    cleaned_text = clean_special_chars(response_text)
    response_lower = cleaned_text.lower().strip()
    
    label = parse_response(response_text)
    
    if label.startswith("ERROR") or label.startswith("Invalid"):
        return label, None
    
    confidence_patterns = [
        r'confidence\s*[:：]\s*([0-9.]+)',
        r'置信度\s*[:：]\s*([0-9.]+)',
        r'score\s*[:：]\s*([0-9.]+)',
        r'probability\s*[:：]\s*([0-9.]+)',
        r'likelihood\s*[:：]\s*([0-9.]+)',
    ]
    
    for pattern in confidence_patterns:
        match = re.search(pattern, response_lower)
        if match:
            try:
                confidence = float(match.group(1))
                return label, confidence
            except:
                pass
    
    return label, None

# ==================== API 调用函数 ====================

def init_client(model_name):
    """Initialize model client"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model name: {model_name}")
    
    config = MODEL_CONFIGS[model_name]
    api_key = os.getenv(config["api_key_env"])
    
    if not api_key:
        raise ValueError(f"Please set environment variable {config['api_key_env']}")
    
    if config["client_type"] == "openai":
        import openai
        client = openai.OpenAI(
            api_key=api_key,
            base_url=config["base_url"] if config["base_url"] else None
        )
        return client, config
    
    elif config["client_type"] == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(config["model_name"])
        return model, config
    
    else:
        raise ValueError(f"Unknown client type: {config['client_type']}")

def extract_logprobs_data(response):
    """
    灵活提取 logprobs 数据，兼容不同 API 返回格式
    """
    logprobs_data = {}
    
    try:
        choice = response.choices[0]
        
        # 尝试不同可能的 logprobs 属性名
        logprobs_obj = None
        
        # 方式 1: 直接 logprobs 属性
        if hasattr(choice, 'logprobs') and choice.logprobs:
            logprobs_obj = choice.logprobs
        
        # 方式 2: 检查是否在 message 中
        if not logprobs_obj and hasattr(choice, 'message'):
            if hasattr(choice.message, 'logprobs') and choice.message.logprobs:
                logprobs_obj = choice.message.logprobs
        
        if logprobs_obj:
            # 尝试转换为字典
            if hasattr(logprobs_obj, 'model_dump'):
                logprobs_data = logprobs_obj.model_dump()
            elif hasattr(logprobs_obj, 'to_dict'):
                logprobs_data = logprobs_obj.to_dict()
            elif isinstance(logprobs_obj, dict):
                logprobs_data = logprobs_obj
            else:
                # 尝试直接序列化
                try:
                    logprobs_data = json.loads(str(logprobs_obj))
                except:
                    logprobs_data = {"raw": str(logprobs_obj)}
            
            # 提取关键信息：token 和 logprob
            if not logprobs_data or len(logprobs_data) == 0:
                # 尝试从 content 中提取
                if hasattr(logprobs_obj, 'content') and logprobs_obj.content:
                    content_logprobs = []
                    for item in logprobs_obj.content:
                        if hasattr(item, 'token') and hasattr(item, 'logprob'):
                            content_logprobs.append({
                                "token": item.token,
                                "logprob": item.logprob
                            })
                    if content_logprobs:
                        logprobs_data = {"content": content_logprobs}
        
        # 如果仍然为空，添加响应基本信息
        if not logprobs_data:
            logprobs_data = {
                "note": "Logprobs requested but API may not support detailed format",
                "model": response.model if hasattr(response, 'model') else "unknown"
            }
            
    except Exception as e:
        logprobs_data = {"error": f"Failed to extract logprobs: {str(e)}"}
    
    return logprobs_data

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(model_name, prompt, temperature=0.7, max_tokens=100, logprobs=False, top_logprobs=5):
    """
    Call LLM with configurable max_tokens and optional logprobs
    """
    client, config = init_client(model_name)
    
    try:
        if config["client_type"] == "openai":
            request_params = {
                "model": config["model_name"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # 只在支持时添加 logprobs 参数
            if logprobs and config.get("supports_logprobs", False):
                request_params["logprobs"] = True
                # 只在支持 top_logprobs 时添加
                if config.get("supports_top_logprobs", False):
                    request_params["top_logprobs"] = top_logprobs
            
            print(f"    [API Request] logprobs={request_params.get('logprobs', False)}, top_logprobs={request_params.get('top_logprobs', 'N/A')}")
            
            response = client.chat.completions.create(**request_params)
            
            response_text = response.choices[0].message.content.strip()
            
            # 提取 logprobs 数据
            logprobs_data = None
            if logprobs and config.get("supports_logprobs", False):
                logprobs_data = extract_logprobs_data(response)
                print(f"    [Logprobs] Extracted: {bool(logprobs_data)}")
            
            return response_text, logprobs_data
    
        elif config["client_type"] == "gemini":
            response = client.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            )
            response_text = response.text.strip()
            return response_text, None
    
    except Exception as e:
        print(f"API call error: {e}")
        raise e

# ==================== 主实验函数 ====================

def run_listener_experiment(
    stimuli_file,
    output_dir,
    runs_per_item,  
    models_to_run=None,
    total_questions=12,
    delay=2,
    prompt_template="direct_label",
    language="en",
    max_items=None,
    temperature=0.7,
    max_tokens=100,
    verbose=True
):
    """
    Run listener experiment (original function, unchanged)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading stimuli file: {stimuli_file}")
    df = pd.read_csv(stimuli_file)
    
    if max_items is not None:
        df = df.head(max_items)
        print(f"Testing first {max_items} items only")
    
    print(f"Total items to test: {len(df)}")
    
    if models_to_run is None:
        models_to_run = list(MODEL_CONFIGS.keys())
    
    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"Running model: {model_name}")
        print(f"Prompt template: {prompt_template} ({language})")
        print(f"Temperature: {temperature}")
        print(f"Max tokens: {max_tokens}")
        print(f"{'='*60}")
        
        safe_model_name = model_name.replace('-', '_').replace('.', '_')
        safe_template_name = prompt_template.replace('-', '_')
        safe_temperature = str(temperature).replace('.', '_')
        
        output_file = f"{output_dir}/{safe_model_name}_{safe_template_name}_temp{safe_temperature}.csv"
        
        total_calls = 0
        error_count = 0
        
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerow([
                "itemID", "list", "condition", "Q1", "Q2", "A",
                "observation", "sentence", "run", "chosen_speaker",
                "raw_response", "timestamp"
            ])
            
            for idx, row in df.iterrows():
                item_id = row['itemID']
                exp_list = row['list']
                condition = row['condition']
                q1 = row['Q1']
                q2 = row['Q2']
                adj = row['A']
                obs_str = row['observation']
                
                correct_counts = parse_observation(obs_str)
                
                prompt = create_prompt(
                    correct_counts, q1, q2, adj, total_questions,
                    template_name=prompt_template, language=language
                )
                sentence = build_sentence(q1, q2, adj)
                
                for run in range(1, runs_per_item + 1):
                    total_calls += 1
                    if verbose:
                        print(f"  Item {item_id} ({idx+1}/{len(df)}), Run {run}/{runs_per_item}")
                    
                    try:
                        raw_response, _ = call_llm(model_name, prompt, temperature=temperature, max_tokens=max_tokens, logprobs=False)
                        
                        cleaned_response = clean_special_chars(raw_response)
                        chosen = parse_response(cleaned_response)
                        
                        writer.writerow([
                            item_id, exp_list, condition, q1, q2, adj,
                            obs_str, sentence, run, chosen,
                            cleaned_response, datetime.now().isoformat()
                        ])
                        csvfile.flush()
                        
                    except Exception as e:
                        error_count += 1
                        print(f"    Error: {e}")
                        writer.writerow([
                            item_id, exp_list, condition, q1, q2, adj,
                            obs_str, sentence, run, "ERROR",
                            f"Error: {str(e)}", datetime.now().isoformat()
                        ])
                        csvfile.flush()
                    
                    if delay > 0:
                        time.sleep(delay)
        
        print(f"\nCompleted model: {model_name}")
        print(f"  Total API calls: {total_calls}")
        print(f"  Errors: {error_count}")
        print(f"  Success rate: {((total_calls - error_count) / total_calls * 100):.1f}%")
        print(f"  Results saved to: {output_file}")
        
        if model_name != models_to_run[-1]:
            print("\nWaiting 5 seconds before running next model...")
            time.sleep(5)
    
    print("\n" + "="*60)
    print("All model experiments completed!")
    print("="*60)

def run_listener_experiment_with_logprobs(
    stimuli_file,
    output_dir,
    runs_per_item,
    models_to_run=None,
    temperatures=[0.2, 0.7, 1.2],
    prompt_templates=None,
    total_questions=12,
    delay=2,
    language="en",
    max_items=30,
    max_tokens=100,
    verbose=True
):
    """
    Run listener experiment with log probabilities extraction
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading stimuli file: {stimuli_file}")
    df = pd.read_csv(stimuli_file)
    
    if max_items is not None:
        df = df.head(max_items)
        print(f"Testing first {max_items} items only")
    
    print(f"Total items to test: {len(df)}")
    
    if models_to_run is None:
        models_to_run = list(MODEL_CONFIGS.keys())
    
    if prompt_templates is None:
        prompt_templates = ["direct_label", "role_description", "chain_of_thought", "structured_justification", "calibration"]
    
    for model_name in models_to_run:
        print(f"\n{'='*80}")
        print(f"Running model: {model_name}")
        print(f"Temperatures: {temperatures}")
        print(f"Prompt templates: {prompt_templates}")
        print(f"Runs per item: {runs_per_item}")
        print(f"Max items: {max_items}")
        print(f"{'='*80}")
        
        supports_logprobs = MODEL_CONFIGS.get(model_name, {}).get("supports_logprobs", False)
        print(f"Model supports logprobs: {supports_logprobs}")
        
        safe_model_name = model_name.replace('-', '_').replace('.', '_')
        logprobs_file = f"{output_dir}/{safe_model_name}_Log_probabilities.csv"
        
        total_calls = 0
        error_count = 0
        
        with open(logprobs_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerow([
                "itemID", "list", "condition", "Q1", "Q2", "A",
                "observation", "sentence", "prompt_template", "temperature", 
                "run", "chosen_speaker", "confidence_logprob", "raw_response", 
                "logprobs_data", "timestamp"
            ])
            
            for template_name in prompt_templates:
                print(f"\n--- Processing prompt template: {template_name} ---")
                
                for temperature in temperatures:
                    print(f"  Temperature: {temperature}")
                    
                    for idx, row in df.iterrows():
                        item_id = row['itemID']
                        exp_list = row['list']
                        condition = row['condition']
                        q1 = row['Q1']
                        q2 = row['Q2']
                        adj = row['A']
                        obs_str = row['observation']
                        
                        correct_counts = parse_observation(obs_str)
                        
                        prompt = create_prompt(
                            correct_counts, q1, q2, adj, total_questions,
                            template_name=template_name, language=language
                        )
                        sentence = build_sentence(q1, q2, adj)
                        
                        for run in range(1, runs_per_item + 1):
                            total_calls += 1
                            if verbose:
                                print(f"    Item {item_id} ({idx+1}/{len(df)}), Template: {template_name}, Temp: {temperature}, Run {run}/{runs_per_item}")
                            
                            try:
                                raw_response, logprobs = call_llm(
                                    model_name, prompt, 
                                    temperature=temperature, 
                                    max_tokens=max_tokens, 
                                    logprobs=supports_logprobs,
                                    top_logprobs=5
                                )
                                
                                cleaned_response = clean_special_chars(raw_response)
                                chosen, confidence = parse_logprobs_response(cleaned_response)
                                
                                # 确保 logprobs 数据被正确序列化
                                logprobs_str = ""
                                if logprobs:
                                    try:
                                        logprobs_str = json.dumps(logprobs, ensure_ascii=False, default=str)
                                    except Exception as e:
                                        logprobs_str = json.dumps({"error": str(e), "raw": str(logprobs)})
                                
                                writer.writerow([
                                    item_id, exp_list, condition, q1, q2, adj,
                                    obs_str, sentence, template_name, temperature,
                                    run, chosen, confidence if confidence is not None else "",
                                    cleaned_response, logprobs_str, datetime.now().isoformat()
                                ])
                                csvfile.flush()
                                
                            except Exception as e:
                                error_count += 1
                                print(f"      Error: {e}")
                                writer.writerow([
                                    item_id, exp_list, condition, q1, q2, adj,
                                    obs_str, sentence, template_name, temperature,
                                    run, "ERROR", "",
                                    f"Error: {str(e)}", "", datetime.now().isoformat()
                                ])
                                csvfile.flush()
                            
                            if delay > 0:
                                time.sleep(delay)
        
        print(f"\nCompleted model: {model_name}")
        print(f"  Total API calls: {total_calls}")
        print(f"  Errors: {error_count}")
        print(f"  Success rate: {((total_calls - error_count) / total_calls * 100):.1f}%")
        print(f"  Log probabilities results saved to: {logprobs_file}")
        
        if model_name != models_to_run[-1]:
            print("\nWaiting 5 seconds before running next model...")
            time.sleep(5)
    
    print("\n" + "="*80)
    print("All logprobs experiments completed!")
    print("="*80)

# ==================== 主程序 ====================
if __name__ == "__main__":
   
    RUNS_PER_ITEM = 20
    TOTAL_QUESTIONS = 12
    MAX_ITEMS = 30
    
    TEMPERATURE = 0.7
    MAX_TOKENS = 100
    DELAY = 2
    
    PROMPT_TEMPLATE = "calibration"
    LANGUAGE = "en"
    
    MODELS = ["Qwen3.5-35B-A3B"]
    
    VERBOSE = True
    
    RUN_LOGPROBS_EXPERIMENT = True
    
    LOGPROBS_TEMPERATURES = [0.2, 0.7, 1.2]
    LOGPROBS_PROMPT_TEMPLATES = ["direct_label", "role_description", "chain_of_thought", "structured_justification", "calibration"]
    LOGPROBS_MAX_ITEMS = 1
    LOGPROBS_RUNS_PER_ITEM = 1
    
    LOGPROBS_MODELS = ["Qwen3.5-35B-A3B"]
    
    print("="*60)
    print("LLM Listener Experiment - Speaker Type Attribution")
    print("="*60)
    print(f"Stimuli file: {STIMULI_FILE}")
    try:
        total_items = len(pd.read_csv(STIMULI_FILE))
        print(f"Total items in file: {total_items}")
    except:
        print(f"Total items in file: Unable to read file")
    
    if not RUN_LOGPROBS_EXPERIMENT:
        print(f"Items to test: {MAX_ITEMS} (first {MAX_ITEMS} items)")
        print(f"Runs per item: {RUNS_PER_ITEM}")
        print(f"Total API calls: {MAX_ITEMS * RUNS_PER_ITEM * len(MODELS)}")
        print(f"Models: {', '.join(MODELS)}")
        print(f"Prompt template: {PROMPT_TEMPLATE} ({LANGUAGE})")
        print(f"Temperature: {TEMPERATURE}")
        print(f"Max tokens: {MAX_TOKENS}")
        print(f"Delay between calls: {DELAY} seconds")
        
        response = input("\nPress Enter to start experiment, or 'q' to quit: ")
        if response.lower() == 'q':
            print("Experiment cancelled")
            exit()
        
        run_listener_experiment(
            STIMULI_FILE,
            OUTPUT_DIR,
            RUNS_PER_ITEM,
            MODELS,
            TOTAL_QUESTIONS,
            DELAY,
            PROMPT_TEMPLATE,
            LANGUAGE,
            MAX_ITEMS,
            TEMPERATURE,
            MAX_TOKENS
        )
    else:
        print("\nRunning logprobs experiment...")
        print(f"Items to test: {LOGPROBS_MAX_ITEMS}")
        print(f"Runs per item: {LOGPROBS_RUNS_PER_ITEM}")
        print(f"Temperatures: {LOGPROBS_TEMPERATURES}")
        print(f"Prompt templates: {LOGPROBS_PROMPT_TEMPLATES}")
        print(f"Models: {', '.join(LOGPROBS_MODELS)}")
        
        response = input("\nPress Enter to start logprobs experiment, or 'q' to quit: ")
        if response.lower() == 'q':
            print("Experiment cancelled")
            exit()
        
        run_listener_experiment_with_logprobs(
            STIMULI_FILE,
            OUTPUT_DIR,
            LOGPROBS_RUNS_PER_ITEM,
            LOGPROBS_MODELS,
            LOGPROBS_TEMPERATURES,
            LOGPROBS_PROMPT_TEMPLATES,
            TOTAL_QUESTIONS,
            DELAY,
            LANGUAGE,
            LOGPROBS_MAX_ITEMS
        )