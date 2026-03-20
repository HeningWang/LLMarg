# listener_experiment2.py

import os
import csv
import time
import json
import ast
import pandas as pd
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# 加载API密钥
load_dotenv()
# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（上一级）
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 配置路径
STIMULI_FILE = os.path.join(PROJECT_ROOT, "listener_stimuli.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")

# ==================== 模型配置 ====================

MODEL_CONFIGS = {
    "deepseek-v3.2": {
        "client_type": "openai",
        "api_key_env": "SILICONFLOW_API_KEY",
        "model_name": "deepseek-ai/DeepSeek-V3.2",
        "base_url": "https://api.siliconflow.cn/v1"
    },
    "gpt-4o-mini": {
        "client_type": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "model_name": "gpt-4o-mini",
        "base_url": "https://free.v36.cm/v1"
    },
    "claude-sonnet-4-6-thinking": {
        "client_type": "openai",
        "api_key_env": "LAOZHANG_API_KEY",
        "model_name": "claude-sonnet-4-6-thinking",
        "base_url": "https://api.laozhang.ai/v1"
    },
    "gemini-pro": {
        "client_type": "openai",
        "api_key_env": "GEMINI_API_KEY",
        "model_name": "gemini-pro",
        "base_url": "https://zenmux.ai/api/v1"
    }
}

# ==================== 工具函数 ====================

def parse_observation(obs_str):
    """解析observation字段"""
    if isinstance(obs_str, str):
        try:
            return ast.literal_eval(obs_str)
        except:
            return json.loads(obs_str)
    return obs_str

def format_exam_table(correct_counts, total_questions=12):
    """格式化考试结果表格"""
    lines = []
    for i, correct in enumerate(correct_counts, 1):
        percentage = (correct / total_questions) * 100
        lines.append(f"学生{i}: 答对{correct}/{total_questions}题 ({percentage:.0f}%)")
    return "\n".join(lines)

def build_sentence(q1, q2, adj):
    """构建完整句子"""
    return f"{q1} of the students got {q2} of the answers {adj}"

def create_prompt(correct_counts, q1, q2, adj, total_questions=12):
    """创建prompt"""
    exam_table = format_exam_table(correct_counts, total_questions)
    sentence = build_sentence(q1, q2, adj)
    
    prompt = f"""你是一位语言理解专家。在一次实验中，我们向参与者展示了一个考试结果，然后呈现了一个句子。参与者需要判断，根据这个考试结果，最可能是哪种类型的说话者说出了这个句子。

【考试结果】
有5名学生参加了一场包含{total_questions}道题目的考试。每个学生答对的题目数量如下：
{exam_table}

【句子】
"{sentence}"

【三种说话者类型】
1. 教师 (high) - 目标是让考试听起来简单（强调高成功率）
2. 学生 (low) - 目标是让考试听起来困难（强调低成功率）
3. 考官 (info) - 目标是客观描述结果，不进行任何框架性修饰

【任务】
基于上述考试结果，判断最可能说出这个句子的是哪种类型的说话者。
请只输出一个词：high、low 或 info。
不要输出任何解释或其他内容。"""
    
    return prompt

def parse_response(response_text):
    """解析模型响应"""
    if not response_text:
        return "ERROR: 空响应"
    
    response_lower = response_text.lower().strip()
    
    if "high" in response_lower:
        return "high"
    elif "low" in response_lower:
        return "low"
    elif "info" in response_lower:
        return "info"
    else:
        return f"无效: {response_text[:50]}"

# ==================== API调用函数 ====================

def init_client(model_name):
    """初始化模型客户端"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"未知的模型名称：{model_name}")
    
    config = MODEL_CONFIGS[model_name]
    api_key = os.getenv(config["api_key_env"])
    
    if not api_key:
        raise ValueError(f"请设置环境变量 {config['api_key_env']}")
    
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
        raise ValueError(f"未知的客户端类型: {config['client_type']}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(model_name, prompt, temperature=0.7):
    """调用LLM"""
    client, config = init_client(model_name)
    
    try:
        if config["client_type"] == "openai":
            response = client.chat.completions.create(
                model=config["model_name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=10
            )
            return response.choices[0].message.content.strip()
    
        elif config["client_type"] == "gemini":
            response = client.generate_content(prompt)
            return response.text.strip()
    
    except Exception as e:
        print(f"API调用错误: {e}")
        raise e

# ==================== 主实验函数 ====================

def run_listener_experiment(
    stimuli_file,
    output_dir,
    runs_per_item,  
    models_to_run=None,
    total_questions=12,
    delay=2
):
    """
    运行听者实验
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取刺激文件
    print(f"读取刺激文件: {stimuli_file}")
    df = pd.read_csv(stimuli_file)
    print(f"共 {len(df)} 个刺激项目")
    
    # 默认运行所有模型
    if models_to_run is None:
        models_to_run = list(MODEL_CONFIGS.keys())
    
    # 为每个模型运行实验
    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"开始运行模型: {model_name}")
        print(f"{'='*60}")
        
        # 输出文件
        output_file = f"{output_dir}/listener_{model_name.replace('-', '_')}.csv"
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "itemID", "list", "condition", "Q1", "Q2", "A",
                "observation", "sentence", "run", "chosen_speaker",
                "raw_response", "timestamp"
            ])
            
            # 遍历每个刺激项目
            for idx, row in df.iterrows():
                item_id = row['itemID']
                exp_list = row['list']
                condition = row['condition']
                q1 = row['Q1']
                q2 = row['Q2']
                adj = row['A']
                obs_str = row['observation']
                
                # 解析observation
                correct_counts = parse_observation(obs_str)
                
                # 创建prompt
                prompt = create_prompt(
                    correct_counts, q1, q2, adj, total_questions
                )
                sentence = build_sentence(q1, q2, adj)
                
                # 多次运行
                for run in range(1, runs_per_item + 1):
                    print(f"  Item {item_id}, Run {run}/{runs_per_item}")
                    
                    try:
                        # 调用模型
                        raw_response = call_llm(model_name, prompt)
                        chosen = parse_response(raw_response)
                        
                        # 记录结果
                        writer.writerow([
                            item_id, exp_list, condition, q1, q2, adj,
                            obs_str, sentence, run, chosen,
                            raw_response, datetime.now().isoformat()
                        ])
                        csvfile.flush()
                        
                    except Exception as e:
                        print(f"    错误: {e}")
                        writer.writerow([
                            item_id, exp_list, condition, q1, q2, adj,
                            obs_str, sentence, run, "ERROR",
                            str(e), datetime.now().isoformat()
                        ])
                    
                    # 延迟，避免限流
                    time.sleep(delay)
        
        print(f"完成！结果保存至: {output_file}")
        
        # 模型之间休息
        if model_name != models_to_run[-1]:
            print("\n等待5秒后运行下一个模型...")
            time.sleep(5)
    
    print("\n所有模型实验完成！")

# ==================== 主程序 ====================

if __name__ == "__main__":
   
    # 配置参数
    RUNS_PER_ITEM = 1 # 先测试次数
    TOTAL_QUESTIONS = 12
    
    # 按次序测试
    MODELS = ["deepseek-v3.2"]
    #MODELS = list(MODEL_CONFIGS.keys())  # 运行所有模型
    
    print("="*60)
    print("LLM 听者实验 - 说话者类型归因")
    print("="*60)
    print(f"刺激文件: {STIMULI_FILE}")
    print(f"项目数量: 30")
    print(f"每个项目运行次数: {RUNS_PER_ITEM}")
    print(f"总调用次数: {30 * RUNS_PER_ITEM * len(MODELS)}")
    print(f"模型: {', '.join(MODELS)}")
    print("="*60)
    
    # 确认继续
    response = input("\n按回车键开始实验，或输入 'q' 退出: ")
    if response.lower() == 'q':
        print("实验已取消")
        exit()
    
    # 运行实验
    run_listener_experiment(
        stimuli_file=STIMULI_FILE,
        output_dir=OUTPUT_DIR,
        runs_per_item=RUNS_PER_ITEM,
        models_to_run=MODELS,
        total_questions=TOTAL_QUESTIONS,
        delay=2
    )