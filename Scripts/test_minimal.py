import os
import time
from dotenv import load_dotenv
import openai

load_dotenv()

def minimal_test():
    """最小化测试 - 只测试一个item"""
    
    api_key = os.getenv("GEMINI_API_KEY")
    client = openai.OpenAI(
        api_key=api_key,
        base_url= "https://generativelanguage.googleapis.com/v1"
    )
    
    # 使用最简单的prompt
    prompt = """你是一位语言理解专家。请判断最可能说出以下句子的是哪种类型的说话者。

【考试结果】
学生1: 答对3/12题 (25%)
学生2: 答对3/12题 (25%)
学生3: 答对3/12题 (25%)
学生4: 答对3/12题 (25%)
学生5: 答对3/12题 (25%)

【句子】
"all of the students got some of the answers wrong"

【三种说话者类型】
1. 教师 (high) - 目标是让考试听起来简单
2. 学生 (low) - 目标是让考试听起来困难
3. 考官 (info) - 目标是客观描述结果

请只输出一个词：high、low 或 info。"""

    print("="*50)
    print("最小化测试 - 开始")
    print("="*50)
    
    try:
        print("发送请求...")
        response = client.chat.completions.create(
            model="Gemini 2.5 Pro", #要更改模型名称
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=10,
            timeout=120
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ 成功!")
        print(f"响应: '{result}'")
        
    except Exception as e:
        print(f"❌ 失败: {type(e).__name__}")
        print(f"错误信息: {e}")
    
    print("="*50)

if __name__ == "__main__":
    minimal_test()