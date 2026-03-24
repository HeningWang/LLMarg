import os
import time
from dotenv import load_dotenv
import openai
import json

load_dotenv()

def minimal_test():
    """最小化测试 - 测试 log probabilities 提取"""
    
    api_key = os.getenv("Huggingface_API_KEY")
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://router.huggingface.co/v1"
    )
    
    # 使用最简单的 prompt
    prompt = """Background: This is an experiment. You will be shown an exam result, followed by a sentence. Based on the exam result, you need to determine which type of speaker most likely produced the sentence.

Speaker Goals:
1. Teacher (high): Make exam seem easy.
2. Student (low): Make exam seem hard.
3. Examiner (info): Objective description.

Exam results: Student 1: 3/12 correct, Student 2: 3/12 correct, Student 3: 3/12 correct, Student 4: 3/12 correct, Student 5: 3/12 correct
Sentence: "all of the students got some of the answers wrong"

Question: Which speaker type produced this? Output only one word: high, low, or info."""

    print("="*60)
    print("最小化测试 - Log Probabilities 提取")
    print("="*60)
    
    try:
        print("发送请求（启用 logprobs）...")
        response = client.chat.completions.create(
            model="zai-org/GLM-5:novita",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=10,
            timeout=120,
            logprobs=True,  # 启用 log probabilities
            top_logprobs=5  # 返回前 5 个候选 token 的概率
        )
        
        # 提取响应文本
        result = response.choices[0].message.content.strip()
        print(f"\n✅ 请求成功!")
        print(f"📝 响应：'{result}'")
        
        # 提取 log probabilities 数据
        print("\n" + "="*60)
        print("📊 Log Probabilities - 模型输出分布信息")
        print("="*60)
        
        logprobs_data = None
        
        # 尝试从 choice.logprobs 提取
        if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
            logprobs_data = response.choices[0].logprobs
            print(f"✓ 找到 logprobs 数据")
        
        # 尝试从 message.logprobs 提取
        if not logprobs_data and hasattr(response.choices[0].message, 'logprobs'):
            logprobs_data = response.choices[0].message.logprobs
            print(f"✓ 找到 message.logprobs 数据")
        
        if logprobs_data:
            print(f"\n📋 Logprobs 数据结构:")
            print(f"  类型：{type(logprobs_data)}")
            
            # 如果有 content 属性（token 级别的 logprobs）
            if hasattr(logprobs_data, 'content') and logprobs_data.content:
                print(f"\n🔤 Token 级别的 Log Probabilities:")
                print("-"*60)
                
                for i, token_info in enumerate(logprobs_data.content[:10]):  # 只显示前 10 个 token
                    token = getattr(token_info, 'token', 'N/A')
                    logprob = getattr(token_info, 'logprob', 'N/A')
                    prob = 2.718281828 ** logprob if isinstance(logprob, (int, float)) else 'N/A'
                    
                    print(f"  Token {i+1}: '{token}'")
                    print(f"    Log Prob: {logprob:.4f}" if isinstance(logprob, (int, float)) else f"    Log Prob: {logprob}")
                    print(f"    Probability: {prob:.4f}" if isinstance(prob, (int, float)) else f"    Probability: {prob}")
                    
                    # 显示 top_logprobs（候选 token 分布）
                    if hasattr(token_info, 'top_logprobs') and token_info.top_logprobs:
                        print(f"    Top Logprobs:")
                        for top_token in token_info.top_logprobs:
                            top_token_str = getattr(top_token, 'token', 'N/A')
                            top_logprob = getattr(top_token, 'logprob', 'N/A')
                            top_prob = 2.718281828 ** top_logprob if isinstance(top_logprob, (int, float)) else 'N/A'
                            print(f"      - '{top_token_str}': logprob={top_logprob:.4f}, prob={top_prob:.4f}" if isinstance(top_logprob, (int, float)) else f"      - '{top_token_str}': logprob={top_logprob}")
                    print()
            
            # 转换为字典保存
            logprobs_dict = {}
            if hasattr(logprobs_data, 'model_dump'):
                logprobs_dict = logprobs_data.model_dump()
            elif hasattr(logprobs_data, '__dict__'):
                logprobs_dict = str(logprobs_data)
            
            print("\n💾 完整 Logprobs 数据（JSON 格式）:")
            print("-"*60)
            print(json.dumps(logprobs_dict, indent=2, ensure_ascii=False, default=str)[:2000] + "...")
            
            # 提取关键标签的概率分布
            print("\n" + "="*60)
            print("🎯 目标标签 (high/low/info) 概率分析")
            print("="*60)
            
            label_probs = {}
            if hasattr(logprobs_data, 'content') and logprobs_data.content:
                for token_info in logprobs_data.content:
                    token = getattr(token_info, 'token', '').lower().strip()
                    logprob = getattr(token_info, 'logprob', None)
                    
                    if token in ['high', 'low', 'info'] and logprob is not None:
                        prob = 2.718281828 ** logprob
                        label_probs[token] = {
                            'logprob': logprob,
                            'probability': prob
                        }
                    
                    # 检查 top_logprobs 中是否包含目标标签
                    if hasattr(token_info, 'top_logprobs') and token_info.top_logprobs:
                        for top_token in token_info.top_logprobs:
                            top_token_str = getattr(top_token, 'token', '').lower().strip()
                            top_logprob = getattr(top_token, 'logprob', None)
                            
                            if top_token_str in ['high', 'low', 'info'] and top_logprob is not None:
                                top_prob = 2.718281828 ** top_logprob
                                if top_token_str not in label_probs or label_probs[top_token_str]['probability'] < top_prob:
                                    label_probs[top_token_str] = {
                                        'logprob': top_logprob,
                                        'probability': top_prob
                                    }
            
            if label_probs:
                print(f"\n  标签概率分布:")
                for label in sorted(label_probs.keys(), key=lambda x: label_probs[x]['probability'], reverse=True):
                    info = label_probs[label]
                    bar = '█' * int(info['probability'] * 20)
                    print(f"    {label:6s}: prob={info['probability']:.4f}, logprob={info['logprob']:.4f} {bar}")
                
                # 预测最可能的标签
                best_label = max(label_probs.keys(), key=lambda x: label_probs[x]['probability'])
                print(f"\n  🏆 最可能的标签：{best_label} (probability={label_probs[best_label]['probability']:.4f})")
            else:
                print(f"\n  ⚠️  未在 logprobs 中找到 high/low/info 标签")
                print(f"  提示：模型可能使用了不同的 tokenization 方式")
        
        else:
            print(f"\n⚠️  警告：API 返回中未找到 logprobs 数据")
            print(f"  可能原因：")
            print(f"    1. 该模型不支持 logprobs")
            print(f"    2. HuggingFace Inference API 限制")
            print(f"    3. 需要更换支持 logprobs 的 API 提供商")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n❌ 失败：{type(e).__name__}")
        print(f"错误信息：{e}")
        
        # 检查是否是 402 额度错误
        error_str = str(e)
        if "402" in error_str or "depleted" in error_str.lower():
            print("\n💡 提示：账户额度已用尽，请更换 API 密钥或模型")
        elif "logprobs" in error_str.lower():
            print("\n💡 提示：该模型或 API 不支持 logprobs 参数")
    
    print("="*60)

if __name__ == "__main__":
    minimal_test()