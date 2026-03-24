# 测试api链接
import os
from dotenv import load_dotenv
import openai

# 加载环境变量
load_dotenv()

def test_deepseek_connection():
    """测试API连接"""
    
    api_key = os.getenv("Huggingface_API_KEY")
    if not api_key:
        print("❌ 错误: 没有找到 Huggingface_API_KEY 环境变量")
        return False
    
    print(f"✅ 找到API密钥")
    
    try:
        # 创建客户端
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://router.huggingface.co/v1"
        )
        
        print("正在尝试连接 API...")
        
        # 尝试一个简单的调用
        response = client.chat.completions.create(
            model="zai-org/GLM-5:novita",
            messages=[{"role": "user", "content": "Say 'test'"}],
            temperature=0,
            max_tokens=10,
            timeout=30  # 设置30秒超时
        )
        
        print(f"✅ 连接成功！")
        print(f"响应内容: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ 连接失败: {type(e).__name__}")
        print(f"错误信息: {e}")
        return False

if __name__ == "__main__":
    print("="*50)
    print("DeepSeek API 连接测试")
    print("="*50)
    test_deepseek_connection()