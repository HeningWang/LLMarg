import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 配置路径
DATA_DIR = '../results' # 请确保 csv 文件在此目录下
SAVE_DIR = '../results/analyze/Item_Comparison'
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_item_performance():
    # 1. 加载并合并数据
    try:
        
        df_ds = pd.read_csv(os.path.join(DATA_DIR, 'listener_deepseek_v3.2.csv'))
        df_gpt = pd.read_csv(os.path.join(DATA_DIR, 'listener_gpt_4o_mini.csv'))
        
        df_ds['model'] = 'DeepSeek-V3.2'
        df_gpt['model'] = 'GPT-4o-mini'
        df = pd.concat([df_ds, df_gpt], ignore_index=True)
    except Exception as e:
        print(f"读取文件失败，请检查路径和文件名: {e}")
        return

    # 2. 预处理：判定是否匹配
    # 统一转为小写处理，增强鲁棒性
    df['is_match'] = (df['chosen_speaker'].str.strip().str.lower() == 
                      df['condition'].str.strip().str.lower()).astype(int)

    # 3. 按 condition 分组生成三张图
    conditions = ['low', 'info', 'high']
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    
    for cond in conditions:
        # 筛选当前 condition 的数据
        cond_df = df[df['condition'].str.lower() == cond]
        
        if cond_df.empty:
            print(f"警告：未找到 condition 为 {cond} 的数据。")
            continue
            
        # 计算每个 Item 的平均准确率（Match Rate）
        # 同时保留 sentence 以便参考（如果同一个 ID 对应同一个句子）
        item_stats = cond_df.groupby(['itemID', 'model', 'sentence'])['is_match'].mean().reset_index()

        plt.figure(figsize=(12, 6))
        
        # 绘制对比柱状图
        ax = sns.barplot(
            data=item_stats,
            x='itemID',
            y='is_match',
            hue='model',
            palette={'DeepSeek-V3.2': '#1f77b4', 'GPT-4o-mini': '#ff7f0e'}
        )

        # 添加装饰
        plt.title(f'Item-by-Item Accuracy Comparison: Condition = {cond.upper()}', fontsize=15)
        plt.ylabel('Match Rate (Accuracy)', fontsize=12)
        plt.xlabel('Item ID', fontsize=12)
        plt.ylim(0, 1.1)  # 留出空间显示图例
        plt.axhline(y=0.33, color='red', linestyle='--', alpha=0.6, label='Random (1/3)')
        plt.legend(title='Model', loc='upper right')

        # 优化 x 轴标签旋转
        plt.xticks(rotation=0)
        
        # 保存图片
        file_name = f'item_perf_{cond}.png'
        save_path = os.path.join(SAVE_DIR, file_name)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        print(f"已生成对比图：{save_path}")

if __name__ == "__main__":
    plot_item_performance()