import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import ttest_ind

# 设置路径
DATA_DIR = '../results'
SAVE_DIR = '../results'
os.makedirs(SAVE_DIR, exist_ok=True)

def run_analysis():
    # 1. 加载数据
    try:
        df_ds = pd.read_csv(os.path.join(DATA_DIR, 'listener_deepseek_v3.2.csv'))
        df_gpt = pd.read_csv(os.path.join(DATA_DIR, 'listener_gpt_4o_mini.csv'))
        df_ds['model'] = 'DeepSeek-V3.2'
        df_gpt['model'] = 'GPT-4o-mini'
        df = pd.concat([df_ds, df_gpt])
    except FileNotFoundError as e:
        print(f"错误: 找不到文件。请确保 CSV 放在 data 文件夹中。\n{e}")
        return

    # 2. 计算匹配率 (Match Rate)
    # 将选择结果与 condition 列对比，计算准确率
    df['is_match'] = (df['chosen_speaker'].str.lower() == df['condition'].str.lower()).astype(int)

    # 3. 按模型和条件汇总
    summary = df.groupby(['model', 'condition'])['is_match'].mean().reset_index()
    
    # 4. 统计对比：论证性 (high/low) vs 客观 (info)
    print("=== 统计分析报告 ===")
    for model in df['model'].unique():
        m_data = df[df['model'] == model]
        arg_perf = m_data[m_data['condition'].isin(['high', 'low'])]['is_match']
        info_perf = m_data[m_data['condition'] == 'info']['is_match']
        
        t_stat, p_val = ttest_ind(arg_perf, info_perf)
        print(f"\n模型: {model}")
        print(f"  论证任务 (high/low) 准确率: {arg_perf.mean():.2%}")
        print(f"  客观任务 (info) 准确率:     {info_perf.mean():.2%}")
        print(f"  显著性差异 (p-value):       {p_val:.4f}")

    # 5. 绘图：模型表现对比
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.barplot(data=summary, x='condition', y='is_match', hue='model', order=['low', 'info', 'high'])
    
    plt.axhline(0.33, ls='--', color='red', label='Random Chance (1/3)')
    plt.title('Experiment 2: LLM Speaker Attribution Performance', fontsize=14)
    plt.ylabel('Proportion of Match Responses', fontsize=12)
    plt.xlabel('Speaker Type (Condition)', fontsize=12)
    plt.ylim(0, 1)
    plt.legend()
    
    # 保存结果
    plt.savefig(os.path.join(SAVE_DIR, 'model_comparison_plot.png'))
    summary.to_csv(os.path.join(SAVE_DIR, 'match_rates_summary.csv'), index=False)
    print(f"\n分析完成！图表和汇总表已保存至 {SAVE_DIR}")

if __name__ == "__main__":
    run_analysis()