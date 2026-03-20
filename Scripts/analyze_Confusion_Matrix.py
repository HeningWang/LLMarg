import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 配置路径
DATA_DIR = '../results/'
SAVE_DIR = '../results/analyze/Confusion_Matrix/'


def plot_confusion_matrix():
    # 加载数据
    try:
        df_ds = pd.read_csv(os.path.join(DATA_DIR, 'listener_deepseek_v3.2.csv'))
        df_gpt = pd.read_csv(os.path.join(DATA_DIR, 'listener_gpt_4o_mini.csv'))
        df_ds['model'] = 'DeepSeek-V3.2'
        df_gpt['model'] = 'GPT-4o-mini'
        df = pd.concat([df_ds, df_gpt])
    except:
        print("请确保数据文件已准备好。")
        return

    models = df['model'].unique()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, model in enumerate(models):
        m_data = df[df['model'] == model]
        # 创建交叉表（归一化到行，即：给定真实条件下，预测分布的比例）
        matrix = pd.crosstab(
            m_data['condition'], 
            m_data['chosen_speaker'], 
            normalize='index'
        )
        
        # 确保顺序一致 [low, info, high]
        order = ['low', 'info', 'high']
        matrix = matrix.reindex(index=order, columns=order).fillna(0)

        sns.heatmap(matrix, annot=True, cmap='Blues', fmt='.2%', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix: {model}')
        axes[i].set_xlabel('Predicted Speaker')
        axes[i].set_ylabel('Ground Truth (Condition)')

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrices.png'))
    plt.show()

if __name__ == "__main__":
    plot_confusion_matrix()