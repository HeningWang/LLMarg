# LLM Listener Experiment

辩论中的大语言模型听者实验研究

## 项目概述

本项目研究大语言模型（LLM）在辩论场景中作为听者（Listener）的理解能力。实验通过不同的量词组合和考试结果数据，测试模型对说话者意图的判断能力。

## 实验设计

### 实验条件
- **info**：信息性条件（中性描述）
- **high**：高信息量条件（强调高成功率）  
- **low**：低信息量条件（强调低成功率）

### 说话者类型
- **教师 (high)** - 让考试听起来简单
- **学生 (low)** - 让考试听起来困难
- **考官 (info)** - 客观描述结果

### 数据格式
实验使用CSV文件存储刺激材料和结果数据：
- `listener_stimuli.csv` - 实验刺激材料
- `results/` - 实验结果目录

## 模型测试

测试了以下LLM模型：
- DeepSeek-V3.2
- GPT-4o-mini  
- Claude Sonnet 4.6

## 项目结构

```
llm_listener_experiment/
├── listener_stimuli.csv      # 实验刺激材料
├── venv/listener_experiment.py  # 实验主程序
├── results/                  # 实验结果
│   ├── analyze/             # 分析结果
│   │   ├── match_rates_summary.csv
│   │   ├── model_comparison_plot.png
│   │   └── ...
│   ├── listener_deepseek_v3.2.csv
│   ├── listener_gpt_4o_mini.csv
│   └── ...
└── README.md
```

## 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置API密钥（在.env文件中）：
```
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
SILICONFLOW_API_KEY=your_key
```

3. 运行实验：
```bash
python venv/listener_experiment.py
```

## 实验结果

详细的实验结果和分析请参考`results/`目录下的文件。注意注意，实验结果输出的目录设置可能有点问题，大家自行查看。
