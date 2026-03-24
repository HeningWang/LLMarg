提取和使用对数概率（Log Probabilities，简称 **logprobs**）是评估大模型输出可靠性、降低幻觉以及计算困惑度（Perplexity）的关键技术。在实际操作中，你需要从模型选择、参数配置和代码实现逻辑三个方面注意以下要点：

### 1. 模型与 API 的选择
并非所有模型或平台都支持提取 logprobs，选择时需注意：
*   **主流 API 支持**：OpenAI 的 GPT 系列（如 GPT-4、GPT-3.5）通过在请求中设置 `logprobs=True` 来返回每个 token 的概率信息。Ollama 推理框架同样支持在 `options` 中开启此参数。
*   **开源模型**：Meta 的 **Llama 3** 官方代码库在 `generate` 方法中提供了 `logprobs` 选项。**Hugging Face** 的 `transformers` 库则通过 `compute_transition_scores` 函数来暴露生成 token 的概率。
*   **推理引擎**：**vLLM** 引擎支持通过 `SamplingParams` 设置 `logprobs` 数量，其实现遵循 OpenAI API 标准。
*   **限制注意**：部分特定模型（如早期文档中的 `gpt-4-vision-preview`）可能不支持此选项。

### 2. 关键参数设置
*   **logprobs 与 top_logprobs**：
    *   `logprobs`（布尔值）：决定是否返回输出 token 的对数概率。
    *   `top_logprobs`（整数，通常 0-5）：指定在每个 token 位置返回多少个最可能的候选 token 及其概率。使用此参数时，必须先将 `logprobs` 设为 `True`。
*   **温度 (Temperature)**：
    *   若需最确定的结果（贪婪采样），应将 `temperature` 设为 **0**。
    *   更高的温度会重新分配概率分布，增加罕见词出现的可能性，提高创造性但降低一致性。

### 3. 代码格式与实现逻辑
在编写代码提取概率时，存在一些常见的“陷阱”：
*   **Token 对齐偏移（Hugging Face 特有）**：在 Hugging Face 的实现中，模型返回的是对“下一个 token”的预测。因此，位置 $i$ 的 logits 实际上对应的是输入序列中位置 $i+1$ 的 token。在配对 token 和概率时，代码逻辑需要进行**一位偏移**。
*   **归一化问题**：Hugging Face 默认返回的 `scores` 往往是**未归一化**的原始对数概率。为了确保所有词汇概率总和为 1，建议在 `generate` 中设置 `renormalize_logits=True`，或在调用 `compute_transition_scores` 时开启 `normalize_logits=True`。
*   **结果排序**：
    *   在 vLLM 中，被采样的 token 概率通常位于返回列表的**第一个位置**。
    *   由于 API 总是返回被采样 token 的概率，结果中可能包含最多 `logprobs + 1` 个元素（如果采样的词不在 top $n$ 候选词中）。

### 4. 使用时的核心注意事项
*   **概率范围**：logprobs 是负数或 0.0。**0.0 对应 100% 的概率**。
*   **联合概率计算**：序列的整体概率等于序列中各 token 概率之和（在对数空间中相加，等同于在概率空间中相乘）。
*   **应对幻觉**：低对数概率通常意味着模型在“瞎编”或信心不足。你可以设置一个信心阈值，自动拒绝低质量响应或引入专家验证。
*   **数值稳定性**：使用对数尺度（log scale）是为了防止计算机在处理极小概率时出现**下溢（underflow）**错误。
