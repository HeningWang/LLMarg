### 1. Direct Label Prediction (直接标签预测)

**策略：** 模拟论文中的“标签评分（Label Scoring）”，强制模型在受限选项中选择，减少随机干扰。

- **English:**
  > Background: 5 students took a test consisting of 12 questions.
  > Exam results: {exam_data}
  > Sentence: "{sentence}"
  > Speaker Type Definitions:
  > - high: make the exam sound easy (high success rate)
  > - low: make the exam sound hard (low success rate)
  > - info: describe the results objectively, no framing
  > Task: Given an exam-results table and a sentence, decide which type of speaker most likely produced that sentence.
  > Output format: Directly output the label only (high, low, or info)

- **中文：**
  > 背景：有 5 名学生参加了一场包含 12 道题目的考试。
  > 考试结果：{exam_data}
  > 句子："{sentence}"
  > 说话者类型定义：
  > - high: 让考试听起来容易（高成功率）
  > - low: 让考试听起来难（低成功率）
  > - info: 客观描述结果，不带倾向性
  > 任务：根据给定的考试结果表格和句子，判断该句子最可能由哪种类型的说话者产生。
  > 输出格式：直接输出标签（high、low 或 info）。

---

### 2. Role-Description Prompting (角色描述提示)

**策略：** 在预测前明确描述目标（Speaker Goals），通过“语境启动”引导模型进入特定逻辑。

- **English:**
  > Background: This is an experiment. You will be shown an exam result, followed by a sentence. Based on the exam result, you need to determine which type of speaker most likely produced the sentence.
  >
  > Speaker Goals:
  > 1. Teacher (high): Make exam seem easy.
  > 2. Student (low): Make exam seem hard.
  > 3. Examiner (info): Objective description.
  >
  > Exam results: {exam_data}
  > Sentence: "{sentence}"
  >
  > Question: Which speaker type produced this? max 20 words.

- **中文：**
  > 背景：这是一个实验。实验会先展示一个考试结果，然后呈现一个句子。你需要根据这个考试结果，判断该句子最可能由哪种类型的说话者说出。
  >
  > 说话者目标：
  > 1. 教师 (high): 强调成功率，显简单。
  > 2. 学生 (low): 强调失败率，显困难。
  > 3. 考官 (info): 客观描述。
  >
  > 考试结果：{exam_data}
  > 句子："{sentence}"
  >
  > 问题：这句话最可能是哪种说话者说的？不超过 20 字。

---

### 3. Chain-of-Thought Probing (思路式探询)

**策略：** 激活模型的推理路径，通过中间推理步骤提高预测的稳定性。

- **English:**
  > Background: 5 students took a test consisting of 12 questions. This is an experiment. You will be shown an exam result, followed by a sentence. Based on the exam result, you need to determine which type of speaker most likely produced the sentence.
  >
  > Speaker Type Definitions:
  > - high (Teacher): make the exam sound easy (high success rate)
  > - low (Student): make the exam sound hard (low success rate)
  > - info (Examiner): describe the results objectively, no framing
  >
  > Exam results: {exam_data}
  > Sentence: "{sentence}"
  >
  > IMPORTANT: You MUST follow this EXACT format:
  > [One sentence analysis, max 20 words]
  > Final label: [high/low/info]
  >
  > Now provide your response:

- **中文：**
  > 背景：有 5 名学生参加了一场包含 12 道题目的考试。这是一个实验。实验会先展示一个考试结果，然后呈现一个句子。你需要根据这个考试结果，判断该句子最可能由哪种类型的说话者说出。
  >
  > 说话者类型定义：
  > - high（教师）：让考试听起来容易（高成功率）
  > - low（学生）：让考试听起来难（低成功率）
  > - info（考官）：客观描述结果，不带倾向性
  >
  > 考试结果：{exam_data}
  > 句子："{sentence}"
  >
  > 重要：你必须严格按照以下格式回答：
  > [一句分析，最多 20 个字]，并且最终标签：[high/low/info]
  >
  > 现在请回答：

---

### 4. Structured Justification (结构化辩护探究)

**策略：** 强制分离“证据”与“结果”，防止模型因为单纯的字面频率而产生偏差。

- **English:**
  > Background: 5 students took a test consisting of 12 questions. This is an experiment. You will be shown an exam result, followed by a sentence. Based on the exam result, you need to determine which type of speaker most likely produced the sentence.
  >
  > Speaker Type Definitions:
  > - high (Teacher): make the exam sound easy (high success rate)
  > - low (Student): make the exam sound hard (low success rate)
  > - info (Examiner): describe the results objectively, no framing
  >
  > Exam results: {exam_data}
  > Sentence: "{sentence}"
  >
  > Provide your response in the following EXACT format:
  > Evidence: [one short sentence, max 10 words]
  > Label: [high, low, or info]

- **中文：**
  > 背景：有 5 名学生参加了一场包含 12 道题目的考试。这是一个实验。实验会先展示一个考试结果，然后呈现一个句子。你需要根据这个考试结果，判断该句子最可能由哪种类型的说话者说出。
  >
  > 说话者类型定义：
  > - high（教师）：让考试听起来容易（高成功率）
  > - low（学生）：让考试听起来难（低成功率）
  > - info（考官）：客观描述结果，不带倾向性
  >
  > 考试结果：{exam_data}
  > 句子："{sentence}"
  >
  > 请严格按照以下格式回答：
  > 证据：[一句简短理由，最多 10 个字]
  > 标签：[high, low, info]

---

### 5. Calibration Variant (校准变体)

**策略：** 获取置信度分数，辅助 WM 识别哪些题目在不同评分方法下更易波动。

- **English:**
  > Background: 5 students took a test consisting of 12 questions. This is an experiment. You will be shown an exam result, followed by a sentence. Based on the exam result, you need to determine which type of speaker most likely produced the sentence.
  >
  > Speaker Type Definitions:
  > - high (Teacher): make the exam sound easy (high success rate)
  > - low (Student): make the exam sound hard (low success rate)
  > - info (Examiner): describe the results objectively, no framing
  >
  > Exam results: {exam_data}
  > Sentence: "{sentence}"
  >
  > Respond with EXACTLY this format:
  > Label: [high/low/info]
  > Confidence: [0.0 to 1.0]

- **中文：**
  > 背景：有 5 名学生参加了一场包含 12 道题目的考试。这是一个实验。实验会先展示一个考试结果，然后呈现一个句子。你需要根据这个考试结果，判断该句子最可能由哪种类型的说话者说出。
  >
  > 说话者类型定义：
  > - high（教师）：让考试听起来容易（高成功率）
  > - low（学生）：让考试听起来难（低成功率）
  > - info（考官）：客观描述结果，不带倾向性
  >
  > 考试结果：{exam_data}
  > 句子："{sentence}"
  >
  > 请严格按照以下格式回答：
  > 标签：[high/low/info]
  > 置信度：[0.0 到 1.0]