# Phase 1 Report: LLM Listener Grid Search

## 1. Overview

This report presents the results of Phase 1 of the LLM Listener Grid Search experiment. The goal is to test whether large language models can infer speaker intent from quantified descriptions of exam results, and to compare LLM predictions against human listener data.

**Experiment design:**
- 30 stimulus items across 3 conditions: *high* (Teacher framing), *low* (Student framing), *info* (Examiner, objective)
- Sentence template: `"{Q1} of the students got {Q2} of the answers {right/wrong}"`
- Human task: forced choice among Teacher / Student / Examiner
- LLM task: forced choice among high / low / info

**Phase 1 scope:**
- Model: Qwen2.5-7B (via Ollama on NVIDIA A6000)
- 5 probing methods x 3 temperatures = 15 conditions
- 20 runs per item, 30 items = 9,000 total API calls
- 0 errors (100% success rate)

---

## 2. Probing Methods

Five text-based probing methods were tested, informed by Tsvilodub et al. (2024) and Wang et al. (2024). Each prompt includes exam data and a sentence, and asks the model to classify the speaker type.

### 2.1 Direct Label

Simple forced choice without explanation. Maps to *label scoring* in the probing literature.

```
Background: 5 students took a test consisting of 12 questions.
Exam results: {exam_data}
Sentence: "{sentence}"
Speaker Type Definitions:
- high: make the exam sound easy (high success rate)
- low: make the exam sound hard (low success rate)
- info: describe the results objectively, no framing
Task: Given an exam-results table and a sentence, decide which type of speaker most likely produced that sentence.
Output format: Directly output the label only (high, low, or info)
```

### 2.2 Role Description

Contextual priming: explicitly defining speaker motivations before prediction.

```
Background: This is an experiment. You will be shown an exam result, followed by a sentence. Based on the exam result, you need to determine which type of speaker most likely produced the sentence.

Speaker Goals:
1. Teacher (high): Make exam seem easy.
2. Student (low): Make exam seem hard.
3. Examiner (info): Objective description.

Exam results: {exam_data}
Sentence: "{sentence}"

Question: Which speaker type produced this? Answer with only one word: high, low, or info.
```

### 2.3 Chain of Thought

Reasoning: forcing the model to analyze before labeling. Maps to *free generation* in the probing literature.

```
Background: 5 students took a test consisting of 12 questions. This is an experiment. You will be shown an exam result, followed by a sentence. Based on the exam result, you need to determine which type of speaker most likely produced the sentence.

Speaker Type Definitions:
- high (Teacher): make the exam sound easy (high success rate)
- low (Student): make the exam sound hard (low success rate)
- info (Examiner): describe the results objectively, no framing

Exam results: {exam_data}
Sentence: "{sentence}"

IMPORTANT: You MUST follow this EXACT format:
[One sentence analysis, max 20 words]
Final label: [high/low/info]

Now provide your response:
```

### 2.4 Structured Justification

Evidence separation: forcing model to separate evidence from conclusion.

```
Background: 5 students took a test consisting of 12 questions. This is an experiment. You will be shown an exam result, followed by a sentence. Based on the exam result, you need to determine which type of speaker most likely produced the sentence.

Speaker Type Definitions:
- high (Teacher): make the exam sound easy (high success rate)
- low (Student): make the exam sound hard (low success rate)
- info (Examiner): describe the results objectively, no framing

Exam results: {exam_data}
Sentence: "{sentence}"

Provide your response in the following EXACT format:
Evidence: [one short sentence, max 10 words]
Label: [high, low, or info]
```

### 2.5 Calibration

Confidence: label plus self-reported confidence score.

```
Background: 5 students took a test consisting of 12 questions. This is an experiment. You will be shown an exam result, followed by a sentence. Based on the exam result, you need to determine which type of speaker most likely produced the sentence.

Speaker Type Definitions:
- high (Teacher): make the exam sound easy (high success rate)
- low (Student): make the exam sound hard (low success rate)
- info (Examiner): describe the results objectively, no framing

Exam results: {exam_data}
Sentence: "{sentence}"

Respond with EXACTLY this format:
Label: [high/low/info]
Confidence: [0.0 to 1.0]
```

---

## 3. Human Baseline

Human data was collected from N=242 participants (3,630 experimental trials) via Prolific. Each participant completed a forced-choice task identifying which speaker type (Teacher/Student/Examiner) most likely produced a given utterance.

### Human match rates

| Condition | Match Rate | Interpretation |
|-----------|-----------|----------------|
| **high** | 63.3% | Reliably above chance |
| **low** | 54.4% | Above chance |
| **info** | 36.3% | Near chance (33%) |

### Human response distribution

![Human Baseline](analyze/human_baseline_plot.png)

The core human pattern: listeners reliably detect argumentative framing in the *high* and *low* conditions, but struggle with *info* (near chance). This asymmetry is the primary target for LLM comparison.

---

## 4. LLM Results

1. overview of the grid search space (5 methods x 4 model family) with fixed temperature (show which fixed temperature we want to use given previous pilot run with QWEN2) and we use reasonalbe large models for A6000
2. show the main result plot(this plot should have three conditions on x axis, fill with bars: one human and 5 probing methods, facet by models)
3. show a focused comparion between direct_labels and meta_label
4. show a focused comparison between self reported confidence and log_probs

---

## 5. Key Findings


## References

- Tsvilodub, P., Wang, H., Grosch, S., & Franke, M. (2024). Predictions from language models for multiple-choice tasks are not robust under variation of scoring methods. *arXiv:2403.00998*.
- Wang, X., Ma, B., Hu, C., Weber-Genzel, L., Roettger, P., Kreuter, F., Hovy, D., & Plank, B. (2024). "My Answer is C": First-Token Probabilities Do Not Match Text Answers in Instruction-Tuned Language Models. *Findings of ACL 2024*, 7407-7416.
