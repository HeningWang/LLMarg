# experiment_config.py
"""
Experiment configuration for the LLM Listener Grid Search.
Defines all axes: probing methods, temperatures, models.
"""

from dataclasses import dataclass


# ==================== Core Prompt (shared across methods) ====================
# Role descriptions are EXACT copies from the human listener experiment.
# Background and question match the human experiment framing.

CORE_PROMPT = """Background: A school recently administered an exam with {total_questions} questions to {n_students} students.
Exam results: {exam_data}

Three types of people described the results:
- Student: wants the exam to sound harder (as if students had a low chance of success), because this makes her own performance appear better by comparison.
- Teacher: wants the exam to sound easier (as if students had a high chance of success), because this reflects positively on teaching quality and students' preparedness.
- Examiner: aims to provide an objective and neutral description of the exam, without making it sound particularly hard or easy.

Someone said: "{sentence}"
"""

# ==================== Probing Method Templates ====================
# Each method appends a probing-specific instruction to the core prompt.

PROMPT_TEMPLATES = {
    "direct_label": {
        "description": "Baseline forced choice with role names (Student/Teacher/Examiner). Matches human response format.",
        "suffix": """Who is most likely to have said this report? Answer with one word only: Student, Teacher, or Examiner.""",
        "response_type": "role_name",
        "max_tokens": 10,
    },
    "meta_label": {
        "description": "Same task but with abstract meta-labels (high/low/info) instead of role names.",
        "suffix": """Who is most likely to have said this report? Answer with one word only: high, low, or info.""",
        "response_type": "meta_label",
        "max_tokens": 10,
    },
    "chain_of_thought": {
        "description": "Reasoning before answer. Forces one sentence of analysis before the label.",
        "suffix": """Who is most likely to have said this report? First give a one-sentence analysis (max 20 words), then answer.
Format:
Analysis: [your reasoning]
Answer: [Student/Teacher/Examiner]""",
        "response_type": "role_name",
        "max_tokens": 100,
    },
    "self_reported_confidence": {
        "description": "Answer plus self-reported confidence score (0.0 to 1.0).",
        "suffix": """Who is most likely to have said this report? Respond in this exact format:
Answer: [Student/Teacher/Examiner]
Confidence: [0.0 to 1.0]""",
        "response_type": "role_name",
        "max_tokens": 30,
    },
    "log_probs": {
        "description": "Forced single-word answer with logprob extraction.",
        "suffix": """Who is most likely to have said this report? Answer with one word only: Student, Teacher, or Examiner.""",
        "response_type": "role_name",
        "max_tokens": 5,
        "requires_logprobs": True,
    },
}

# Response label mappings
ROLE_TO_CONDITION = {
    "student": "low",
    "teacher": "high",
    "examiner": "info",
}

CONDITION_TO_ROLE = {v: k for k, v in ROLE_TO_CONDITION.items()}

# Method lists
TEXT_PROBING_METHODS = [
    "direct_label",
    "meta_label",
    "chain_of_thought",
    "self_reported_confidence",
]

LOGPROBS_PROBING_METHODS = [
    "log_probs",
]

ALL_PROBING_METHODS = TEXT_PROBING_METHODS + LOGPROBS_PROBING_METHODS

# Temperature grid
TEMPERATURES = [0.2, 0.7, 1.2]

# ==================== Model Configurations ====================

MODEL_CONFIGS = {
    "claude-sonnet-4-6-thinking": {
        "client_type": "openai",
        "api_key_env": "LAOZHANG_API_KEY",
        "model_name": "claude-sonnet-4-6-thinking",
        "base_url": "https://api.laozhang.ai/v1",
        "supports_logprobs": False,
        "family": "claude",
        "size": "large",
    },
    "deepseek-v3.2": {
        "client_type": "openai",
        "api_key_env": "SILICONFLOW_API_KEY",
        "model_name": "deepseek-ai/DeepSeek-V3.2",
        "base_url": "https://api.siliconflow.cn/v1",
        "supports_logprobs": True,
        "family": "deepseek",
        "size": "large",
    },
    "gpt-4o-mini": {
        "client_type": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "model_name": "gpt-4o-mini",
        "base_url": "https://free.v36.cm/v1",
        "supports_logprobs": True,
        "family": "openai",
        "size": "small",
    },
    "Qwen3.5-35B-A3B": {
        "client_type": "openai",
        "api_key_env": "Huggingface_API_KEY",
        "model_name": "Qwen/Qwen3.5-35B-A3B:novita",
        "base_url": "https://router.huggingface.co/v1",
        "supports_logprobs": True,
        "family": "qwen",
        "size": "mid",
    },
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable configuration for a single experiment run."""
    models: tuple[str, ...]
    probing_methods: tuple[str, ...]
    temperatures: tuple[float, ...]
    runs_per_item: int = 20
    max_items: int = 30
    total_questions: int = 12
    language: str = "en"
    delay: float = 2.0
    output_dir: str = "results"


# ==================== Preset Configs ====================

PHASE1_PILOT_CONFIG = ExperimentConfig(
    models=("Qwen3.5-35B-A3B",),
    probing_methods=tuple(TEXT_PROBING_METHODS),
    temperatures=tuple(TEMPERATURES),
    runs_per_item=20,
    max_items=30,
)

PHASE1_VERIFY_CONFIG = ExperimentConfig(
    models=("Qwen3.5-35B-A3B",),
    probing_methods=tuple(ALL_PROBING_METHODS),
    temperatures=tuple(TEMPERATURES),
    runs_per_item=1,
    max_items=1,
)
