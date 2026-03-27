# speaker_config.py
"""
Experiment configuration for the LLM Speaker-side experiment.
Production task: given exam results + framing goal, produce an utterance.
"""

from dataclasses import dataclass


# ==================== Core Prompt ====================

SPEAKER_CORE_PROMPT = """Background: A school recently administered an exam with {n_questions} questions to {n_students} students.
Exam results: {exam_data}

Your task: Describe these exam results using a sentence of the exact form:
"[OQ] of the students got [IQ] of the answers [ADJ]"

where:
- OQ (outer quantifier): none, some, most, or all
- IQ (inner quantifier): none, some, most, or all
- ADJ: right or wrong

The sentence must be truthful (not contradict the exam results).
"""

# ==================== Output Format Variants ====================

SPEAKER_METHODS = {
    "full_sentence": {
        "description": "Output the complete sentence in template format.",
        "suffix": """Describe these results so as to make it appear as if there is a {framing} success rate without lying.
Answer in the exact format only: [OQ] of the students got [IQ] of the answers [ADJ]""",
        "max_tokens": 30,
    },
    "three_blanks": {
        "description": "Output only the three blanks (OQ, IQ, ADJ) as comma-separated values.",
        "suffix": """Describe these results so as to make it appear as if there is a {framing} success rate without lying.
Answer with exactly three words separated by commas: [OQ], [IQ], [ADJ]""",
        "max_tokens": 15,
    },
}

# Valid values for parsing
VALID_OQ = {"none", "some", "most", "all"}
VALID_IQ = {"none", "some", "most", "all"}
VALID_ADJ = {"right", "wrong"}

# ==================== Model Configurations ====================
# Same as listener experiment - Ollama models

MODEL_CONFIGS = {
    "qwen2.5-7b": {
        "client_type": "openai",
        "api_key_env": "OLLAMA_API_KEY",
        "model_name": "qwen2.5:7b",
        "base_url": "http://localhost:11434/v1",
        "supports_logprobs": True,
        "family": "qwen",
        "size": "mid",
    },
    "llama3.1-8b": {
        "client_type": "openai",
        "api_key_env": "OLLAMA_API_KEY",
        "model_name": "llama3.1:8b",
        "base_url": "http://localhost:11434/v1",
        "supports_logprobs": True,
        "family": "llama",
        "size": "mid",
    },
    "mistral-7b": {
        "client_type": "openai",
        "api_key_env": "OLLAMA_API_KEY",
        "model_name": "mistral:7b",
        "base_url": "http://localhost:11434/v1",
        "supports_logprobs": True,
        "family": "mistral",
        "size": "mid",
    },
    "gemma3-12b": {
        "client_type": "openai",
        "api_key_env": "OLLAMA_API_KEY",
        "model_name": "gemma3:12b",
        "base_url": "http://localhost:11434/v1",
        "supports_logprobs": True,
        "family": "gemma",
        "size": "mid",
    },
}


@dataclass(frozen=True)
class SpeakerExperimentConfig:
    """Immutable configuration for the speaker experiment."""
    models: tuple[str, ...]
    output_formats: tuple[str, ...]
    framings: tuple[str, ...] = ("high", "low")
    runs_per_item: int = 10
    n_questions: int = 12
    n_students: int = 5
    temperature: float = 0.7
    delay: float = 0.0
    output_dir: str = "results"


# ==================== Stimuli (wideShort: 5 students x 12 questions) ====================

SPEAKER_ARRAYS = [
    [12, 12, 12, 12, 12],
    [12, 12, 12, 3, 3],
    [12, 12, 12, 9, 9],
    [12, 12, 12, 0, 0],
    [12, 12, 9, 9, 9],
    [12, 12, 9, 3, 3],
    [12, 12, 9, 0, 0],
    [12, 12, 3, 3, 3],
    [12, 12, 3, 0, 0],
    [12, 12, 0, 0, 0],
    [9, 9, 9, 9, 9],
    [9, 9, 9, 3, 3],
    [9, 9, 9, 0, 0],
    [9, 9, 3, 3, 3],
    [9, 9, 3, 0, 0],
    [9, 9, 0, 0, 0],
    [3, 3, 3, 3, 3],
    [3, 3, 3, 0, 0],
    [3, 3, 0, 0, 0],
    [0, 0, 0, 0, 0],
]

# ==================== Preset Configs ====================

SPEAKER_FULL_CONFIG = SpeakerExperimentConfig(
    models=("qwen2.5-7b", "llama3.1-8b", "mistral-7b", "gemma3-12b"),
    output_formats=("full_sentence", "three_blanks"),
    runs_per_item=10,
)

SPEAKER_VERIFY_CONFIG = SpeakerExperimentConfig(
    models=("qwen2.5-7b",),
    output_formats=("full_sentence", "three_blanks"),
    runs_per_item=1,
)
