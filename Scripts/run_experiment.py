# run_experiment.py
"""
Config-driven experiment runner for the LLM Listener Grid Search.
Reads stimuli, runs all conditions from an ExperimentConfig, saves results.

Usage:
    python run_experiment.py --config phase1_verify   # small pilot
    python run_experiment.py --config phase1           # full Phase 1
"""

import argparse
import ast
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from experiment_config import (
    CORE_PROMPT,
    PROMPT_TEMPLATES,
    MODEL_CONFIGS,
    ROLE_TO_CONDITION,
    ExperimentConfig,
    PHASE1_PILOT_CONFIG,
    PHASE1_VERIFY_CONFIG,
)

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STIMULI_FILE = os.path.join(PROJECT_ROOT, "listener_stimuli.csv")


# ==================== Utility Functions ====================

def parse_observation(obs_str: str) -> list[int]:
    """Parse observation field from CSV string to list of ints."""
    if isinstance(obs_str, str):
        try:
            return ast.literal_eval(obs_str)
        except (ValueError, SyntaxError):
            return json.loads(obs_str)
    return obs_str


def format_exam_data(correct_counts: list[int], total_questions: int = 12) -> str:
    """Format exam results for prompt insertion."""
    parts = [
        f"Student {i}: {correct}/{total_questions} correct"
        for i, correct in enumerate(correct_counts, 1)
    ]
    return ", ".join(parts)


def build_sentence(q1: str, q2: str, adj: str) -> str:
    """Build the target sentence from quantifiers and adjective."""
    return f"{q1} of the students got {q2} of the answers {adj}"


def create_prompt(
    correct_counts: list[int],
    q1: str,
    q2: str,
    adj: str,
    total_questions: int,
    template_name: str,
) -> str:
    """Create a prompt from the core template + method-specific suffix."""
    exam_data = format_exam_data(correct_counts, total_questions)
    sentence = build_sentence(q1, q2, adj)
    n_students = len(correct_counts)

    core = CORE_PROMPT.format(
        total_questions=total_questions,
        n_students=n_students,
        exam_data=exam_data,
        sentence=sentence,
    )

    suffix = PROMPT_TEMPLATES[template_name]["suffix"]
    return core + "\n" + suffix


def clean_special_chars(text: str) -> str:
    """Clean special characters that may cause encoding issues."""
    if not text:
        return text
    replacements = {
        "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2026": "...", "\u200b": "",
        "\ufeff": "", "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


# ==================== Response Parsing ====================

# Valid labels in both formats
VALID_ROLES = {"student", "teacher", "examiner"}
VALID_META = {"high", "low", "info"}


def normalize_to_condition(label: str, response_type: str) -> str:
    """Normalize a parsed label to condition format (high/low/info)."""
    label_lower = label.lower().strip()

    if response_type == "role_name":
        if label_lower in ROLE_TO_CONDITION:
            return ROLE_TO_CONDITION[label_lower]
    elif response_type == "meta_label":
        if label_lower in VALID_META:
            return label_lower

    # If already a valid condition label, return it
    if label_lower in VALID_META:
        return label_lower
    if label_lower in ROLE_TO_CONDITION:
        return ROLE_TO_CONDITION[label_lower]

    return label  # return as-is if unrecognized


def parse_response(response_text: str, response_type: str) -> str:
    """Parse model response to extract label. Handles both role names and meta-labels."""
    if not response_text:
        return "ERROR: Empty response"

    cleaned = clean_special_chars(response_text)
    lower = cleaned.lower().strip()

    if response_type == "role_name":
        valid_labels = VALID_ROLES
    else:
        valid_labels = VALID_META

    # Pattern-based extraction (ordered by specificity)
    label_patterns = [
        r"answer\s*[:：]\s*(\w+)",
        r"(?:final\s+)?label\s*[:：]\s*(\w+)",
        r"^\s*(\w+)\s*$",
    ]
    for pattern in label_patterns:
        match = re.search(pattern, lower, re.MULTILINE)
        if match:
            candidate = match.group(1).strip()
            if candidate in valid_labels:
                return normalize_to_condition(candidate, response_type)

    # Check last few lines for bare label
    lines = lower.split("\n")
    for line in reversed(lines[-5:]):
        word = line.strip().rstrip(".")
        if word in valid_labels:
            return normalize_to_condition(word, response_type)

    # Last word scan
    words = re.findall(r"\b(\w+)\b", lower)
    for word in reversed(words):
        if word in valid_labels:
            return normalize_to_condition(word, response_type)

    # Unique mention check
    found = [w for w in valid_labels if w in lower]
    if len(found) == 1:
        return normalize_to_condition(found[0], response_type)

    # Fallback: also check the other label set
    all_labels = VALID_ROLES | VALID_META
    for word in reversed(words):
        if word in all_labels:
            if word in ROLE_TO_CONDITION:
                return ROLE_TO_CONDITION[word]
            return word

    preview = cleaned[:100] + "..." if len(cleaned) > 100 else cleaned
    return f"Invalid: {preview}"


def parse_confidence(response_text: str) -> Optional[float]:
    """Extract confidence score from response text."""
    if not response_text:
        return None
    lower = response_text.lower()
    patterns = [
        r"confidence\s*[:：]\s*([0-9.]+)",
        r"score\s*[:：]\s*([0-9.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return None


# ==================== API Client ====================

def init_client(model_name: str):
    """Initialize OpenAI-compatible client for a model."""
    config = MODEL_CONFIGS[model_name]
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise ValueError(
            f"Missing environment variable: {config['api_key_env']}"
        )
    import openai
    client = openai.OpenAI(
        api_key=api_key,
        base_url=config["base_url"],
    )
    return client, config


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(
    model_name: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 100,
    logprobs: bool = False,
    top_logprobs: int = 5,
) -> tuple[str, Optional[dict]]:
    """Call LLM and return (response_text, logprobs_data)."""
    client, config = init_client(model_name)

    request_params = {
        "model": config["model_name"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if logprobs and config.get("supports_logprobs", False):
        request_params["logprobs"] = True
        request_params["top_logprobs"] = top_logprobs

    response = client.chat.completions.create(**request_params)
    response_text = response.choices[0].message.content.strip()

    # Extract logprobs if available
    logprobs_data = None
    if logprobs and config.get("supports_logprobs", False):
        choice = response.choices[0]
        if hasattr(choice, "logprobs") and choice.logprobs:
            lp = choice.logprobs
            if hasattr(lp, "content") and lp.content:
                logprobs_data = [
                    {
                        "token": tok.token,
                        "logprob": tok.logprob,
                        "top_logprobs": [
                            {"token": t.token, "logprob": t.logprob}
                            for t in (tok.top_logprobs or [])
                        ],
                    }
                    for tok in lp.content
                ]

    return response_text, logprobs_data


# ==================== Experiment Runner ====================

def run_experiment(config: ExperimentConfig) -> str:
    """
    Run the full grid search experiment.
    Returns path to the output CSV file.
    """
    output_dir = os.path.join(PROJECT_ROOT, config.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(STIMULI_FILE)
    if config.max_items is not None:
        df = df.head(config.max_items)

    total_conditions = (
        len(config.models)
        * len(config.probing_methods)
        * len(config.temperatures)
    )
    total_calls = total_conditions * len(df) * config.runs_per_item

    print(f"{'=' * 70}")
    print("LLM Listener Grid Search Experiment")
    print(f"{'=' * 70}")
    print(f"Models:          {', '.join(config.models)}")
    print(f"Probing methods: {', '.join(config.probing_methods)}")
    print(f"Temperatures:    {config.temperatures}")
    print(f"Items:           {len(df)}")
    print(f"Runs per item:   {config.runs_per_item}")
    print(f"Total conditions:{total_conditions}")
    print(f"Total API calls: {total_calls}")
    print(f"{'=' * 70}")

    for model_name in config.models:
        safe_model = model_name.replace("-", "_").replace(".", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            output_dir, f"grid_{safe_model}_{timestamp}.csv"
        )

        total_done = 0
        error_count = 0

        with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow([
                "itemID", "list", "condition", "Q1", "Q2", "A",
                "observation", "sentence", "prompt_template",
                "temperature", "run", "chosen_speaker",
                "confidence", "logprobs_data", "raw_response",
                "timestamp",
            ])

            for method in config.probing_methods:
                method_config = PROMPT_TEMPLATES[method]
                response_type = method_config["response_type"]
                method_max_tokens = method_config.get("max_tokens", 100)
                needs_logprobs = method_config.get("requires_logprobs", False)

                for temp in config.temperatures:
                    print(
                        f"\n--- {model_name} | {method} | temp={temp} ---"
                    )

                    for idx, row in df.iterrows():
                        item_id = row["itemID"]
                        correct_counts = parse_observation(row["observation"])
                        prompt = create_prompt(
                            correct_counts,
                            row["Q1"], row["Q2"], row["A"],
                            config.total_questions,
                            method,
                        )
                        sentence = build_sentence(
                            row["Q1"], row["Q2"], row["A"]
                        )

                        for run in range(1, config.runs_per_item + 1):
                            total_done += 1
                            progress = f"[{total_done}/{total_calls}]"

                            try:
                                raw, lp_data = call_llm(
                                    model_name, prompt,
                                    temperature=temp,
                                    max_tokens=method_max_tokens,
                                    logprobs=needs_logprobs,
                                )
                                cleaned = clean_special_chars(raw)
                                label = parse_response(
                                    cleaned, response_type
                                )
                                confidence = parse_confidence(cleaned)

                                logprobs_str = ""
                                if lp_data:
                                    logprobs_str = json.dumps(
                                        lp_data, ensure_ascii=False
                                    )

                                print(
                                    f"  {progress} Item {item_id}, "
                                    f"Run {run}/{config.runs_per_item} "
                                    f"-> {label}"
                                )

                                writer.writerow([
                                    item_id, row["list"], row["condition"],
                                    row["Q1"], row["Q2"], row["A"],
                                    row["observation"], sentence,
                                    method, temp, run, label,
                                    confidence if confidence is not None else "",
                                    logprobs_str, cleaned,
                                    datetime.now().isoformat(),
                                ])
                                f.flush()

                            except Exception as e:
                                error_count += 1
                                print(f"  {progress} ERROR: {e}")
                                writer.writerow([
                                    item_id, row["list"], row["condition"],
                                    row["Q1"], row["Q2"], row["A"],
                                    row["observation"], sentence,
                                    method, temp, run, "ERROR", "",
                                    "", f"Error: {e}",
                                    datetime.now().isoformat(),
                                ])
                                f.flush()

                            if config.delay > 0:
                                time.sleep(config.delay)

        print(f"\n{'=' * 70}")
        print(f"Model {model_name} complete")
        print(f"  Total calls: {total_done}")
        print(f"  Errors:      {error_count}")
        success_pct = (total_done - error_count) / max(total_done, 1) * 100
        print(f"  Success:     {success_pct:.1f}%")
        print(f"  Output:      {output_file}")
        print(f"{'=' * 70}")

    return output_file


# ==================== CLI ====================

PRESET_CONFIGS = {
    "phase1_verify": PHASE1_VERIFY_CONFIG,
    "phase1": PHASE1_PILOT_CONFIG,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Listener Grid Search Experiment"
    )
    parser.add_argument(
        "--config",
        choices=list(PRESET_CONFIGS.keys()),
        default="phase1_verify",
        help="Preset configuration to use",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompt",
    )
    args = parser.parse_args()

    config = PRESET_CONFIGS[args.config]

    if not args.no_confirm:
        response = input("\nPress Enter to start, or 'q' to quit: ")
        if response.lower() == "q":
            print("Cancelled.")
            sys.exit(0)

    output_file = run_experiment(config)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
