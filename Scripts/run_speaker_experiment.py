# run_speaker_experiment.py
"""
Speaker-side experiment runner.
Given exam results + framing goal (high/low), the LLM produces an utterance.

Usage:
    python run_speaker_experiment.py --config verify   # quick test
    python run_speaker_experiment.py --config full      # full experiment
"""

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from speaker_config import (
    SPEAKER_CORE_PROMPT,
    SPEAKER_METHODS,
    MODEL_CONFIGS,
    SPEAKER_ARRAYS,
    VALID_OQ,
    VALID_IQ,
    VALID_ADJ,
    SpeakerExperimentConfig,
    SPEAKER_FULL_CONFIG,
    SPEAKER_VERIFY_CONFIG,
)

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


# ==================== Utility Functions ====================

def format_exam_data(scores: list[int], n_questions: int) -> str:
    """Format exam results for prompt insertion."""
    parts = [
        f"Student {i}: {s}/{n_questions} correct"
        for i, s in enumerate(scores, 1)
    ]
    return ", ".join(parts)


def create_speaker_prompt(
    scores: list[int],
    n_questions: int,
    framing: str,
    method_name: str,
) -> str:
    """Create the full speaker prompt."""
    exam_data = format_exam_data(scores, n_questions)
    n_students = len(scores)

    core = SPEAKER_CORE_PROMPT.format(
        n_questions=n_questions,
        n_students=n_students,
        exam_data=exam_data,
    )

    suffix = SPEAKER_METHODS[method_name]["suffix"].format(framing=framing)
    return core + "\n" + suffix


def clean_special_chars(text: str) -> str:
    """Clean special characters."""
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


def parse_speaker_response(
    response_text: str,
    method_name: str,
) -> tuple[str, str, str]:
    """
    Parse speaker response to extract (OQ, IQ, ADJ).
    Returns ("INVALID", "INVALID", "INVALID") if parsing fails.
    """
    if not response_text:
        return ("INVALID", "INVALID", "INVALID")

    cleaned = clean_special_chars(response_text).lower().strip()
    # Remove surrounding quotes and brackets
    cleaned = cleaned.strip('"').strip("'").strip()
    cleaned = cleaned.replace("[", "").replace("]", "")

    if method_name == "three_blanks":
        # Expect: "some, most, right" or "some,most,right"
        parts = [p.strip().rstrip(".") for p in re.split(r"[,\s]+", cleaned) if p.strip()]
        # Filter to only valid quantifier/adj words
        valid_words = VALID_OQ | VALID_IQ | VALID_ADJ
        parts = [p for p in parts if p in valid_words]
        if len(parts) >= 3:
            oq, iq, adj = parts[0], parts[1], parts[2]
            if oq in VALID_OQ and iq in VALID_IQ and adj in VALID_ADJ:
                return (oq, iq, adj)

    # Full sentence format: "[OQ] of the students got [IQ] of the answers [ADJ]"
    pattern = r"(\w+)\s+of\s+the\s+students\s+got\s+(\w+)\s+of\s+the\s+answers\s+(\w+)"
    match = re.search(pattern, cleaned)
    if match:
        oq, iq, adj = match.group(1), match.group(2), match.group(3).rstrip(".")
        if oq in VALID_OQ and iq in VALID_IQ and adj in VALID_ADJ:
            return (oq, iq, adj)

    # Try answer line pattern
    answer_pattern = r"answer\s*[:：]\s*(.+)"
    match = re.search(answer_pattern, cleaned)
    if match:
        answer_text = match.group(1).strip()
        match2 = re.search(pattern, answer_text)
        if match2:
            oq, iq, adj = match2.group(1), match2.group(2), match2.group(3).rstrip(".")
            if oq in VALID_OQ and iq in VALID_IQ and adj in VALID_ADJ:
                return (oq, iq, adj)

    return ("INVALID", "INVALID", "INVALID")


# ==================== API Client ====================

def init_client(model_name: str):
    """Initialize OpenAI-compatible client."""
    config = MODEL_CONFIGS[model_name]
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise ValueError(f"Missing: {config['api_key_env']}")
    import openai
    return openai.OpenAI(
        api_key=api_key,
        base_url=config["base_url"],
    ), config


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call LLM and return response text."""
    client, config = init_client(model_name)
    response = client.chat.completions.create(
        model=config["model_name"],
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ==================== Experiment Runner ====================

def run_speaker_experiment(config: SpeakerExperimentConfig) -> str:
    """Run the speaker experiment. Returns output file path."""
    output_dir = os.path.join(PROJECT_ROOT, config.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    n_arrays = len(SPEAKER_ARRAYS)
    total_calls = (
        len(config.models)
        * len(config.output_formats)
        * len(config.framings)
        * n_arrays
        * config.runs_per_item
    )

    print(f"{'=' * 70}")
    print("LLM Speaker-Side Experiment")
    print(f"{'=' * 70}")
    print(f"Models:          {', '.join(config.models)}")
    print(f"Output formats:  {', '.join(config.output_formats)}")
    print(f"Framings:        {', '.join(config.framings)}")
    print(f"Arrays:          {n_arrays}")
    print(f"Runs per item:   {config.runs_per_item}")
    print(f"Temperature:     {config.temperature}")
    print(f"Total API calls: {total_calls}")
    print(f"{'=' * 70}")

    for model_name in config.models:
        safe_model = model_name.replace("-", "_").replace(".", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            output_dir, f"speaker_{safe_model}_{timestamp}.csv"
        )

        total_done = 0
        error_count = 0

        with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow([
                "array_idx", "students_array", "n_students", "n_questions",
                "framing", "output_format", "run",
                "oq", "iq", "adj", "sentence",
                "raw_response", "is_valid", "timestamp",
            ])

            for fmt in config.output_formats:
                method_config = SPEAKER_METHODS[fmt]
                max_tokens = method_config["max_tokens"]

                for framing in config.framings:
                    print(f"\n--- {model_name} | {fmt} | {framing} ---")

                    for arr_idx, scores in enumerate(SPEAKER_ARRAYS):
                        prompt = create_speaker_prompt(
                            scores, config.n_questions,
                            framing, fmt,
                        )

                        for run in range(1, config.runs_per_item + 1):
                            total_done += 1
                            progress = f"[{total_done}/{total_calls}]"

                            try:
                                raw = call_llm(
                                    model_name, prompt,
                                    config.temperature,
                                    max_tokens,
                                )
                                cleaned = clean_special_chars(raw)
                                oq, iq, adj = parse_speaker_response(
                                    cleaned, fmt
                                )
                                is_valid = oq != "INVALID"
                                sentence = (
                                    f"{oq} of the students got {iq} of the answers {adj}"
                                    if is_valid else "INVALID"
                                )

                                print(
                                    f"  {progress} arr={arr_idx}, "
                                    f"run={run}/{config.runs_per_item} "
                                    f"-> {oq}, {iq}, {adj}"
                                )

                                writer.writerow([
                                    arr_idx, str(scores),
                                    config.n_students, config.n_questions,
                                    framing, fmt, run,
                                    oq, iq, adj, sentence,
                                    cleaned, is_valid,
                                    datetime.now().isoformat(),
                                ])
                                f.flush()

                            except Exception as e:
                                error_count += 1
                                print(f"  {progress} ERROR: {e}")
                                writer.writerow([
                                    arr_idx, str(scores),
                                    config.n_students, config.n_questions,
                                    framing, fmt, run,
                                    "ERROR", "ERROR", "ERROR", "ERROR",
                                    str(e), False,
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
    "verify": SPEAKER_VERIFY_CONFIG,
    "full": SPEAKER_FULL_CONFIG,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Speaker-Side Experiment"
    )
    parser.add_argument(
        "--config",
        choices=list(PRESET_CONFIGS.keys()),
        default="verify",
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

    output_file = run_speaker_experiment(config)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
