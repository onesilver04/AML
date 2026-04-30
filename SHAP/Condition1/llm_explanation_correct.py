import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import types
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_EXPLANATION_MODEL = "qwen3.6:35b"
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

SHAP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SHAP_DIR.parent
TRANSLATION_PATH = REPO_ROOT / "RAG" / "Final Summary" / "translation.py"
DEFAULT_OUTPUT_DIR = SHAP_DIR / "Condition1" / "Results" / "selected_shap_only_explanations_ko"
DEFAULT_SHAP_SEARCH_DIRS = [
    SHAP_DIR / "Task" / "correct_102_local_shap",
    SHAP_DIR / "Task" / "wrong_18_local_shap",
]

TARGET_SAMPLE_IDS = [
    123, 142, 176, 45, 81, 83, 197, 187, 100, 10, 177, 126, 136, 146, 169, 189, 194,
    144, 14, 24, 47, 182, 172, 93, 120, 127, 135, 178, 179, 48, 63, 16, 124, 78
]


SHAP_EXPLANATION_SYSTEM_PROMPT = """You are a financial risk analyst specialized in credit scoring and SHAP-based explanations.

Generate a user-friendly English explanation for a credit risk prediction using only the provided top-3 SHAP values.

Rules:
1. Output ONLY valid JSON.
2. The JSON object must contain ONLY final_explanation.
3. Do NOT use RAG, retrieved evidence, literature evidence, or external knowledge.
4. Do NOT mention missing evidence, RAG, sources, or literature.
5. Include the three SHAP features in the same order as provided.
6. Use [1], [2], and [3] only as feature-rank markers.
"""


SHAP_EXPLANATION_USER_TEMPLATE = """Generate final_explanation for this credit risk prediction using only the top-3 SHAP values.

Prediction:
- true label: {true_label}
- predicted label: {prediction_label}
- predicted bad-risk probability: {prediction_probability}
- decision threshold: {prediction_threshold}

Top-3 SHAP values:
{feature_context}

Instructions:
- Start with exactly: Based on the AI model's prediction,
- Write one sentence for each SHAP feature.
- Convert raw feature names into natural English phrases.
- If shap_direction is "model increases risk", say the feature increases the model's risk estimate.
- If shap_direction is "model decreases risk", say the feature decreases the model's risk estimate.
- Do not claim causality beyond SHAP contribution.
- Sentence 1 must end with [1]
- Sentence 2 must end with [2]
- Sentence 3 must end with [3]
- End with exactly one of these final sentences:
  - Overall, this applicant is likely to have a low risk of default.
  - Overall, this applicant is likely to have a high risk of default.

Output format:
{{
  "final_explanation": "..."
}}
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find top-3 SHAP JSON files for the specified samples, generate SHAP-only "
            "English explanations, and translate them with translation.py's Korean prompt."
        )
    )
    parser.add_argument(
        "--sample-ids",
        default=" ".join(str(sample_id) for sample_id in TARGET_SAMPLE_IDS),
        help="Whitespace- or comma-separated sample_idx values. Defaults to the requested sample list.",
    )
    parser.add_argument(
        "--shap-search-dirs",
        nargs="+",
        type=Path,
        default=DEFAULT_SHAP_SEARCH_DIRS,
        help="Directories to search for shap_tuples_non_prefix_<sample_idx>.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EXPLANATION_MODEL,
        help=f"Ollama model for SHAP-only English explanation. Default: {DEFAULT_EXPLANATION_MODEL}",
    )
    parser.add_argument(
        "--translation-model",
        default=None,
        help="Ollama model for Korean translation. Default: translation.py MODEL_NAME.",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama base URL. Default: {DEFAULT_OLLAMA_URL}",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV summary output path.",
    )
    return parser.parse_args()


def parse_sample_ids(text: str) -> list[int]:
    sample_ids = [int(token) for token in re.split(r"[\s,]+", text.strip()) if token]

    if not sample_ids:
        raise ValueError("At least one sample id is required.")

    return sample_ids


def unique_preserving_order(values: list[int]) -> list[int]:
    seen = set()
    unique = []

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)

    return unique


def load_translation_module():
    if not TRANSLATION_PATH.exists():
        raise FileNotFoundError(f"translation.py does not exist: {TRANSLATION_PATH}")

    try:
        import langchain_ollama  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("langchain_ollama")

        class ChatOllama:  # pragma: no cover - only used to load translation.py without dependency.
            def __init__(self, *args, **kwargs):
                raise ModuleNotFoundError(
                    "langchain_ollama is not installed; this script uses Ollama REST instead."
                )

        stub.ChatOllama = ChatOllama
        sys.modules["langchain_ollama"] = stub

    spec = importlib.util.spec_from_file_location("translation_prompt_module", TRANSLATION_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import translation.py: {TRANSLATION_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_ollama_url(base_url: str) -> str:
    return base_url.rstrip("/")


def ollama_chat(model: str, messages: list[dict], base_url: str) -> str:
    url = f"{normalize_ollama_url(base_url)}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Could not reach Ollama at {url}: {exc}") from exc

    try:
        return data["message"]["content"].strip()
    except KeyError as exc:
        raise ValueError(f"Unexpected Ollama response: {data}") from exc


def clean_llm_text(text: str) -> str:
    cleaned = (text or "").strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    return cleaned.strip()


def parse_llm_json(text: str):
    cleaned = clean_llm_text(text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)

        if not match:
            raise ValueError(f"LLM output is not valid JSON: {cleaned}")

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM output JSON parsing failed: {cleaned}") from exc


def find_shap_json(sample_idx: int, search_dirs: list[Path]) -> Path:
    filename = f"shap_tuples_non_prefix_{sample_idx}.json"
    matches = []

    for search_dir in search_dirs:
        candidate = search_dir / filename
        if candidate.exists():
            matches.append(candidate)

    if not matches:
        searched = ", ".join(str(path) for path in search_dirs)
        raise FileNotFoundError(f"No SHAP JSON found for sample_idx={sample_idx}. Searched: {searched}")

    if len(matches) > 1:
        print(f"Multiple SHAP JSON files for sample_idx={sample_idx}; using {matches[0]}", file=sys.stderr)

    return matches[0]


def load_shap_payload(sample_idx: int, search_dirs: list[Path]):
    path = find_shap_json(sample_idx, search_dirs)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f), path


def validate_shap_payload(payload):
    tuples = payload.get("tuples") or []

    if len(tuples) != 3:
        raise ValueError(
            f"sample_idx={payload.get('sample_idx')} must contain exactly 3 SHAP tuples, "
            f"but found {len(tuples)}."
        )

    return tuples


def direction_label(direction: str) -> str:
    if direction == "increase_risk":
        return "model increases risk"
    if direction == "decrease_risk":
        return "model decreases risk"
    return "model direction unknown"


def format_feature_context(tuples) -> str:
    blocks = []

    for rank, item in enumerate(tuples, 1):
        blocks.append(
            "\n".join(
                [
                    f"[{rank}]",
                    f"- feature: {item.get('feature', 'unknown')}",
                    f"- definition: {item.get('definition', 'unknown')}",
                    f"- shap_direction: {direction_label(item.get('direction', 'UNKNOWN'))}",
                    f"- shap_value: {item.get('shap_value', 'unknown')}",
                    f"- absolute_shap_value: {item.get('abs_shap', 'unknown')}",
                ]
            )
        )

    return "\n\n".join(blocks)


def generate_shap_explanation(shap_payload, model: str, base_url: str):
    tuples = validate_shap_payload(shap_payload)
    prediction = shap_payload.get("prediction") or {}
    prompt = SHAP_EXPLANATION_USER_TEMPLATE.format(
        true_label=shap_payload.get("true_label", "UNKNOWN"),
        prediction_label=prediction.get("label", "UNKNOWN"),
        prediction_probability=prediction.get("probability", "UNKNOWN"),
        prediction_threshold=prediction.get("threshold", "UNKNOWN"),
        feature_context=format_feature_context(tuples),
    )
    content = ollama_chat(
        model,
        [
            {"role": "system", "content": SHAP_EXPLANATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        base_url,
    )
    payload = parse_llm_json(content)
    final_explanation = payload.get("final_explanation")

    if not isinstance(final_explanation, str) or not final_explanation.strip():
        raise ValueError(f"LLM output is missing final_explanation: {payload}")

    missing_refs = [ref for ref in ("[1]", "[2]", "[3]") if ref not in final_explanation]
    if missing_refs:
        raise ValueError(f"final_explanation is missing markers {missing_refs}: {final_explanation}")

    return final_explanation.strip()


def translate_with_translation_py_prompt(text: str, translation_module, model: str, base_url: str) -> str:
    text = translation_module.force_replace_feature_names(text)
    prompt = translation_module.build_prompt(text)
    translated = ollama_chat(model, [{"role": "user", "content": prompt}], base_url)
    return translation_module.force_replace_feature_names(translated.strip())


def build_shap_features(tuples):
    features = []

    for rank, item in enumerate(tuples, 1):
        features.append(
            {
                "rank": rank,
                "feature": item.get("feature", "unknown"),
                "definition": item.get("definition", "unknown"),
                "shap_value": item.get("shap_value", "unknown"),
                "abs_shap": item.get("abs_shap", "unknown"),
                "direction": item.get("direction", "UNKNOWN"),
            }
        )

    return features


def build_output_payload(shap_path: Path, shap_payload, final_explanation: str, final_explanation_ko: str):
    prediction = shap_payload.get("prediction") or {}
    tuples = validate_shap_payload(shap_payload)

    return {
        "sample_idx": shap_payload.get("sample_idx"),
        "source_shap_json": str(shap_path),
        "true_label": shap_payload.get("true_label", "UNKNOWN"),
        "prediction_label": prediction.get("label", "UNKNOWN"),
        "prediction_probability": prediction.get("probability", "UNKNOWN"),
        "prediction_threshold": prediction.get("threshold", "UNKNOWN"),
        "final_explanation": final_explanation,
        "final_explanation_ko": final_explanation_ko,
        "shap_features": build_shap_features(tuples),
    }


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_csv(path: Path, payloads):
    fieldnames = [
        "sample_idx",
        "source_shap_json",
        "true_label",
        "prediction_label",
        "prediction_probability",
        "prediction_threshold",
        "final_explanation",
        "final_explanation_ko",
        "shap_features",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for payload in payloads:
            row = payload.copy()
            row["shap_features"] = json.dumps(row.get("shap_features", []), ensure_ascii=False)
            writer.writerow(row)


def main():
    args = parse_args()
    requested_sample_ids = parse_sample_ids(args.sample_ids)
    sample_ids = unique_preserving_order(requested_sample_ids)
    duplicate_count = len(requested_sample_ids) - len(sample_ids)
    translation_module = load_translation_module()
    translation_model = args.translation_model or translation_module.MODEL_NAME
    payloads = []

    if duplicate_count:
        print(f"Duplicate sample ids removed: {duplicate_count}")

    for sample_idx in sample_ids:
        try:
            shap_payload, shap_path = load_shap_payload(sample_idx, args.shap_search_dirs)
            final_explanation = generate_shap_explanation(shap_payload, args.model, args.ollama_url)
            final_explanation_ko = translate_with_translation_py_prompt(
                final_explanation,
                translation_module,
                translation_model,
                args.ollama_url,
            )
            output_payload = build_output_payload(
                shap_path,
                shap_payload,
                final_explanation,
                final_explanation_ko,
            )
            output_path = args.output_dir / f"sample_{sample_idx}_shap_only_summary_ko.json"
            write_json(output_path, output_payload)
            payloads.append(output_payload)
            print(f"Saved JSON: {output_path}")

        except Exception as exc:
            print(f"SKIPPED: sample_idx={sample_idx} / ERROR: {exc}", file=sys.stderr)

    summary_path = args.output_dir / "selected_shap_only_summaries_ko.json"
    csv_output = args.csv_output or args.output_dir / "selected_shap_only_summaries_ko.csv"
    write_json(summary_path, payloads)
    write_csv(csv_output, payloads)

    print(f"Saved summary JSON: {summary_path}")
    print(f"Saved CSV         : {csv_output}")
    print(f"Total summaries   : {len(payloads)}")


if __name__ == "__main__":
    try:
        main()
    except (ConnectionError, FileNotFoundError, ImportError, ModuleNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
