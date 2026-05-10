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
DEFAULT_OUTPUT_DIR = SHAP_DIR / "Condition1" / "Results" / "True_Far"
DEFAULT_SHAP_SEARCH_DIRS = [
    SHAP_DIR / "Test Dataset Local Shap 25",
]

TARGET_SAMPLE_IDS = [
1, 2, 3, 8, 10, 15, 21, 24, 26, 31, 34, 37, 40, 44, 46, 50, 51, 60, 65, 68, 69, 71, 72, 73, 74, 77, 80, 81, 84, 85, 88, 90, 91, 93, 96, 98, 103, 107, 112, 115, 116, 117, 120, 122, 123, 124, 125, 130, 134, 136, 137, 140, 144, 146, 149, 151, 154, 156, 158, 164, 165, 169, 172, 173, 176, 177, 182, 185, 186, 189, 191, 192, 199, 206, 208, 215, 220, 227, 234, 235, 237, 242, 244, 246, 247, 6, 45, 67, 87, 89, 113, 194, 200, 205, 210, 228]
# 번역 프롬프트
def build_shap_only_translation_prompt(text: str, translation_module):
    feature_mapping_text = translation_module.build_feature_mapping_text()

    return f"""You are a professional Korean translator for financial risk explanations.

Translate the following English explanation into natural Korean.

GENERAL RULES:
- The first sentence MUST start with: "본 AI가 예측한 결과에 따르면, \n"
- After the first sentence, immediately start the feature explanations.
- Do NOT generate a separate overall prediction sentence before the feature explanations.
- Do NOT include the phrase "분석 결과" anywhere in the output.
- Do NOT translate "default" as "디폴트".
- Instead, translate it into natural Korean financial terms such as:
  - "신용 위험"
  - "상환 위험"
  - "대출 상환 가능성"
- Prefer "신용 위험" in most cases.
- Output ONLY Korean.
- Preserve the meaning and tone.
- Do NOT use markdown.
- Do NOT add new information.
- Keep the translated result as connected prose.
- Use the newline character "\n" to separate sentences.
- Insert a newline after each feature explanation ends.
- Do NOT force citation markers such as [1], [2], [3].
- Separate each feature explanation with the newline character "\n".
- If citation markers exist in the input, preserve them.
- If citation markers do not exist in the input, do NOT add them.
- Do NOT force exactly 4 sentences.
- Translate all sentences normally EXCEPT the final sentence.

FEATURE NAME RULE:
- Feature names are fixed terms.
- You MUST follow FEATURE_NAME_MAP exactly.
- When translating a feature, use the Korean feature name exactly as written in FEATURE_NAME_MAP.
- Do NOT create your own Korean feature names.
- Do NOT paraphrase the mapped Korean feature names.
- Do NOT translate feature terms word-by-word.
- Do NOT produce terms like "체크 계좌", "체크링 계좌", or "체킹 계좌".
- If a feature appears in the input, replace it with the exact Korean value from FEATURE_NAME_MAP.
- If a feature is not in FEATURE_NAME_MAP, translate it naturally.

LINE BREAK RULE:
- Each feature explanation MUST be a separate sentence.
- After each feature explanation sentence, insert the newline character "\n".
- **CRITICAL: Do NOT generate any intermediate summary or concluding remarks (e.g., "따라서...", "이러한 이유로...") between the feature explanations and the final sentence.**
- **CRITICAL: Do NOT summarize the findings before the final fixed sentence.**
- The output should only consist of:
  1. The opening sentence.
  2. Individual feature explanation sentences.
  3. The exact final sentence required below.
  
- The output must separate feature explanations by "\n".
- Do NOT combine multiple feature explanations into one sentence.
- Do NOT explain two or more features in the same sentence.
- The final overall sentence must appear on the next line after all feature explanations.
- Do NOT use bullet points or numbering.

FINAL SENTENCE RULE:
- Ignore the original final sentence completely.
- The explanation MUST end with exactly this sentence:

전반적으로 신용 위험이 높은 수준으로 평가되어 대출 승인 가능성이 낮습니다.


If the original final sentence contains:
- "low risk"
- "low risk of default"
- "GOOD CREDIT RISK"
Then end with this sentence exactly:
전반적으로 신용 위험이 낮은 수준으로 평가되어 대출 승인 가능성이 높습니다.

If the original final sentence contains:
- "high risk"
- "high risk of default"
- "BAD CREDIT RISK"
Then end with this sentence exactly:
전반적으로 신용 위험이 높은 수준으로 평가되어 대출 승인 가능성이 낮습니다.

- **Do NOT add any text, words, or sentences after or before this final sentence.**

FEATURE_NAME_MAP:
{feature_mapping_text}

Text:
{text}

Korean:
"""


def translate_with_shap_only_prompt(text: str, translation_module, model: str, base_url: str) -> str:
    text = translation_module.force_replace_feature_names(text)
    prompt = build_shap_only_translation_prompt(text, translation_module)
    translated = ollama_chat(model, [{"role": "user", "content": prompt}], base_url)

    result = translated.strip()

    # 기존 시작 문구 제거
    result = re.sub(r"^본 AI 모델의 예측에 따르면[,\s\n]*", "", result)
    result = re.sub(r"^본 AI가 예측한 결과에 따르면[,\s\n]*", "", result)
    result = re.sub(r"^본 AI 모델의 예측 결과[,\s\n]*", "", result)

    # 결론성/전체 예측 요약 문장 제거
    remove_patterns = [
        r"고객은.*?신용 위험이 낮은 수준으로.*?판단됩니다\.",
        r"고객은.*?신용 위험이 높은 수준으로.*?판단됩니다\.",
        r"이 신청은.*?신용 위험이 낮은 수준으로.*?평가됩니다\.",
        r"이 신청은.*?신용 위험이 높은 수준으로.*?평가됩니다\.",
        r"결과적으로.*?예측합니다\.?",
        r"전반적으로.*?대출 승인 가능성이.*?습니다\.?",
        r"Overall.*?\.?"
    ]

    for pattern in remove_patterns:
        result = re.sub(pattern, "", result, flags=re.DOTALL)

    # 줄바꿈 정리
    result = re.sub(r"\n\s*\n", "\n", result).strip()

    # 시작 문구 강제 삽입
    result = "본 AI가 예측한 결과에 따르면,\n" + result

    # 마지막 문장 강제 삽입
    result = result.rstrip() + "\n전반적으로 신용 위험이 낮은 수준으로 평가되어 대출 승인 가능성이 높습니다."

    return result

SHAP_EXPLANATION_SYSTEM_PROMPT = """You are a financial risk analyst specialized in credit scoring and SHAP-based explanations.

Your job is to generate a user-friendly English explanation for credit risk predictions using ONLY the top-3 SHAP features.

You must transform technical financial expressions into intuitive, easy-to-understand English language for general users.

You are NOT a translator.
You must not directly output encoded feature names or technical expressions.
Instead, rewrite them into natural English phrases that users can easily understand.

STRICT RULES:
1. Output ONLY valid JSON.
2. Do NOT include any text outside JSON.
3. Do NOT expose reasoning steps.
4. Generate ONLY final_explanation.
5. Use ONLY the provided top-3 SHAP feature information.
"""


SHAP_EXPLANATION_USER_TEMPLATE = """Generate only the final English explanation for this credit risk prediction.

Prediction:
- label: {prediction_label}
- probability: {prediction_probability}

Top-3 SHAP features:
{feature_evidence}

Instructions:
- Use ONLY the provided top-3 SHAP features and their SHAP directions.
- Do NOT use RAG, evidence, papers, studies, literature comparisons, or SHAP values in the final explanation.
- Do NOT keep raw encoded feature names; rewrite them into intuitive English phrases.
- If shap_direction is "increases risk" or "increase_risk", describe the feature as increasing risk.
- If shap_direction is "decreases risk" or "decrease_risk", describe the feature as decreasing risk.
- Mention all three features in the same order as provided.
- Keep the explanation natural, concise, non-redundant, and analytical.
- Do NOT use explicit numbering such as [1], [2], [3].

Required flow:
1. Start with exactly: "Based on the AI model's prediction,"
2. State whether the prediction indicates low or high credit risk.
3. Explain how the three SHAP features contributed to the prediction.
4. End by reflecting the final prediction naturally.

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
        prediction_label=prediction.get("label", "UNKNOWN"),
        prediction_probability=prediction.get("probability", "UNKNOWN"),
        prediction_threshold=prediction.get("threshold", "UNKNOWN"),
        feature_evidence=format_feature_context(tuples),
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

    return final_explanation.strip()


def translate_with_translation_py_prompt(text: str, translation_module, model: str, base_url: str) -> str:
    text = translation_module.force_replace_feature_names(text)
    prompt = translation_module.build_prompt(text) # translation.py의 LG Exaone 모델 사용
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
            final_explanation_ko = translate_with_shap_only_prompt(
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
