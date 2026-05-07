import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_INPUT = Path("RAG/QA/Summary")
DEFAULT_INPUT_DIR = Path("RAG/QA/Summary")
DEFAULT_OUTPUT_DIR = Path("RAG/Final Summary/250 Results/Condition 12/True/Near")
DEFAULT_MODEL = "qwen3.6:35b"
EXPECTED_FEATURE_COUNT = 3


SUMMARY_SYSTEM_PROMPT = """You are a financial risk analyst specialized in credit scoring, SHAP-based explanations, and RAG-grounded reasoning.

Your job is to generate a user-friendly English explanation for credit risk predictions.

You must transform technical financial expressions into intuitive, easy-to-understand English language for general users.

You are NOT a translator.
You must not directly output encoded feature names or technical expressions.
Instead, rewrite them into natural English phrases that users can easily understand.

STRICT RULES:
1. Output ONLY valid JSON.
2. Do NOT include any text outside JSON.
3. Do NOT expose reasoning steps.
4. Generate ONLY final_explanation.
5. Do NOT generate evidence.
6. Evidence sentences already exist in the input JSONL and will be copied by code.
"""


SUMMARY_USER_TEMPLATE = """Generate only the final Korean explanation for this credit risk prediction.

Prediction:
- label: {prediction_label}
- probability: {prediction_probability}

Feature evidence:
{feature_evidence}

Instructions:

Step 1: Interpret the input for writing the final explanation.
- Identify the prediction label and probability.
- Identify the three SHAP features.
- Understand each feature's meaning, SHAP direction, and RAG answer.
- Use this interpretation only to write final_explanation.

Step 2: Convert technical feature expressions into user-friendly Korean for final_explanation.
- Do NOT keep raw encoded feature names in final_explanation.
- Do NOT include raw numeric thresholds such as "100 DM 미만" if they can be rewritten more naturally.
- Rewrite technical expressions into intuitive Korean phrases.

Feature naming rule:
- In final_explanation, you MUST use the exact Korean feature names shown in the Examples mapping.
- Do NOT create new Korean names for features.
- Do NOT paraphrase the mapped Korean feature names.
- If a feature appears in Examples, copy the mapped Korean name exactly.


Step 3: Determine each feature's impact for final_explanation.
- If shap_direction is "모델에서는 위험을 높이는 방향", describe the feature as increasing risk.
- If shap_direction is "모델에서는 위험을 낮추는 방향", describe the feature as decreasing risk.
- Use natural Korean, not literal translation.

Step 4: Generate final_explanation.

- The explanation MUST start with:
  "Based on the AI model's prediction,"

- The output MUST follow this exact structure:

Line 1:
Based on the AI model's prediction,

Line 2~4:
Each feature must be written as a separate line.

For each feature:

1. Use the mapped English-friendly feature name (NOT raw feature name).

2. Extract ONE key supporting evidence phrase from the "answer" field:
   - Use ONLY the provided answer
   - Convert into a short English phrase
   - MUST include causal phrasing such as:
     "because", "as", "due to"

3. Convert feature into "The fact that ~" structure:
   Example:
   - "Low account balance" → "The fact that the account balance is low"
   - "No checking account" → "The fact that there is no checking account"

4. Determine comparison type:

- CASE 1: opposite direction
- CASE 2: same direction

5. Generate sentence using EXACTLY this structure:

[CASE 1: opposite]
"The fact that {{feature}} {{reason}}, because {{reason}}, is known in the literature as a signal that {{literature_direction}}, but the model instead identifies it as a factor that {{model_direction}}."

[CASE 2: same]
"The fact that {{feature}} {{reason}}, because {{reason}}, is known in the literature as a signal that {{literature_direction}}, and the model also identifies it as a factor that {{model_direction}}."

6. Replace:
- {{feature}} → natural English feature phrase
- {{reason}} → extracted evidence phrase
- {{literature_direction}} → "increases credit risk" or "decreases credit risk"
- {{model_direction}} → "increases risk" or "decreases risk"

7. Each line MUST end with [1], [2], [3]

Line 5:

IF prediction_label == "GOOD CREDIT RISK":
"Overall, this applicant is likely to have a low risk of default."

IF prediction_label == "BAD CREDIT RISK":
"Overall, this applicant is likely to have a high risk of default."

- Do NOT paraphrase.

Step 5: Do NOT generate evidence.
- Evidence sentences already exist in the input JSONL.
- Do NOT select, translate, paraphrase, shorten, or rewrite evidence sentences.
- The code will copy existing evidence sentences from the JSONL file into the final output.
- Your output must contain only final_explanation.

Output format:
{{
  "final_explanation": "..."
}}
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize all feature QA JSONL files into final summary outputs."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory of sample_*_feature_qa.jsonl files to summarize. Default: {DEFAULT_INPUT_DIR}",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for final summary JSON and CSV outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )

    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV path for batch output.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama chat model for summarization. Default: {DEFAULT_MODEL}",
    )

    return parser.parse_args()

def load_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc

    return records


def validate_records(records, path: Path):
    if len(records) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"{path} must contain exactly {EXPECTED_FEATURE_COUNT} records, "
            f"but found {len(records)}."
        )

    sample_ids = {record.get("sample_idx") for record in records}
    if len(sample_ids) != 1:
        raise ValueError(f"{path} must contain records for exactly one sample: {sample_ids}")

    return sorted(records, key=lambda record: int(record.get("feature_rank", 0)))


def page_display(meta_page):
    if isinstance(meta_page, int):
        return meta_page + 1
    return meta_page


def direction_label(direction: str):
    if direction == "increase_risk":
        return "모델에서는 위험을 높이는 방향"
    if direction == "decrease_risk":
        return "모델에서는 위험을 낮추는 방향"
    return "모델 방향 미확인"


def compact_answer(answer: str, max_chars: int = 1800):
    normalized = " ".join((answer or "").split())

    if len(normalized) <= max_chars:
        return normalized

    return normalized[:max_chars].rsplit(" ", 1)[0] + "..."


def get_existing_evidence_sentences(record):
    evidence_sentences = record.get("evidence_sentences") or []

    if not evidence_sentences:
        return []

    existing_evidence = []

    for evidence in evidence_sentences:
        existing_evidence.append(
            {
                "sentence": evidence.get("sentence", ""),
                "source": evidence.get("source", "unknown"),
                "page": page_display(evidence.get("page", "unknown")),
                "score": evidence.get("score", "unknown"),
            }
        )

    return existing_evidence


def format_existing_evidence_sentences(existing_evidence):
    if not existing_evidence:
        return "- none"

    rows = []

    for idx, evidence in enumerate(existing_evidence, 1):
        rows.append(
            (
                f"- evidence {idx}: {evidence['sentence']} "
                f"(source: {evidence['source']}, page: {evidence['page']})"
            )
        )

    return "\n".join(rows)


def format_feature_evidence(records):
    blocks = []

    for ref_number, record in enumerate(records, 1):
        existing_evidence = get_existing_evidence_sentences(record)

        blocks.append(
            "\n".join(
                [
                    f"[{ref_number}]",
                    f"- feature: {record.get('feature', 'unknown')}",
                    f"- definition: {record.get('feature_definition', 'unknown')}",
                    f"- shap_direction: {direction_label(record.get('feature_direction', 'UNKNOWN'))}",
                    f"- shap_value: {record.get('feature_shap_value', 'unknown')}",
                    "- existing_evidence_sentences:",
                    format_existing_evidence_sentences(existing_evidence),
                    f"- rag_answer: {compact_answer(record.get('answer', ''))}",
                ]
            )
        )

    return "\n\n".join(blocks)


def build_evidence_from_records(records):
    evidence = []

    for ref_number, record in enumerate(records, 1):
        existing_evidence = get_existing_evidence_sentences(record)

        evidence.append(
            {
                "ref": ref_number,
                "feature": record.get("feature", "unknown"),
                "evidence_sentences": existing_evidence,
            }
        )

    return evidence


def clean_llm_text(text: str):
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


def validate_llm_output(payload):
    if not isinstance(payload, dict):
        raise ValueError("LLM output must be a JSON object.")

    if "final_explanation" not in payload:
        raise ValueError("LLM output is missing 'final_explanation'.")

    final_explanation = payload["final_explanation"]

    if not isinstance(final_explanation, str) or not final_explanation.strip():
        raise ValueError("final_explanation must be a non-empty string.")

    missing_refs = [ref for ref in ("[1]", "[2]", "[3]") if ref not in final_explanation]

    if missing_refs:
        raise ValueError(
            f"final_explanation is missing citation marker(s): {', '.join(missing_refs)}. "
            f"final_explanation: {final_explanation}"
        )

    return payload


def build_llm(model: str):
    try:
        from langchain_ollama import ChatOllama
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'langchain_ollama'. Install it with: pip install langchain-ollama"
        ) from exc

    return ChatOllama(model=model, temperature=0.0)


def summarize_records(records, llm):
    try:
        from langchain_core.prompts import ChatPromptTemplate
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'langchain_core'. Install it with: pip install langchain-core"
        ) from exc

    first = records[0]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUMMARY_SYSTEM_PROMPT),
            ("user", SUMMARY_USER_TEMPLATE),
        ]
    )

    chain = prompt | llm

    result = chain.invoke(
        {
            "prediction_label": first.get("prediction_label", "UNKNOWN"),
            "prediction_probability": first.get("prediction_probability", "UNKNOWN"),
            "feature_evidence": format_feature_evidence(records),
        }
    )

    llm_payload = parse_llm_json(result.content)
    validate_llm_output(llm_payload)

    return llm_payload


def build_payload(records, llm_payload):
    first = records[0]

    return {
        "sample_idx": first.get("sample_idx"),
        "prediction_label": first.get("prediction_label", "UNKNOWN"),
        "prediction_probability": first.get("prediction_probability", "UNKNOWN"),
        "final_explanation": llm_payload["final_explanation"],
        "evidence": build_evidence_from_records(records),
    }


def output_path_for_input(input_path: Path, output_dir: Path):
    match = re.search(r"sample_(\d+)_feature_qa\.jsonl$", input_path.name)

    if match:
        return output_dir / f"sample_{match.group(1)}_final_summary.json"

    return output_dir / f"{input_path.stem}_final_summary.json"


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_csv(path: Path, payloads):
    fieldnames = [
        "sample_idx",
        "prediction_label",
        "prediction_probability",
        "final_explanation",
        "evidence",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for payload in payloads:
            row = payload.copy()
            row["evidence"] = json.dumps(row.get("evidence", []), ensure_ascii=False)
            writer.writerow(row)


def summarize_file(input_path: Path, output_path: Path, llm):
    records = validate_records(load_jsonl(input_path), input_path)
    llm_payload = summarize_records(records, llm)
    payload = build_payload(records, llm_payload)
    write_json(output_path, payload)
    return payload


def iter_input_files(input_dir: Path, target_indices=None):
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted(input_dir.glob("sample_*_feature_qa.jsonl"))

    if target_indices:
        files = [
            f for f in files
            if any(f"sample_{idx}_" in f.name for idx in target_indices)
        ]

    if not files:
        raise FileNotFoundError("No matching JSONL files found.")

    return files

def main():
    args = parse_args()
    llm = build_llm(args.model)

    input_files = iter_input_files(args.input_dir, target_indices=[5, 7, 17, 20, 49, 63, 70, 76, 79, 94, 100, 111, 132, 159, 187, 197, 202, 211, 221, 223, 238, 14, 39, 42, 43, 47, 53, 109, 118, 147, 236])
    payloads = []

    for input_path in input_files:
        output_path = output_path_for_input(input_path, args.output_dir)

        try:
            payload = summarize_file(input_path, output_path, llm)
            payloads.append(payload)
            print(f"Saved JSON: {output_path}")

        except Exception as exc:
            print(f"SKIPPED: {input_path} / ERROR: {exc}", file=sys.stderr)

    # csv_output = args.csv_output or args.output_dir / "final_summaries.csv"
    # write_csv(csv_output, payloads)

    # print(f"Saved CSV : {csv_output}")
    # print(f"Total summaries: {len(payloads)}")

if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ModuleNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
