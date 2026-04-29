import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_INPUT = Path("RAG/QA/Answers/correct_102_answers/sample_62_feature_qa.jsonl")
DEFAULT_INPUT_DIR = Path("RAG/QA/Answers")
DEFAULT_OUTPUT_DIR = Path("RAG/Final Summary/Results")
DEFAULT_MODEL = "qwen3.6:35b"
EXPECTED_FEATURE_COUNT = 3


SUMMARY_SYSTEM_PROMPT = """You are a financial risk analyst specialized in credit scoring, SHAP-based explanations, and RAG-grounded reasoning.

Your job is to generate a user-friendly Korean explanation for credit risk predictions.

You must transform technical financial expressions into intuitive, easy-to-understand Korean language for general users.

You are NOT a translator.
You must not directly translate encoded feature names, numeric thresholds, or technical financial expressions.
Instead, rewrite them into natural Korean phrases that users can understand at a glance.

STRICT RULES:
1. Output ONLY valid JSON.
2. Do NOT include any text outside JSON.
3. Do NOT expose reasoning steps.
4. Generate ONLY final_explanation.
5. Do NOT generate evidence.
6. Evidence sentences already exist in the input JSONL and will be copied by code.
7. The final explanation must be 1 to 3 Korean sentences.
8. The final explanation must include all three features in the same order as provided.
9. Each feature clause must include citation markers [1], [2], and [3].
10. If SHAP direction and RAG evidence conflict, explicitly mention the inconsistency in Korean.
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

Examples:
- "duration" -> "대출 기간"
- "credit_amount" -> "대출 금액"
- "installment_commitment" -> "월 상환 부담 수준"
- "residence_since" -> "현재 거주 기간"
- "age" -> "나이"
- "existing_credits" -> "기존 대출 수"
- "num_dependents" -> "부양 가족 수"

- "checking_status_<0" -> "계좌 잔액 부족"
- "checking_status_0<=X<200" -> "계좌 잔액 적음"
- "checking_status_>=200" -> "계좌 잔액 충분"
- "checking_status_no checking" -> "입출금 계좌 없음"

- "credit_history_all paid" -> "대출 전액 상환 이력"
- "credit_history_critical/other existing credit" -> "신용 문제 이력"
- "credit_history_delayed previously" -> "연체 이력"
- "credit_history_existing paid" -> "기존 대출 상환 중"
- "credit_history_no credits/all paid" -> "대출 이력 없음 또는 전액 상환"

- "purpose_business" -> "사업 자금 목적"
- "purpose_domestic appliance" -> "가전제품 구매 목적"
- "purpose_education" -> "교육비 목적"
- "purpose_furniture/equipment" -> "가구 또는 장비 구매 목적"
- "purpose_new car" -> "신차 구매 목적"
- "purpose_used car" -> "중고차 구매 목적"
- "purpose_radio/tv" -> "전자제품 구매 목적"
- "purpose_repairs" -> "수리비 목적"
- "purpose_retraining" -> "직업 재교육 목적"
- "purpose_other" -> "기타 목적"

- "savings_status_<100" or "저축 잔액 100 DM 미만" -> "저축 잔액이 거의 없음"
- "savings_status_100<=X<500" -> "저축 잔액 적음"
- "savings_status_500<=X<1000" -> "저축 잔액 보통"
- "savings_status_>=1000" -> "저축 잔액 충분"
- "savings_status_no known savings" -> "저축 내역 없음"

- "employment_<1" -> "재직 기간 1년 미만"
- "employment_1<=X<4" -> "재직 기간 1~4년"
- "employment_4<=X<7" -> "재직 기간 4~7년"
- "employment_>=7" -> "재직 기간 7년 이상"
- "employment_unemployed" -> "무직"

- "personal_status_female div/dep/mar" -> "여성 (이혼/별거/기혼 상태)"
- "personal_status_male div/sep" -> "남성 (이혼 또는 별거)"
- "personal_status_male mar/wid" -> "남성 (기혼 또는 사별)"
- "personal_status_male single" -> "남성 (미혼)"

- "other_parties_co applicant" -> ""공동 신청자 있음""
- "other_parties_guarantor" -> "보증인 있음"
- "other_parties_none" -> "공동 신청자, 보증인 없음"

- "property_magnitude_real estate" -> "부동산 보유"
- "property_magnitude_life insurance" -> "보험 자산 보유"
- "property_magnitude_car" -> "차량 보유"
- "property_magnitude_no known property" -> "보유 자산 없음"

- "other_payment_plans_bank" -> "타 은행 상환 중"
- "other_payment_plans_stores" -> "할부·외상 상환 중"
- "other_payment_plans_none" -> "추가 상환 없음"

- "housing_own" -> "자가 거주"
- "housing_rent" -> "임차 거주"
- "housing_for free" -> "무상 거주"

- "job_high qualif/self emp/mgmt" -> "전문직, 자영업, 관리직"
- "job_skilled" -> "숙련 기술직"
- "job_unskilled resident" -> "단순 노무직"
- "job_unemp/unskilled non res" -> "무직, 단순직 외국인"

- "own_telephone_yes" -> "전화 보유"
- "own_telephone_none" -> "전화 미보유"

- "foreign_worker_yes" -> "외국인 근로자"
- "foreign_worker_no" -> "내국인 근로자"
- "class" -> "신용 등급"
- "Sex" -> "성별"
- "Married" -> "결혼 여부"

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

- The explanation MUST start with the exact phrase:
  "AI 모델이 예측한 결과,"

- The output MUST follow this exact structure:

Line 1:
AI 모델이 예측한 결과,

Line 2~4:
Each feature must be written as a separate line.

For each feature, you MUST:

1. Use the exact Korean feature name from the Examples mapping.

2. Extract ONE key supporting evidence phrase from the "answer" field:
   - Use ONLY the provided answer
   - Do NOT generate new content
   - Convert it into a short Korean phrase
   - 반드시 "~이기 때문에", "~때문에", "~으로 인해" 형태로 변환

3. Convert feature into "~한 점은" structure:
   - Example:
     - "계좌 잔액 부족" → "계좌 잔액이 부족한 점은"
     - "입출금 계좌 없음" → "입출금 계좌가 없는 점은"
     - "신용 문제 이력" → "신용 문제 이력이 있는 점은"

4. Determine comparison type:
   - CASE 1: opposite
   - CASE 2: same
   - CASE 3: insufficient evidence

5. Generate sentence using EXACTLY this unified structure:

[CASE 1: 반대 방향]
"{{feature 변환형}} {{이유}} 때문에, 문헌에서는 {{문헌 방향}} 신호로 알려져 있지만 이 모델에서는 {{모델 방향}} 요인으로 분석되었습니다."

[CASE 2: 같은 방향]
"{{feature 변환형}} {{이유}} 때문에, 문헌에서 {{문헌 방향}} 신호로 알려져 있으며 이 모델에서도 {{모델 방향}} 요인으로 분석되었습니다."

[CASE 3: 증거 부족]
"{{feature 변환형}} {{이유}} 때문에 모델에서는 {{모델 방향}} 요인으로 분류되었지만, 문헌에서는 충분히 다루어지지 않아 비교가 어렵습니다."

6. Replace:
   - {{feature 변환형}} → "~한 점은" 형태로 변환된 feature
   - {{이유}} → answer 기반 근거 phrase
   - {{문헌 방향}} → "신용 위험을 높이는" 또는 "신용 위험을 낮추는"
   - {{모델 방향}} → "위험을 높이는" 또는 "위험을 낮추는"

7. Rules:
   - Include "때문에"
   - Use "~한 점은" structure
   - Do Not Change Sentence Structure
   - No additional explanation

8. Each line MUST end with [1], [2], [3]

Line 5:
- Write one final summary sentence describing overall prediction.

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
        description="Summarize per-feature RAG answers into a Korean UI explanation with existing JSONL evidence."
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Single feature QA JSONL file to summarize. Default: {DEFAULT_INPUT}",
    )
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help=f"Directory of sample_*_feature_qa.jsonl files to summarize. Default batch dir: {DEFAULT_INPUT_DIR}",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for final summary JSON and CSV outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path for single-file mode.",
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


def iter_input_files(input_dir: Path):
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted(input_dir.glob("sample_*_feature_qa.jsonl"))

    if not files:
        raise FileNotFoundError(f"No sample_*_feature_qa.jsonl files found in: {input_dir}")

    return files


def main():
    args = parse_args()
    llm = build_llm(args.model)

    if args.input_dir:
        input_files = iter_input_files(args.input_dir)
        payloads = []

        for input_path in input_files:
            output_path = output_path_for_input(input_path, args.output_dir)
            payload = summarize_file(input_path, output_path, llm)
            payloads.append(payload)
            print(f"Saved JSON: {output_path}")

        csv_output = args.csv_output or args.output_dir / "final_summaries.csv"
        write_csv(csv_output, payloads)

        print(f"Saved CSV : {csv_output}")
        print(f"Total summaries: {len(payloads)}")
        return

    output_path = args.output or output_path_for_input(args.input, args.output_dir)
    payload = summarize_file(args.input, output_path, llm)

    print(f"Saved JSON: {output_path}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ModuleNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
