import argparse
import json
import sys
from pathlib import Path
import re

from langchain_openai import ChatOpenAI


DEFAULT_INPUT = Path("RAG/Final Summary/250 Results/Condition 3/False/Far")
DEFAULT_OUTPUT = Path("RAG/Final Summary/250 Results/Condition 3/False/Far_ko")
MODEL_NAME = "LGAI-EXAONE/EXAONE-4.5-33B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate final_explanation using Ollama EXAONE"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--sample-indices",
        default=None,
        help="Comma-separated sample indices. Example: 32,160,153",
    )   
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=MODEL_NAME)
    return parser.parse_args()


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_index_text(text: str):
    if not text:
        return None

    return [
        int(part.strip())
        for part in text.replace("\n", ",").split(",")
        if part.strip()
    ]


def get_sample_idx_from_filename(path: Path):
    match = re.search(r"sample_(\d+)(?:_|\.|$)", path.name)
    if not match:
        return None
    return int(match.group(1))


def filter_json_files_by_sample_indices(json_files, target_indices):
    if not target_indices:
        return json_files

    target_set = set(target_indices)

    return [
        f for f in json_files
        if get_sample_idx_from_filename(f) in target_set
    ]


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def output_path_for_input(input_path: Path):
    return input_path.with_name(f"{input_path.stem}_ko{input_path.suffix}")

FEATURE_NAME_MAP = {
    "duration": "대출 기간",
    "credit_amount": "대출 금액",
    "installment_commitment": "월 상환 부담 수준",
    "residence_since": "현재 거주 기간",
    "age": "나이",
    "existing_credits": "기존 대출 수",
    "num_dependents": "부양 가족 수",

    "checking_status_<0": "계좌 잔액 부족",
    "checking_status_0<=X<200": "계좌 잔액 적음",
    "checking_status_>=200": "계좌 잔액 충분",
    "checking_status_no checking": "입출금 계좌 없음",

    "credit_history_all paid": "대출 전액 상환 이력",
    "credit_history_critical/other existing credit": "신용 문제 이력",
    "credit_history_delayed previously": "연체 이력",
    "credit_history_existing paid": "기존 대출 상환 중",
    "credit_history_no credits/all paid": "대출 이력 없음 또는 전액 상환",

    "purpose_business": "사업 자금 목적",
    "purpose_domestic appliance": "가전제품 구매 목적",
    "purpose_education": "교육비 목적",
    "purpose_furniture/equipment": "가구 또는 장비 구매 목적",
    "purpose_new car": "신차 구매 목적",
    "purpose_used car": "중고차 구매 목적",
    "purpose_radio/tv": "전자제품 구매 목적",
    "purpose_repairs": "수리비 목적",
    "purpose_retraining": "직업 재교육 목적",
    "purpose_other": "기타 목적",

    "savings_status_<100": "저축 잔액이 거의 없음",
    "저축 잔액 100 DM 미만": "저축 잔액이 거의 없음",
    "savings_status_100<=X<500": "저축 잔액 적음",
    "savings_status_500<=X<1000": "저축 잔액 보통",
    "savings_status_>=1000": "저축 잔액 충분",
    "savings_status_no known savings": "저축 내역 없음",

    "employment_<1": "재직 기간 1년 미만",
    "employment_1<=X<4": "재직 기간 1~4년",
    "employment_4<=X<7": "재직 기간 4~7년",
    "employment_>=7": "재직 기간 7년 이상",
    "employment_unemployed": "무직",

    "personal_status_female div/dep/mar": "여성 (이혼/별거/기혼 상태)",
    "personal_status_male div/sep": "남성 (이혼 또는 별거)",
    "personal_status_male mar/wid": "남성 (기혼 또는 사별)",
    "personal_status_male single": "남성 (미혼)",

    "other_parties_co applicant": "공동 신청자 있음",
    "other_parties_guarantor": "보증인 있음",
    "other_parties_none": "공동 신청자, 보증인 없음",

    "property_magnitude_real estate": "부동산 보유",
    "property_magnitude_life insurance": "보험 자산 보유",
    "property_magnitude_car": "차량 보유",
    "property_magnitude_no known property": "보유 자산 없음",

    "other_payment_plans_bank": "타 은행 상환 중",
    "other_payment_plans_stores": "할부·외상 상환 중",
    "other_payment_plans_none": "추가 상환 없음",

    "housing_own": "자가 거주",
    "housing_rent": "임차 거주",
    "housing_for free": "무상 거주",

    "job_high qualif/self emp/mgmt": "전문직, 자영업, 관리직",
    "job_skilled": "숙련 기술직",
    "job_unskilled resident": "단순 노무직",
    "job_unemp/unskilled non res": "무직, 단순직 외국인",

    "own_telephone_yes": "전화 보유",
    "own_telephone_none": "전화 미보유",

    "foreign_worker_yes": "외국인 근로자",
    "foreign_worker_no": "내국인 근로자",

    "class": "신용 등급",
    "Sex": "성별",
    "Married": "결혼 여부",
    "신청자는":"고객은",
    "default":"채무 불이행"
}


def force_replace_feature_names(text: str):
    replaced = text

    for raw_name, korean_name in sorted(
        FEATURE_NAME_MAP.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    ):
        replaced = replaced.replace(f"[{raw_name}]", korean_name)
        replaced = replaced.replace(raw_name, korean_name)

    return replaced


def build_feature_mapping_text():
    return "\n".join(
        f'- "{raw}" -> "{ko}"'
        for raw, ko in FEATURE_NAME_MAP.items()
    )
    
def build_prompt(text: str):
    feature_mapping_text = build_feature_mapping_text()

    return f"""You are a financial risk analyst specialized in credit scoring, SHAP-based explanations, and RAG-grounded reasoning.

Your job is to generate a user-friendly Korean explanation for credit risk predictions.

IMPORTANT:
- You must translate the entire text naturally into Korean.

HOWEVER:
- feature names are fixed terms.
- These must follow the FEATURE_NAME_MAP exactly.

GENERAL RULES:
- Output ONLY Korean.
- Preserve the meaning and tone.
- Do NOT use markdown.
- Do NOT add new information.
- Preserve citation markers [1], [2], [3] exactly.
- The output must have exactly 4 sentences:
  1. one sentence ending with [1]
  2. one sentence ending with [2]
  3. one sentence ending with [3]
  4. one final overall sentence
- Do NOT create extra introductory sentences.
- Do NOT create extra follow-up sentences.
- Do NOT split one feature explanation into multiple sentences.

OPENING SENTENCE RULE (CRITICAL):
- The FIRST sentence must ALWAYS start with:
본 AI가 예측한 결과에 따르면,\\n
- This prefix must appear exactly once at the very beginning.
- Do NOT modify or paraphrase this phrase.
- Do NOT create a separate introductory sentence.
- The first feature explanation must come immediately after this prefix.

MODEL REFERENCE RULE (CRITICAL):
- Any expression referring to the model as a subject must use "본 모델은".
- Do NOT use generic expressions such as "모델은", "이 모델은", "AI 모델은".
- Replace them with "본 모델은".
- Examples:
  WRONG: "모델은 이를 위험 감소 요인으로 판단합니다."
  WRONG: "이 모델은 이를 위험 감소 요인으로 판단합니다."
  WRONG: "AI 모델은 이를 위험 감소 요인으로 판단합니다."
  CORRECT: "본 모델은 이를 위험 감소 요인으로 판단합니다."

TRANSLATION RULES:
- Translate the entire text into natural Korean.
- Preserve the meaning and tone.
- Do NOT add or remove content.
- Do NOT summarize.
- Do NOT restructure into a different number of sentences.
- Translate all sentences normally EXCEPT the final sentence; the final sentence must be replaced by the fixed template in FINAL SENTENCE OVERRIDE RULE.

FEATURE NAME RULE (CRITICAL):
- You MUST follow FEATURE_NAME_MAP exactly.
- The Korean feature name must be selected ONLY from FEATURE_NAME_MAP.
- Do NOT create your own Korean feature names.
- Do NOT paraphrase feature names.
- Do NOT wrap feature names in brackets.
- The only bracketed tokens allowed in the output are [1], [2], [3].
- If a feature appears in the input, replace it with the exact Korean value from FEATURE_NAME_MAP.
- If a feature is not in FEATURE_NAME_MAP, keep the original expression as-is.

FEATURE INTERPRETATION BAN (CRITICAL):
- Do NOT interpret or explain feature names.
- Do NOT expand feature names into descriptive phrases.

For example:
WRONG:
"체크링 계좌 잔액이 음수인 것은"
"계좌 잔액이 0보다 작은 경우"
"체크 계좌 잔액이 음수라는 것은"
CORRECT:
"계좌 잔액 부족"

STRICT MAPPING ENFORCEMENT:

- Feature names must be used EXACTLY as defined in FEATURE_NAME_MAP.
- You must NOT change grammar or wording of mapped feature names.

WRONG:
"체크링 계좌 잔액이 음수인 것은"
"계좌 잔액이 부족한 경우"

CORRECT:
"계좌 잔액 부족"

FEATURE TERM OVERRIDE RULE (CRITICAL):

- Words like "checking", "savings", "credit", "account" must NOT be translated individually.

- You must NOT translate partial feature terms.

- You must NOT produce phrases like:
  "체크 계좌"
  "체크링 계좌"
  "체킹 계좌"

- These are INVALID outputs.

- Feature-related terms must ONLY appear through FEATURE_NAME_MAP.

- If a phrase contains a feature, you must replace the ENTIRE phrase using FEATURE_NAME_MAP,
  not translate it word-by-word.

ANTI-REDUNDANCY RULE:
- Do NOT repeat the same feature meaning twice.
- Do NOT use phrases such as "즉", "다시 말해", "예를 들어", "특히", "이는 곧".
- Do NOT write patterns like "A, 즉 B" or "A (B)".
- Each [1], [2], [3] sentence must mention the feature only once.
- Do NOT explain the feature and then restate the mapped feature name.
- Keep each sentence concise.

FINAL SENTENCE OVERRIDE RULE (CRITICAL):

- Ignore the meaning and wording of the original English final sentence.
- Do NOT translate the original English final sentence.
- The original final sentence is only used to decide whether the result is LOW risk or HIGH risk.

If the original final sentence contains:
- "low risk"
- "low risk of default"
- "GOOD CREDIT RISK"

Then copy this sentence exactly:
전반적으로 신용 위험이 낮은 수준으로 평가되어 대출 승인 가능성이 높습니다.

If the original final sentence contains:
- "high risk"
- "high risk of default"
- "BAD CREDIT RISK"

Then copy this sentence exactly:
전반적으로 신용 위험이 높은 수준으로 평가되어 대출 승인 가능성이 낮습니다.

- The final sentence is a fixed template, not a translation target.
- Do NOT use words such as "디폴트", "신청자는" in the final sentence.

FEATURE_NAME_MAP:
{feature_mapping_text}

Text:
{text}

Korean:
"""

def load_model(model_name: str):
    return ChatOpenAI(
        model=model_name,
        openai_api_base="http://localhost:8000/v1",
        openai_api_key="EMPTY",          # vLLM은 키 불필요, 아무 값이나 가능
        temperature=0.0,
        max_tokens=1024,
    )
    
def final_sentence_from_source(text: str) -> str:
    source = (text or "").lower()

    if "bad credit risk" in source or "high risk" in source or "high risk of default" in source:
        return "전반적으로 신용 위험이 높은 수준으로 평가되어 대출 승인 가능성이 낮습니다."

    if "good credit risk" in source or "low risk" in source or "low risk of default" in source:
        return "전반적으로 신용 위험이 낮은 수준으로 평가되어 대출 승인 가능성이 높습니다."

    return "전반적으로 신용 위험 수준을 명확히 판단하기 어렵습니다."

def translate(text: str, llm):
    text = force_replace_feature_names(text)

    prompt = build_prompt(text)
    result = llm.invoke(prompt)

    translated = result.content.strip()
    translated = force_replace_feature_names(translated)

    opening = "본 AI가 예측한 결과에 따르면,\\n"

    translated = re.sub(
        r"^(본\s*AI가\s*예측한\s*결과에\s*따르면,?\s*(\\n|\n)?|"
        r"본\s*AI\s*모델의\s*예측에\s*따르면,?\s*(\\n|\n)?|"
        r"본\s*모델의\s*예측에\s*따르면,?\s*(\\n|\n)?|"
        r"AI\s*모델의\s*예측에\s*따르면,?\s*(\\n|\n)?)",
        "",
        translated,
    ).strip()

    translated = re.sub(r"(이|AI)\s*모델", "본 모델", translated)
    translated = re.sub(r"(?<!본\s)모델", "본 모델", translated)
    translated = translated.replace("본 본 모델", "본 모델")
    
    fixed_final_sentence = final_sentence_from_source(text)

    translated = re.sub(
        r"전반적으로.*?(평가됩니다|가능성이\s*(높|낮)습니다)\.?",
        "",
        translated,
        flags=re.DOTALL,
    ).strip()

    translated = re.sub(r"\n\s*\n", "\n", translated).strip()

    translated = opening + translated
    translated = translated.rstrip() + "\n" + fixed_final_sentence

    return translated


def collect_final_explanations_ko(input_dir: Path, output_path: Path, target_indices):
    results = []

    json_files = sorted(input_dir.glob("*.json"))
    json_files = [
        f for f in json_files
        if f.stem.endswith("_ko")
    ]

    json_files = filter_json_files_by_sample_indices(json_files, target_indices)

    for json_path in json_files:
        data = load_json(json_path)

        if "final_explanation_ko" in data:
            results.append(
                {
                    "file": json_path.name,
                    "final_explanation_ko": data["final_explanation_ko"],
                }
            )

    write_json(output_path, results)
    print(f"Collected file saved: {output_path}")


def main():
    args = parse_args()

    llm = load_model(args.model)

    input_path = args.input
    target_indices = parse_index_text(args.sample_indices)

    if input_path.is_dir():
        json_files = sorted(input_path.glob("*.json"))

        json_files = [
            f for f in json_files
            if not f.stem.endswith("_ko")
        ]

        json_files = filter_json_files_by_sample_indices(
            json_files,
            target_indices,
        )

        if not json_files:
            raise FileNotFoundError(f"No matching JSON files found in: {input_path}")

        for json_path in json_files:
            payload = load_json(json_path)

            if "final_explanation" not in payload:
                print(f"SKIP: final_explanation field missing in {json_path}")
                continue

            translated = translate(payload["final_explanation"], llm)
            payload["final_explanation_ko"] = translated

            output_path = output_path_for_input(json_path)
            write_json(output_path, payload)

            print(f"Saved: {output_path}")

        collect_output_path = input_path / "final_explanations_ko.json"
        collect_final_explanations_ko(
            input_path,
            collect_output_path,
            target_indices,
        )

    else:
        payload = load_json(input_path)

        sample_idx = get_sample_idx_from_filename(input_path)

        if target_indices and sample_idx not in set(target_indices):
            raise ValueError(
                f"Input file sample_idx={sample_idx} is not in --sample-indices."
            )

        if "final_explanation" not in payload:
            raise ValueError("final_explanation field missing.")

        translated = translate(payload["final_explanation"], llm)
        payload["final_explanation_ko"] = translated

        output_path = args.output or output_path_for_input(input_path)
        write_json(output_path, payload)

        print(f"Saved: {output_path}")
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)