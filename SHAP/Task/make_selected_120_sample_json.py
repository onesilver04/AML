# 120 샘플 저장 형식
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTED_INPUT = (
    PROJECT_ROOT / "SHAP/Task/selected_120_confidence_relative_percent.csv"
)
DEFAULT_CONFIDENCE_INPUT = PROJECT_ROOT / "SHAP/confidence.csv"
DEFAULT_CORRECT_SHAP_DIR = PROJECT_ROOT / "SHAP/Task/correct_102_local_shap"
DEFAULT_WRONG_SHAP_DIR = PROJECT_ROOT / "SHAP/Task/wrong_18_local_shap"
DEFAULT_CORRECT_RAG_DIR = PROJECT_ROOT / "RAG/Final Summary/Correct_Results"
DEFAULT_WRONG_RAG_DIR = PROJECT_ROOT / "RAG/Final Summary/Wrong_Results"
DEFAULT_OUTPUT = PROJECT_ROOT / "SHAP/Task/selected_120_samples.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "SHAP/120_samples_individual_json"
DEFAULT_EXPLANATION_DIRS = [
    PROJECT_ROOT / "SHAP/Condition1/Results/selected_shap_only_explanations_wrong_ko",
    PROJECT_ROOT / "SHAP/Condition1/Results/selected_shap_only_explanations_correct_ko",
]

FEATURE_PREFIX = "feature_"
SHAP_FILE_PATTERN = "shap_tuples_non_prefix_*.json"
CUSTOMER_FIELD_RENAMES = {
    "duration": "대출 기간",
    "credit_amount": "대출 금액",
    "credit": "대출 금액",
    "installment_commitment": "월 상환 부담 수준",
    "installment": "월 상환 부담 수준",
    "residence_since": "현재 거주 기간",
    "age": "나이",
    "existing_credits": "기존 대출 수",
    "existing": "기존 대출 수",
    "num_dependents": "부양 가족 수",

    "checking_status_<0": "부족",
    "checking_status_0<=X<200": "적음",
    "checking_status_>=200": "충분",
    "checking_status_no checking": "입출금 계좌 없음",
    "checking_status": "계좌 잔액",

    "credit_history_all paid": "대출 전액 상환 이력",
    "credit_history_critical/other existing credit": "신용 문제 이력 있음",
    "credit_history_delayed previously": "연체 이력 있음",
    "credit_history_existing paid": "기존 대출 상환 중",
    "credit_history_no credits/all paid": "대출 이력 없음 또는 전액 상환",
    "credit_history": "신용 이력",

    "purpose_business": "사업 자금",
    "purpose_domestic appliance": "가전제품 구매",
    "purpose_education": "교육비 목적",
    "purpose_furniture/equipment": "가구 또는 장비 구매",
    "purpose_new car": "신차 구매",
    "purpose_used car": "중고차 구매",
    "purpose_radio/tv": "전자제품 구매",
    "purpose_repairs": "수리비",
    "purpose_retraining": "직업 재교육",
    "purpose_other": "기타",
    "purpose": "대출 목적",

    "savings_status_<100": "거의 없음",
    "savings_status_100<=X<500": "적음",
    "savings_status_500<=X<1000": "보통",
    "savings_status_>=1000": "충분",
    "savings_status_no known savings": "없음",
    "savings_status": "저축 잔액",

    "employment_<1": "1년 미만",
    "employment_1<=X<4": "1~4년",
    "employment_4<=X<7": "4~7년",
    "employment_>=7": "7년 이상",
    "employment_unemployed": "무직",
    "employment": "재직 기간",

    "housing_own": "자가 거주",
    "housing_rent": "임차 거주",
    "housing_for free": "무상 거주",
    "housing": "거주 상태",

    "job_high qualif/self emp/mgmt": "전문직, 자영업, 관리직",
    "job_skilled": "숙련 기술직",
    "job_unskilled resident": "단순 노무직",
    "job_unemp/unskilled non res": "무직, 단순직 외국인",
    "job": "직업",

    "Sex": "성별",
    "Married": "결혼 여부",
}
SHAP_FEATURE_RENAMES = {
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
    "credit_history_critical/other existing credit": "신용 문제 이력 있음",
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
}

CONDITION_1_SAMPLE_INDICES = {
    # True: none=26, strong=8 / False: none=5, weak=1
    10,
    13,
    14,
    16,
    30,
    39,
    45,
    47,
    50,
    51,
    56,
    57,
    63,
    72,
    77,
    81,
    83,
    100,
    110,
    111,
    120,
    123,
    124,
    126,
    136,
    142,
    144,
    146,
    169,
    172,
    176,
    177,
    178,
    179,
    182,
    183,
    187,
    189,
    194,
    197,
}
CONDITION_2_SAMPLE_INDICES = {
    # True: none=31, strong=3 / False: none=3, weak=3
    4,
    12,
    15,
    17,
    18,
    19,
    20,
    21,
    22,
    26,
    29,
    33,
    64,
    66,
    67,
    70,
    75,
    87,
    90,
    102,
    104,
    113,
    116,
    121,
    138,
    145,
    147,
    148,
    150,
    152,
    158,
    161,
    163,
    165,
    173,
    174,
    175,
    180,
    196,
    199,
}
CONDITION_3_SAMPLE_INDICES = {
    # True: weak=17, strong=17 / False: none=3, weak=3
    3,
    5,
    6,
    7,
    24,
    25,
    35,
    40,
    48,
    49,
    52,
    53,
    58,
    60,
    62,
    65,
    78,
    86,
    89,
    91,
    92,
    93,
    97,
    98,
    109,
    115,
    127,
    128,
    129,
    132,
    135,
    139,
    141,
    143,
    149,
    155,
    167,
    184,
    188,
    191,
}


def build_condition_by_sample() -> dict[int, int]:
    condition_sets = {
        1: CONDITION_1_SAMPLE_INDICES,
        2: CONDITION_2_SAMPLE_INDICES,
        3: CONDITION_3_SAMPLE_INDICES,
    }

    for condition, sample_indices in condition_sets.items():
        if len(sample_indices) != 40:
            raise ValueError(
                f"Condition {condition} must contain 40 samples, "
                f"got {len(sample_indices)}."
            )

    condition_by_sample: dict[int, int] = {}
    for condition, sample_indices in condition_sets.items():
        for sample_idx in sample_indices:
            if sample_idx in condition_by_sample:
                raise ValueError(
                    f"sample_idx={sample_idx} appears in both condition "
                    f"{condition_by_sample[sample_idx]} and {condition}."
                )
            condition_by_sample[sample_idx] = condition

    if len(condition_by_sample) != 120:
        raise ValueError(
            f"Condition mapping must contain 120 unique samples, "
            f"got {len(condition_by_sample)}."
        )

    return condition_by_sample


CONDITION_BY_SAMPLE = build_condition_by_sample()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-sample JSON payloads for the selected 120 test samples "
            "using confidence, relative percentile, customer data, and local SHAP."
        )
    )
    parser.add_argument("--selected-input", type=Path, default=DEFAULT_SELECTED_INPUT)
    parser.add_argument("--confidence-input", type=Path, default=DEFAULT_CONFIDENCE_INPUT)
    parser.add_argument("--correct-shap-dir", type=Path, default=DEFAULT_CORRECT_SHAP_DIR)
    parser.add_argument("--wrong-shap-dir", type=Path, default=DEFAULT_WRONG_SHAP_DIR)
    parser.add_argument("--correct-rag-dir", type=Path, default=DEFAULT_CORRECT_RAG_DIR)
    parser.add_argument("--wrong-rag-dir", type=Path, default=DEFAULT_WRONG_RAG_DIR)
    parser.add_argument(
        "--explanation-dirs",
        nargs="+",
        type=Path,
        default=DEFAULT_EXPLANATION_DIRS,
        help="Directories of sample_*_shap_only_summary_ko.json files for explanation.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Only write the combined JSON array, not one JSON file per sample.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_value(value: str) -> Any:
    if value is None:
        return None

    stripped = value.strip()
    if stripped == "":
        return None

    lower = stripped.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False

    try:
        number = float(stripped)
    except ValueError:
        return stripped

    if number.is_integer():
        return int(number)
    return number


def parse_bool(value: str) -> bool:
    lower = str(value).strip().lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    raise ValueError(f"Expected boolean value, got: {value}")


def distance_from_sigma_group(sigma_group: str) -> str:
    if sigma_group.startswith("within"):
        return "near"
    if sigma_group.startswith("outside"):
        return "far"
    raise ValueError(f"Unexpected sigma_group: {sigma_group}")


def warning_type_from_confidence(value: str | None) -> str:
    if value is None:
        return "none"

    normalized = value.strip().lower()
    if normalized in {"strong", "strong_warning"}:
        return "strong"
    if normalized in {"weak", "weak_warning"}:
        return "weak"
    if normalized in {"", "none", "no_warning"}:
        return "none"
    raise ValueError(f"Unexpected warning_type: {value}")


def display_predicted_label(value: str) -> str:
    if value == "GOOD CREDIT RISK":
        return "낮음"
    if value == "BAD CREDIT RISK":
        return "높음"
    return value


def load_confidence_by_sample(path: Path) -> dict[int, dict[str, str]]:
    confidence_by_sample: dict[int, dict[str, str]] = {}
    for row in read_csv_rows(path):
        sample_idx = int(row["sample_idx"])
        confidence_by_sample[sample_idx] = row
    return confidence_by_sample


def sample_idx_from_shap_path(path: Path) -> int:
    match = re.search(r"shap_tuples_non_prefix_(\d+)\.json$", path.name)
    if not match:
        raise ValueError(f"Unexpected SHAP file name: {path.name}")
    return int(match.group(1))


def load_shap_top3_by_sample(shap_dirs: list[Path]) -> dict[int, list[dict[str, Any]]]:
    top3_by_sample: dict[int, list[dict[str, Any]]] = {}

    for shap_dir in shap_dirs:
        shap_dir = resolve_path(shap_dir)
        if not shap_dir.exists():
            raise FileNotFoundError(f"SHAP directory does not exist: {shap_dir}")

        for path in sorted(shap_dir.glob(SHAP_FILE_PATTERN)):
            sample_idx = sample_idx_from_shap_path(path)
            with path.open(encoding="utf-8") as f:
                payload = json.load(f)

            tuples = payload.get("tuples", [])
            top3_by_sample[sample_idx] = [
                {
                    "feature": SHAP_FEATURE_RENAMES.get(item["feature"], item["feature"]),
                    "value": round(float(item["shap_value"]), 2),
                    "_raw_feature": item["feature"],
                }
                for item in tuples[:3]
            ]

    return top3_by_sample


def sample_idx_from_rag_path(path: Path) -> int:
    match = re.search(r"sample_(\d+)_final_summary(?:_ko)?\.json$", path.name)
    if not match:
        raise ValueError(f"Unexpected RAG summary file name: {path.name}")
    return int(match.group(1))


def clean_evidence_sentence(evidence_sentence: dict[str, Any]) -> dict[str, Any]:
    cleaned = {
        "sentence": evidence_sentence.get("sentence"),
        "source": evidence_sentence.get("source"),
        "page": evidence_sentence.get("page"),
        "score": evidence_sentence.get("score"),
    }
    return {key: value for key, value in cleaned.items() if value is not None}


def load_rag_evidence_by_sample(
    rag_dirs: list[Path],
) -> dict[int, dict[str, dict[str, Any]]]:
    evidence_by_sample: dict[int, dict[str, dict[str, Any]]] = {}
    for rag_dir in rag_dirs:
        rag_dir = resolve_path(rag_dir)
        if not rag_dir.exists():
            raise FileNotFoundError(f"RAG directory does not exist: {rag_dir}")

        for path in sorted(rag_dir.glob("sample_*_final_summary.json")):
            sample_idx = sample_idx_from_rag_path(path)
            with path.open(encoding="utf-8") as f:
                payload = json.load(f)

            evidence_by_feature: dict[str, dict[str, Any]] = {}
            for item in payload.get("evidence", []):
                feature = item.get("feature")
                evidence_sentences = item.get("evidence_sentences") or []
                if feature and evidence_sentences:
                    evidence_by_feature[feature] = clean_evidence_sentence(
                        evidence_sentences[0]
                    )

            evidence_by_sample[sample_idx] = evidence_by_feature

    return evidence_by_sample


def sample_idx_from_explanation_path(path: Path) -> int:
    match = re.search(r"sample_(\d+)_shap_only_summary_ko\.json$", path.name)
    if not match:
        raise ValueError(f"Unexpected explanation file name: {path.name}")
    return int(match.group(1))


def load_explanations_by_sample(explanation_dirs: list[Path]) -> dict[int, str]:
    explanations_by_sample: dict[int, str] = {}

    for explanation_dir in explanation_dirs:
        explanation_dir = resolve_path(explanation_dir)
        if not explanation_dir.exists():
            raise FileNotFoundError(f"Explanation directory does not exist: {explanation_dir}")

        for path in sorted(explanation_dir.glob("sample_*_shap_only_summary_ko.json")):
            sample_idx = sample_idx_from_explanation_path(path)
            with path.open(encoding="utf-8") as f:
                payload = json.load(f)

            explanation = payload.get("final_explanation_ko")
            if isinstance(explanation, str) and explanation.strip():
                explanations_by_sample[sample_idx] = explanation.strip()

    return explanations_by_sample


def load_rag_explanations_by_sample(
    rag_dirs: list[Path],
    existing_explanations: dict[int, str],
) -> dict[int, str]:
    rag_explanations_by_sample: dict[int, str] = {}

    for rag_dir in rag_dirs:
        rag_dir = resolve_path(rag_dir)
        if not rag_dir.exists():
            raise FileNotFoundError(f"RAG directory does not exist: {rag_dir}")

        for path in sorted(rag_dir.glob("*.json")):
            with path.open(encoding="utf-8") as f:
                payload = json.load(f)

            if not isinstance(payload, dict):
                continue
            if "explanation" in payload:
                continue

            sample_idx = payload.get("sample_idx")
            if sample_idx is None:
                try:
                    sample_idx = sample_idx_from_rag_path(path)
                except ValueError:
                    continue
            sample_idx = int(sample_idx)

            if sample_idx in existing_explanations:
                continue

            explanation = payload.get("final_explanation_ko")
            if isinstance(explanation, str) and explanation.strip():
                rag_explanations_by_sample[sample_idx] = explanation.strip()

    return rag_explanations_by_sample

def rename_customer_value(raw_customer_key: str, parsed_value: Any) -> Any:
    if raw_customer_key == "Sex" and isinstance(parsed_value, bool):
        return "남성" if parsed_value else "여성"
    if raw_customer_key == "Married" and isinstance(parsed_value, bool):
        return "o" if parsed_value else "x"
    if not isinstance(parsed_value, str):
        return parsed_value

    mapped_value_key = f"{raw_customer_key}_{parsed_value}"
    return CUSTOMER_FIELD_RENAMES.get(mapped_value_key, parsed_value)


def build_customer_data(row: dict[str, str]) -> dict[str, Any]:
    customer_data: dict[str, Any] = {}
    for key, value in row.items():
        if not key.startswith(FEATURE_PREFIX):
            continue

        raw_customer_key = key.removeprefix(FEATURE_PREFIX)
        customer_key = CUSTOMER_FIELD_RENAMES.get(raw_customer_key, raw_customer_key)
        parsed_value = rename_customer_value(raw_customer_key, parse_value(value))

        customer_data[customer_key] = parsed_value

    return customer_data


def build_sample_payload(
    row: dict[str, str],
    confidence_by_sample: dict[int, dict[str, str]],
    shap_top3_by_sample: dict[int, list[dict[str, Any]]],
    rag_evidence_by_sample: dict[int, dict[str, dict[str, Any]]],
    explanations_by_sample: dict[int, str],
) -> dict[str, Any]:
    sample_idx = int(row["sample_idx"])
    if sample_idx not in shap_top3_by_sample:
        raise ValueError(f"Missing local SHAP JSON for sample_idx={sample_idx}")
    if sample_idx not in confidence_by_sample:
        raise ValueError(f"Missing confidence.csv row for sample_idx={sample_idx}")

    confidence_row = confidence_by_sample[sample_idx]
    for key in ("true_label", "predicted_label", "is_correct"):
        if str(row[key]) != str(confidence_row[key]):
            raise ValueError(
                f"Mismatch for sample_idx={sample_idx}, field={key}: "
                f"selected={row[key]}, confidence={confidence_row[key]}"
            )

    local_shap_top3_features = []
    rag_evidence_by_feature = rag_evidence_by_sample.get(sample_idx, {})
    for shap_feature in shap_top3_by_sample[sample_idx]:
        feature_payload = dict(shap_feature)
        raw_feature = feature_payload.pop("_raw_feature", feature_payload["feature"])
        evidence = rag_evidence_by_feature.get(raw_feature)
        if evidence:
            feature_payload["evidence_sentence"] = evidence
        local_shap_top3_features.append(feature_payload)

    if sample_idx not in CONDITION_BY_SAMPLE:
        raise ValueError(f"No condition mapping for sample_idx={sample_idx}")

    payload = {
        "sample_idx": sample_idx,
        "condition": CONDITION_BY_SAMPLE[sample_idx],
    }

    payload.update(
        {
            "distance": distance_from_sigma_group(row["sigma_group"]),
            "is_correct": parse_bool(row["is_correct"]),
            "customer_data": build_customer_data(row),
            "predicted_label": display_predicted_label(row["predicted_label"]),
            "confidence": round(float(confidence_row["predicted_confidence"]) * 100),
            "true_label": display_predicted_label(row["true_label"]),
            "local_shap_top3_features": local_shap_top3_features,
            "explanation": explanations_by_sample.get(sample_idx),
            "class_confidence_relative_percent": round(
                float(row["class_confidence_relative_percent"]), 2
            ),
            "decision_boundary_abs_distance": float(row["decision_boundary_abs_distance"]),
            "warning_type": warning_type_from_confidence(
                confidence_row.get("warning_type")
            ),
        }
    )
    return payload


def write_json(path: Path, payload: Any) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()

    selected_rows = read_csv_rows(args.selected_input)
    if len(selected_rows) != 120:
        raise ValueError(f"Expected 120 selected rows, got {len(selected_rows)}")

    confidence_by_sample = load_confidence_by_sample(args.confidence_input)
    shap_top3_by_sample = load_shap_top3_by_sample(
        [args.correct_shap_dir, args.wrong_shap_dir]
    )
    rag_evidence_by_sample = load_rag_evidence_by_sample(
        [args.correct_rag_dir, args.wrong_rag_dir]
    )
    explanations_by_sample = load_explanations_by_sample(args.explanation_dirs)
    explanations_by_sample.update(
        load_rag_explanations_by_sample(
            [args.correct_rag_dir, args.wrong_rag_dir],
            explanations_by_sample,
        )
    )

    payloads = [
        build_sample_payload(
            row,
            confidence_by_sample,
            shap_top3_by_sample,
            rag_evidence_by_sample,
            explanations_by_sample,
        )
        for row in selected_rows
    ]

    write_json(args.output, payloads)

    if not args.no_individual:
        output_dir = resolve_path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for payload in payloads:
            write_json(output_dir / f"sample_{payload['sample_idx']}.json", payload)

    print(f"Selected samples : {len(payloads)}")
    print(f"Combined output  : {resolve_path(args.output)}")
    if not args.no_individual:
        print(f"Individual output: {resolve_path(args.output_dir)}")


if __name__ == "__main__":
    main()
