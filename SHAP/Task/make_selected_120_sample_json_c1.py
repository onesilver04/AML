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

DEFAULT_CONFIDENCE_GOOD_INPUT = PROJECT_ROOT / "SHAP/confidence_good.csv"
DEFAULT_CONFIDENCE_BAD_INPUT = PROJECT_ROOT / "SHAP/confidence_bad.csv"

DEFAULT_COND1_CORRECT_DIR = (
    PROJECT_ROOT
    / "SHAP/Condition1/Results/selected_shap_only_explanations_correct_ko"
)
DEFAULT_COND1_WRONG_DIR = (
    PROJECT_ROOT
    / "SHAP/Condition1/Results/selected_shap_only_explanations_wrong_ko"
)

DEFAULT_OUTPUT = PROJECT_ROOT / "SHAP/Condition1/Results/condition1_selected_samples.json"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "SHAP/Condition1/Results/condition1_selected_samples_individual_json"
)

FEATURE_PREFIX = "feature_"


# 여기에 기존 CUSTOMER_FIELD_RENAMES 그대로 붙여넣기
# CUSTOMER_FIELD_RENAMES = {...}
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

CONDITION_1_SAMPLE_INDICES = {
    10, 45, 81, 83, 100, 123, 126, 136, 142, 146,
    169, 176, 177, 187, 189, 194, 197, 77, 110, 111,
    47, 63, 178, 14, 16, 120, 124, 144, 172, 179,
    182, 32, 160, 153, 117, 61, 42, 39, 183, 13,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build selected sample JSON files for Condition1 only."
    )

    parser.add_argument("--selected-input", type=Path, default=DEFAULT_SELECTED_INPUT)
    parser.add_argument("--confidence-input", type=Path, default=DEFAULT_CONFIDENCE_INPUT)

    parser.add_argument(
        "--confidence-good-input",
        type=Path,
        default=DEFAULT_CONFIDENCE_GOOD_INPUT,
    )
    parser.add_argument(
        "--confidence-bad-input",
        type=Path,
        default=DEFAULT_CONFIDENCE_BAD_INPUT,
    )

    parser.add_argument("--cond1-correct-dir", type=Path, default=DEFAULT_COND1_CORRECT_DIR)
    parser.add_argument("--cond1-wrong-dir", type=Path, default=DEFAULT_COND1_WRONG_DIR)

    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Only write the combined JSON array.",
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


def load_class_confidence_by_sample(
    good_path: Path,
    bad_path: Path,
) -> dict[int, float]:
    """
    class_confidence_relative_percent에 넣을 값을 만든다.

    - SHAP/confidence_good.csv 안의 predicted_confidence
    - SHAP/confidence_bad.csv 안의 predicted_confidence

    두 파일을 합쳐서 sample_idx 기준으로 조회한다.
    """

    class_confidence_by_sample: dict[int, float] = {}

    for row in read_csv_rows(good_path):
        sample_idx = int(row["sample_idx"])
        class_confidence_by_sample[sample_idx] = float(row["predicted_confidence"])

    for row in read_csv_rows(bad_path):
        sample_idx = int(row["sample_idx"])
        class_confidence_by_sample[sample_idx] = float(row["predicted_confidence"])

    return class_confidence_by_sample


def sample_idx_from_condition1_path(path: Path) -> int:
    match = re.search(r"sample_(\d+)_shap_only_summary_ko\.json$", path.name)

    if not match:
        raise ValueError(f"Unexpected Condition1 file name: {path.name}")

    return int(match.group(1))


def load_condition1_payloads_by_sample(
    dirs: list[Path],
) -> dict[int, dict[str, Any]]:
    payloads_by_sample: dict[int, dict[str, Any]] = {}

    for directory in dirs:
        directory = resolve_path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Condition1 explanation directory does not exist: {directory}"
            )

        for path in sorted(directory.glob("sample_*_shap_only_summary_ko.json")):
            sample_idx = sample_idx_from_condition1_path(path)

            with path.open(encoding="utf-8") as f:
                payload = json.load(f)

            payloads_by_sample[sample_idx] = payload

    return payloads_by_sample


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


def get_condition1_explanation(condition1_payload: dict[str, Any]) -> str | None:
    candidates = [
        "final_explanation_ko",
        "explanation",
        "shap_only_explanation_ko",
        "summary_ko",
    ]

    for key in candidates:
        value = condition1_payload.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def get_condition1_top3_features(
    condition1_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = [
        "local_shap_top3_features",
        "shap_top3_features",
        "top3_features",
        "features",
    ]

    for key in candidates:
        value = condition1_payload.get(key)

        if isinstance(value, list):
            return value

    return []


def build_sample_payload(
    row: dict[str, str],
    confidence_by_sample: dict[int, dict[str, str]],
    class_confidence_by_sample: dict[int, float],
    condition1_payloads_by_sample: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    sample_idx = int(row["sample_idx"])

    if sample_idx not in CONDITION_1_SAMPLE_INDICES:
        raise ValueError(f"sample_idx={sample_idx} is not a Condition1 sample.")

    if sample_idx not in confidence_by_sample:
        raise ValueError(f"Missing confidence.csv row for sample_idx={sample_idx}")

    if sample_idx not in class_confidence_by_sample:
        raise ValueError(
            f"Missing confidence_good/bad.csv row for sample_idx={sample_idx}"
        )

    if sample_idx not in condition1_payloads_by_sample:
        raise ValueError(f"Missing Condition1 JSON for sample_idx={sample_idx}")

    confidence_row = confidence_by_sample[sample_idx]
    condition1_payload = condition1_payloads_by_sample[sample_idx]

    for key in ("true_label", "predicted_label", "is_correct"):
        if str(row[key]) != str(confidence_row[key]):
            raise ValueError(
                f"Mismatch for sample_idx={sample_idx}, field={key}: "
                f"selected={row[key]}, confidence={confidence_row[key]}"
            )

    explanation = get_condition1_explanation(condition1_payload)
    local_shap_top3_features = get_condition1_top3_features(condition1_payload)

    class_confidence_value = class_confidence_by_sample[sample_idx]

    payload = {
        "sample_idx": sample_idx,
        "condition": 1,
        "distance": distance_from_sigma_group(row["sigma_group"]),
        "is_correct": parse_bool(row["is_correct"]),
        "customer_data": build_customer_data(row),
        "predicted_label": display_predicted_label(row["predicted_label"]),
        "confidence": round(float(confidence_row["predicted_confidence"]) * 100),
        "true_label": display_predicted_label(row["true_label"]),
        "local_shap_top3_features": local_shap_top3_features,
        "explanation": explanation,

        # 기존 selected_120_confidence_relative_percent.csv의
        # class_confidence_relative_percent를 쓰지 않고,
        # confidence_good.csv / confidence_bad.csv의 predicted_confidence를 사용
        "class_confidence_relative_percent": round(class_confidence_value, 4),

        "decision_boundary_abs_distance": float(row["decision_boundary_abs_distance"]),
        "warning_type": warning_type_from_confidence(
            confidence_row.get("warning_type")
        ),
    }

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

    condition1_rows = [
        row for row in selected_rows
        if int(row["sample_idx"]) in CONDITION_1_SAMPLE_INDICES
    ]

    if len(condition1_rows) != 40:
        raise ValueError(f"Expected 40 Condition1 rows, got {len(condition1_rows)}")

    confidence_by_sample = load_confidence_by_sample(args.confidence_input)

    class_confidence_by_sample = load_class_confidence_by_sample(
        good_path=args.confidence_good_input,
        bad_path=args.confidence_bad_input,
    )

    condition1_payloads_by_sample = load_condition1_payloads_by_sample(
        [args.cond1_correct_dir, args.cond1_wrong_dir]
    )

    missing_condition1_files = sorted(
        sample_idx
        for sample_idx in CONDITION_1_SAMPLE_INDICES
        if sample_idx not in condition1_payloads_by_sample
    )

    if missing_condition1_files:
        raise ValueError(
            "Missing Condition1 JSON files for sample_idx: "
            + ", ".join(map(str, missing_condition1_files))
        )

    missing_class_confidence = sorted(
        sample_idx
        for sample_idx in CONDITION_1_SAMPLE_INDICES
        if sample_idx not in class_confidence_by_sample
    )

    if missing_class_confidence:
        raise ValueError(
            "Missing confidence_good/bad.csv rows for sample_idx: "
            + ", ".join(map(str, missing_class_confidence))
        )

    payloads = [
        build_sample_payload(
            row=row,
            confidence_by_sample=confidence_by_sample,
            class_confidence_by_sample=class_confidence_by_sample,
            condition1_payloads_by_sample=condition1_payloads_by_sample,
        )
        for row in condition1_rows
    ]

    write_json(args.output, payloads)

    if not args.no_individual:
        output_dir = resolve_path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for payload in payloads:
            write_json(output_dir / f"sample_{payload['sample_idx']}.json", payload)

    print(f"Selected Condition1 samples : {len(payloads)}")
    print(f"Combined output             : {resolve_path(args.output)}")

    if not args.no_individual:
        print(f"Individual output           : {resolve_path(args.output_dir)}")


if __name__ == "__main__":
    main()