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

DEFAULT_SHAP_DIR = PROJECT_ROOT / "SHAP/All Dataset Local Shap"
DEFAULT_RAG_CORRECT_DIR = PROJECT_ROOT / "RAG/Final Summary/Correct_Results"
DEFAULT_RAG_WRONG_DIR = PROJECT_ROOT / "RAG/Final Summary/New_Wrong_Results"

DEFAULT_OUTPUT = PROJECT_ROOT / "SHAP/Condition3/Results/condition3_selected_samples.json"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "SHAP/Condition3/Results/condition3_selected_samples_individual_json"
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

CONDITION_3_SAMPLE_INDICES = {
5, 6, 53, 58, 65, 86, 98, 128, 132, 139, 141, 155, 184, 25, 40, 62, 97,	8,11,41,
91, 115, 129, 188, 24, 48, 78, 93, 127, 135, 7, 52, 89, 92, 143, 167, 191,	49, 60, 149}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build selected sample JSON files for Condition3 from mixed RAG/QA/Answers folders."
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

    parser.add_argument("--shap-dir", type=Path, default=DEFAULT_SHAP_DIR)
    parser.add_argument("--rag-correct-dir", type=Path, default=DEFAULT_RAG_CORRECT_DIR)
    parser.add_argument("--rag-wrong-dir", type=Path, default=DEFAULT_RAG_WRONG_DIR)

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
) -> dict[tuple[int, int], float]:
    class_confidence_by_sample: dict[tuple[int, int], float] = {}

    for row in read_csv_rows(good_path):
        sample_idx = int(row["sample_idx"])
        class_confidence_by_sample[(sample_idx, 0)] = float(row["predicted_confidence"])

    for row in read_csv_rows(bad_path):
        sample_idx = int(row["sample_idx"])
        class_confidence_by_sample[(sample_idx, 1)] = float(row["predicted_confidence"])

    return class_confidence_by_sample


def sample_idx_from_shap_path(path: Path) -> int:
    match = re.search(r"shap_tuples_non_prefix_(\d+)\.json$", path.name)
    if not match:
        raise ValueError(f"Unexpected SHAP file name: {path.name}")
    return int(match.group(1))


def load_shap_top3_by_sample(shap_dir: Path) -> dict[int, list[dict[str, Any]]]:
    shap_dir = resolve_path(shap_dir)
    if not shap_dir.exists():
        raise FileNotFoundError(f"SHAP directory does not exist: {shap_dir}")

    top3_by_sample: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(shap_dir.glob("shap_tuples_non_prefix_*.json")):
        sample_idx = sample_idx_from_shap_path(path)
        if sample_idx not in CONDITION_3_SAMPLE_INDICES:
            continue

        with path.open(encoding="utf-8") as f:
            payload = json.load(f)

        tuples = payload.get("tuples", [])
        top3_by_sample[sample_idx] = [
            {
                "feature": item["feature"],
                "value": round(float(item["shap_value"]), 2),
            }
            for item in tuples[:3]
        ]

    return top3_by_sample


def sample_idx_from_final_summary_path(path: Path) -> int:
    match = re.search(r"sample_(\d+)_final_summary(?:_ko)?\.json$", path.name)
    if not match:
        raise ValueError(f"Unexpected final summary file name: {path.name}")
    return int(match.group(1))


def load_rag_summary_payloads_by_sample(
    dirs: list[Path],
) -> dict[int, dict[str, Any]]:
    payloads_by_sample: dict[int, dict[str, Any]] = {}

    for directory in dirs:
        directory = resolve_path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"RAG final summary directory does not exist: {directory}")

        # Prefer Korean translated summaries when both files exist.
        paths = list(directory.glob("sample_*_final_summary_ko.json"))
        paths += [
            path
            for path in directory.glob("sample_*_final_summary.json")
            if path.with_name(path.stem + "_ko.json") not in paths
        ]

        for path in sorted(paths):
            sample_idx = sample_idx_from_final_summary_path(path)
            if sample_idx not in CONDITION_3_SAMPLE_INDICES:
                continue

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


def get_rag_explanation(rag_payload: dict[str, Any]) -> str | None:
    for key in ("final_explanation_ko", "summary_ko"):
        value = rag_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def get_rag_top3_features(rag_payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        "local_shap_top3_features",
        "shap_top3_features",
        "top3_features",
        "features",
        "tuples",
    ]

    for key in candidates:
        value = rag_payload.get(key)

        if isinstance(value, list):
            return value[:3]

    return []


def build_sample_payload(
    row: dict[str, str],
    confidence_by_sample: dict[int, dict[str, str]],
    class_confidence_by_sample: dict[tuple[int, int], float],
    shap_top3_by_sample: dict[int, list[dict[str, Any]]],
    rag_payloads_by_sample: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    sample_idx = int(row["sample_idx"])

    if sample_idx not in CONDITION_3_SAMPLE_INDICES:
        raise ValueError(f"sample_idx={sample_idx} is not a Condition3 sample.")

    if sample_idx not in confidence_by_sample:
        raise ValueError(f"Missing confidence.csv row for sample_idx={sample_idx}")

    true_class = int(row["true_class"])
    class_confidence_key = (sample_idx, true_class)

    if class_confidence_key not in class_confidence_by_sample:
        raise ValueError(
            f"Missing class-specific confidence row for sample_idx={sample_idx}, "
            f"true_class={true_class}"
        )

    if sample_idx not in shap_top3_by_sample:
        raise ValueError(f"Missing all-dataset SHAP JSON for sample_idx={sample_idx}")

    confidence_row = confidence_by_sample[sample_idx]
    rag_payload = rag_payloads_by_sample.get(sample_idx, {})

    for key in ("true_label", "predicted_label", "is_correct"):
        if str(row[key]) != str(confidence_row[key]):
            raise ValueError(
                f"Mismatch for sample_idx={sample_idx}, field={key}: "
                f"selected={row[key]}, confidence={confidence_row[key]}"
            )

    explanation = get_rag_explanation(rag_payload)
    local_shap_top3_features = shap_top3_by_sample[sample_idx]
    class_confidence_value = class_confidence_by_sample[class_confidence_key]

    payload = {
        "sample_idx": sample_idx,
        "condition": 3,
        "distance": distance_from_sigma_group(row["sigma_group"]),
        "is_correct": parse_bool(row["is_correct"]),
        "customer_data": build_customer_data(row),
        "predicted_label": display_predicted_label(row["predicted_label"]),
        "confidence": round(float(confidence_row["predicted_confidence"]) * 100),
        "true_label": display_predicted_label(row["true_label"]),
        "local_shap_top3_features": local_shap_top3_features,
        "explanation": explanation,
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

    condition3_rows = [
        row for row in selected_rows
        if int(row["sample_idx"]) in CONDITION_3_SAMPLE_INDICES
    ]

    if len(condition3_rows) != 40:
        raise ValueError(f"Expected 40 Condition3 rows, got {len(condition3_rows)}")

    confidence_by_sample = load_confidence_by_sample(args.confidence_input)

    class_confidence_by_sample = load_class_confidence_by_sample(
        good_path=args.confidence_good_input,
        bad_path=args.confidence_bad_input,
    )

    shap_top3_by_sample = load_shap_top3_by_sample(args.shap_dir)

    rag_payloads_by_sample = load_rag_summary_payloads_by_sample(
        [args.rag_correct_dir, args.rag_wrong_dir]
    )

    missing_rag_files = sorted(
        sample_idx
        for sample_idx in CONDITION_3_SAMPLE_INDICES
        if sample_idx not in rag_payloads_by_sample
    )

    if missing_rag_files:
        print(
            "WARNING: Missing RAG final summary JSON files for Condition3 sample_idx: "
            + ", ".join(map(str, missing_rag_files))
        )

    missing_korean_explanations = sorted(
        sample_idx
        for sample_idx in CONDITION_3_SAMPLE_INDICES
        if get_rag_explanation(rag_payloads_by_sample.get(sample_idx, {})) is None
    )

    if missing_korean_explanations:
        print(
            "WARNING: Missing Korean final explanation for Condition3 sample_idx: "
            + ", ".join(map(str, missing_korean_explanations))
        )

    missing_shap_files = sorted(
        sample_idx
        for sample_idx in CONDITION_3_SAMPLE_INDICES
        if sample_idx not in shap_top3_by_sample
    )

    if missing_shap_files:
        raise ValueError(
            "Missing all-dataset SHAP JSON files for Condition3 sample_idx: "
            + ", ".join(map(str, missing_shap_files))
        )

    selected_true_class_by_sample = {
        int(row["sample_idx"]): int(row["true_class"])
        for row in condition3_rows
    }
    missing_class_confidence = sorted(
        sample_idx
        for sample_idx, true_class in selected_true_class_by_sample.items()
        if (sample_idx, true_class) not in class_confidence_by_sample
    )

    if missing_class_confidence:
        raise ValueError(
            "Missing class-specific confidence_good/bad.csv rows for sample_idx: "
            + ", ".join(map(str, missing_class_confidence))
        )

    payloads = [
        build_sample_payload(
            row=row,
            confidence_by_sample=confidence_by_sample,
            class_confidence_by_sample=class_confidence_by_sample,
            shap_top3_by_sample=shap_top3_by_sample,
            rag_payloads_by_sample=rag_payloads_by_sample,
        )
        for row in condition3_rows
    ]

    write_json(args.output, payloads)

    if not args.no_individual:
        output_dir = resolve_path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for payload in payloads:
            write_json(output_dir / f"sample_{payload['sample_idx']}.json", payload)

    print(f"Selected Condition3 samples : {len(payloads)}")
    print(f"Combined output             : {resolve_path(args.output)}")

    if not args.no_individual:
        print(f"Individual output           : {resolve_path(args.output_dir)}")


if __name__ == "__main__":
    main()