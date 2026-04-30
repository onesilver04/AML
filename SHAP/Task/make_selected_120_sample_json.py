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

FEATURE_PREFIX = "feature_"
SHAP_FILE_PATTERN = "shap_tuples_non_prefix_*.json"
CUSTOMER_FIELD_RENAMES = {
    "credit": "credit_amount",
    "Sex": "gender",
    "Married": "married",
    "existing": "existing_credits",
}
CONDITION_1_SAMPLE_INDICES = {
    123,
    142,
    176,
    45,
    81,
    83,
    197,
    187,
    100,
    10,
    177,
    126,
    136,
    146,
    169,
    189,
    194,
    144,
    14,
    24,
    47,
    182,
    172,
    93,
    120,
    127,
    135,
    178,
    179,
    48,
    63,
    16,
    124,
    78,
    13,
    39,
    77,
    110,
    111,
    183,
}


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
                    "feature": item["feature"],
                    "value": round(float(item["shap_value"]), 2),
                }
                for item in tuples[:3]
            ]

    return top3_by_sample


def sample_idx_from_rag_path(path: Path) -> int:
    match = re.search(r"sample_(\d+)_final_summary\.json$", path.name)
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


def build_customer_data(row: dict[str, str]) -> dict[str, Any]:
    customer_data: dict[str, Any] = {}
    for key, value in row.items():
        if not key.startswith(FEATURE_PREFIX):
            continue

        customer_key = key.removeprefix(FEATURE_PREFIX)
        customer_key = CUSTOMER_FIELD_RENAMES.get(customer_key, customer_key)
        parsed_value = parse_value(value)

        if customer_key == "gender":
            parsed_value = "male" if parsed_value is True else "female"

        customer_data[customer_key] = parsed_value

    return customer_data


def build_sample_payload(
    row: dict[str, str],
    confidence_by_sample: dict[int, dict[str, str]],
    shap_top3_by_sample: dict[int, list[dict[str, Any]]],
    rag_evidence_by_sample: dict[int, dict[str, dict[str, Any]]],
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
        evidence = rag_evidence_by_feature.get(shap_feature["feature"])
        if evidence:
            feature_payload["evidence_sentence"] = evidence
        local_shap_top3_features.append(feature_payload)

    payload = {
        "sample_idx": sample_idx,
    }
    if sample_idx in CONDITION_1_SAMPLE_INDICES:
        payload["condition"] = 1

    payload.update(
        {
            "distance": distance_from_sigma_group(row["sigma_group"]),
            "is_correct": parse_bool(row["is_correct"]),
            "customer_data": build_customer_data(row),
            "predicted_label": row["predicted_label"],
            "confidence": round(float(confidence_row["predicted_confidence"]) * 100),
            "true_label": row["true_label"],
            "local_shap_top3_features": local_shap_top3_features,
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

    payloads = [
        build_sample_payload(
            row,
            confidence_by_sample,
            shap_top3_by_sample,
            rag_evidence_by_sample,
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
