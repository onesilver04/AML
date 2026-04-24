import argparse
import csv
import json
import re
import sys
from pathlib import Path


DEFAULT_ANSWERS_DIR = Path("RAG/QA/Answers")
DEFAULT_Y_TEST_PATH = Path("y_test.csv")
DEFAULT_OUTPUT_PATH = Path("RAG/QA/Answers/misclassified_answer_samples.csv")


CSV_FIELDNAMES = [
    "sample_idx",
    "true_label",
    "prediction_label",
    "prediction_probability",
    "is_misclassified",
    "top_features",
    "feature_directions",
    "feature_shap_values",
    "file_path",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find misclassified RAG answer samples by comparing prediction_label with y_test.csv."
    )
    parser.add_argument(
        "--answers-dir",
        type=Path,
        default=DEFAULT_ANSWERS_DIR,
        help=f"Directory containing sample_*_feature_qa.jsonl files. Default: {DEFAULT_ANSWERS_DIR}",
    )
    parser.add_argument(
        "--y-test",
        type=Path,
        default=DEFAULT_Y_TEST_PATH,
        help=f"CSV file containing the true class column. Default: {DEFAULT_Y_TEST_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"CSV output path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--include-correct",
        action="store_true",
        help="Include correctly classified samples in the CSV output.",
    )
    return parser.parse_args()


def to_risk_label(value: int) -> str:
    # y_test.csv stores the original boolean class as 1=True(good), 0=False(bad).
    return "GOOD CREDIT RISK" if int(value) == 1 else "BAD CREDIT RISK"


def load_y_test(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"y_test file does not exist: {path}")

    labels = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "class" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain a 'class' column.")
        for row_number, row in enumerate(reader, 2):
            try:
                labels.append(int(row["class"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid class value at {path}:{row_number}: {row.get('class')!r}"
                ) from exc
    return labels


def sample_idx_from_path(path: Path) -> int:
    match = re.fullmatch(r"sample_(\d+)_feature_qa\.jsonl", path.name)
    if not match:
        raise ValueError(f"Unexpected answer file name: {path}")
    return int(match.group(1))


def load_answer_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc

    if not records:
        raise ValueError(f"{path} does not contain any JSONL records.")
    return records


def validate_records(path: Path, sample_idx: int, records):
    record_sample_indices = {record.get("sample_idx") for record in records}
    if record_sample_indices != {sample_idx}:
        raise ValueError(
            f"{path} contains inconsistent sample_idx values: {sorted(record_sample_indices)} "
            f"(expected {sample_idx})."
        )

    prediction_labels = {record.get("prediction_label") for record in records}
    if None in prediction_labels or "" in prediction_labels:
        raise ValueError(f"{path} has a record missing prediction_label.")
    if len(prediction_labels) != 1:
        raise ValueError(f"{path} contains inconsistent prediction_label values: {prediction_labels}")


def summarize_answer_file(path: Path, y_true):
    sample_idx = sample_idx_from_path(path)
    if sample_idx >= len(y_true):
        raise IndexError(
            f"sample_idx {sample_idx} from {path} is out of range for y_test "
            f"with {len(y_true)} rows."
        )

    records = sorted(
        load_answer_records(path),
        key=lambda record: int(record.get("feature_rank", 0)),
    )
    validate_records(path, sample_idx, records)

    first = records[0]
    true_label = to_risk_label(y_true[sample_idx])
    prediction_label = first["prediction_label"]
    top_features = [str(record.get("feature", "UNKNOWN")) for record in records]
    feature_directions = [str(record.get("feature_direction", "UNKNOWN")) for record in records]
    feature_shap_values = [
        str(record.get("feature_shap_value", "UNKNOWN")) for record in records
    ]

    return {
        "sample_idx": sample_idx,
        "true_label": true_label,
        "prediction_label": prediction_label,
        "prediction_probability": first.get("prediction_probability", "UNKNOWN"),
        "is_misclassified": true_label != prediction_label,
        "top_features": " | ".join(top_features),
        "feature_directions": " | ".join(feature_directions),
        "feature_shap_values": " | ".join(feature_shap_values),
        "file_path": str(path),
    }


def iter_answer_files(answers_dir: Path):
    if not answers_dir.exists():
        raise FileNotFoundError(f"Answers directory does not exist: {answers_dir}")
    files = sorted(answers_dir.glob("sample_*_feature_qa.jsonl"))
    if not files:
        raise FileNotFoundError(f"No sample_*_feature_qa.jsonl files found in: {answers_dir}")
    return files


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(all_rows):
    misclassified = [row for row in all_rows if row["is_misclassified"]]
    correct_count = len(all_rows) - len(misclassified)
    misclassified_indices = [str(row["sample_idx"]) for row in misclassified]

    print(f"Total samples: {len(all_rows)}")
    print(f"Misclassified: {len(misclassified)}")
    print(f"Correct: {correct_count}")
    print("Misclassified sample_idx:")
    print(", ".join(misclassified_indices) if misclassified_indices else "(none)")


def main():
    args = parse_args()
    y_true = load_y_test(args.y_test)

    all_rows = [
        summarize_answer_file(path, y_true)
        for path in iter_answer_files(args.answers_dir)
    ]
    output_rows = all_rows if args.include_correct else [
        row for row in all_rows if row["is_misclassified"]
    ]

    write_csv(args.output, output_rows)
    print_summary(all_rows)
    print(f"Saved CSV: {args.output}")
    if not args.include_correct:
        print("CSV contains only misclassified samples. Use --include-correct to save all samples.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, IndexError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
