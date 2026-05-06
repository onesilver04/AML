import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read existing local SHAP JSON files and compute confidence summaries "
            "without retraining the model. Warning type is computed using "
            "the full test-set reference SHAP JSON files."
        )
    )
    parser.add_argument(
        "--shap-dir",
        default="SHAP/SHAP/Test Dataset Local Shap 25",
        help="Directory containing selected shap_tuples_non_prefix_*.json files.",
    )
    parser.add_argument(
        "--reference-shap-dir",
        default="SHAP/Test Dataset Local Shap",
        help="Directory containing full test-set shap_tuples_non_prefix_*.json files.",
    )
    parser.add_argument(
        "--output-csv",
        default="SHAP/confidence.csv",
    )
    parser.add_argument(
        "--output-json",
        default="SHAP/test_confidence_summary_bad_positive.json",
    )
    return parser.parse_args()


def resolve_output_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def label_to_class(label: str) -> int:
    label = str(label).strip().upper()

    if label == "BAD CREDIT RISK":
        return 1
    if label == "GOOD CREDIT RISK":
        return 0

    raise ValueError(f"Unknown label: {label}")


def to_bad_positive_label(value: int) -> str:
    return "BAD CREDIT RISK" if int(value) == 1 else "GOOD CREDIT RISK"


def extract_sample_idx_from_filename(path: Path) -> int:
    match = re.search(r"shap_tuples_non_prefix_(\d+)\.json$", path.name)

    if not match:
        raise ValueError(f"Cannot extract sample_idx from filename: {path.name}")

    return int(match.group(1))


def load_shap_confidence_records(shap_dir: Path) -> pd.DataFrame:
    json_paths = sorted(
        shap_dir.glob("shap_tuples_non_prefix_*.json"),
        key=extract_sample_idx_from_filename,
    )

    if not json_paths:
        raise FileNotFoundError(
            f"No shap_tuples_non_prefix_*.json files found in: {shap_dir}"
        )

    rows = []

    for path in json_paths:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        sample_idx = int(data.get("sample_idx", extract_sample_idx_from_filename(path)))

        true_label = data.get("true_label")
        if true_label is None:
            raise KeyError(f"{path} does not contain 'true_label'.")

        prediction = data.get("prediction", {})

        predicted_label = prediction.get("predict_label")
        if predicted_label is None:
            predicted_label = prediction.get("label")

        if predicted_label is None:
            raise KeyError(
                f"{path} does not contain prediction.predict_label or prediction.label."
            )

        confidence = prediction.get("calibrated_confidence")
        if confidence is None:
            confidence = prediction.get("predicted_confidence")

        if confidence is None:
            raise KeyError(
                f"{path} does not contain prediction.calibrated_confidence."
            )

        true_class = label_to_class(true_label)
        predicted_class = label_to_class(predicted_label)

        rows.append(
            {
                "sample_idx": sample_idx,
                "true_class": true_class,
                "predicted_class": predicted_class,
                "predicted_confidence": float(confidence),
                "true_label": to_bad_positive_label(true_class),
                "predicted_label": to_bad_positive_label(predicted_class),
                "is_correct": true_class == predicted_class,
                "source_json": str(path),
            }
        )

    confidence_df = pd.DataFrame(rows)
    confidence_df = confidence_df.sort_values("sample_idx").reset_index(drop=True)
    return confidence_df


def add_warning_type_by_reference(
    confidence_df: pd.DataFrame,
    reference_df: pd.DataFrame,
):
    """
    warning_type은 confidence_df 내부 120개가 아니라,
    reference_df 전체 테스트셋 기준 calibrated confidence 분포로 계산한다.

    predicted_class별 기준:
    - confidence > mean: none
    - mean - 1σ < confidence <= mean: weak_warning
    - confidence <= mean - 1σ: strong_warning
    """

    confidence_df = confidence_df.copy()
    confidence_df["warning_type"] = "none"
    class_stats = []

    for class_value in (0, 1):
        ref_mask = reference_df["predicted_class"] == class_value
        ref_subset = reference_df.loc[ref_mask, "predicted_confidence"]

        if ref_subset.empty:
            continue

        mean_conf = ref_subset.mean()
        std_conf = ref_subset.std(ddof=0)
        mean_minus_1sigma = mean_conf - std_conf

        target_mask = confidence_df["predicted_class"] == class_value

        confidence_df.loc[
            target_mask
            & (confidence_df["predicted_confidence"] <= mean_conf)
            & (confidence_df["predicted_confidence"] > mean_minus_1sigma),
            "warning_type",
        ] = "weak_warning"

        confidence_df.loc[
            target_mask
            & (confidence_df["predicted_confidence"] <= mean_minus_1sigma),
            "warning_type",
        ] = "strong_warning"

        class_stats.append(
            {
                "class_value": int(class_value),
                "class_label": to_bad_positive_label(class_value),
                "reference_sample_count": int(len(ref_subset)),
                "reference_mean": float(mean_conf),
                "reference_std": float(std_conf),
                "reference_mean_minus_1sigma": float(mean_minus_1sigma),
            }
        )

    return confidence_df, class_stats


def summarize_warning_types(confidence_df: pd.DataFrame):
    warning_types = ["strong_warning", "weak_warning", "none"]
    summary = []

    for warning_type in warning_types:
        subset = confidence_df[confidence_df["warning_type"] == warning_type]
        misclassified = subset[subset["is_correct"] == False]

        summary.append(
            {
                "warning_type": warning_type,
                "sample_count": int(len(subset)),
                "misclassified_count": int(len(misclassified)),
                "misclassified_sample_indices": [
                    int(idx) for idx in misclassified["sample_idx"].tolist()
                ],
                "mean_confidence": (
                    float(subset["predicted_confidence"].mean())
                    if not subset.empty
                    else None
                ),
                "min_confidence": (
                    float(subset["predicted_confidence"].min())
                    if not subset.empty
                    else None
                ),
                "max_confidence": (
                    float(subset["predicted_confidence"].max())
                    if not subset.empty
                    else None
                ),
            }
        )

    return summary


def summarize_group(df: pd.DataFrame, class_col: str, confidence_col: str):
    summary_rows = []

    for class_value in (0, 1):
        subset = df[df[class_col] == class_value]
        confidence_values = subset[confidence_col].dropna()

        if confidence_values.empty:
            continue

        summary_rows.append(
            {
                "class_value": int(class_value),
                "class_label": to_bad_positive_label(class_value),
                "sample_count": int(len(confidence_values)),
                "mean_confidence": float(confidence_values.mean()),
                "median_confidence": float(confidence_values.median()),
                "std_confidence": float(confidence_values.std(ddof=0)),
                "min_confidence": float(confidence_values.min()),
                "max_confidence": float(confidence_values.max()),
            }
        )

    return summary_rows


def main():
    args = parse_args()

    shap_dir = resolve_output_path(args.shap_dir)
    reference_shap_dir = resolve_output_path(args.reference_shap_dir)

    output_csv = resolve_output_path(args.output_csv)
    output_json = resolve_output_path(args.output_json)

    confidence_df = load_shap_confidence_records(shap_dir)
    reference_df = load_shap_confidence_records(reference_shap_dir)

    confidence_df, warning_class_stats = add_warning_type_by_reference(
        confidence_df=confidence_df,
        reference_df=reference_df,
    )

    warning_summary = summarize_warning_types(confidence_df)

    predicted_summary = summarize_group(
        confidence_df,
        class_col="predicted_class",
        confidence_col="predicted_confidence",
    )

    true_summary = summarize_group(
        confidence_df,
        class_col="true_class",
        confidence_col="predicted_confidence",
    )

    overall_summary = {
        "sample_count": int(len(confidence_df)),
        "reference_sample_count": int(len(reference_df)),
        "mean_predicted_confidence": float(
            confidence_df["predicted_confidence"].mean()
        ),
        "std_predicted_confidence": float(
            confidence_df["predicted_confidence"].std(ddof=0)
        ),
        "accuracy": float(confidence_df["is_correct"].mean()),
    }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    confidence_df.to_csv(output_csv, index=False)

    good_df = confidence_df[confidence_df["true_class"] == 0]
    bad_df = confidence_df[confidence_df["true_class"] == 1]

    good_csv_path = output_csv.parent / "confidence_good.csv"
    bad_csv_path = output_csv.parent / "confidence_bad.csv"

    good_df.to_csv(good_csv_path, index=False)
    bad_df.to_csv(bad_csv_path, index=False)

    payload = {
        "input_shap_dir": str(shap_dir),
        "reference_shap_dir": str(reference_shap_dir),
        "positive_class": {"value": 1, "label": "BAD CREDIT RISK"},
        "warning_reference": (
            "warning_type is computed using calibrated_confidence distribution "
            "from reference_shap_dir, grouped by predicted_class."
        ),
        "overall": overall_summary,
        "warning_type_class_stats": warning_class_stats,
        "warning_type_summary": warning_summary,
        "by_predicted_class": predicted_summary,
        "by_true_class": true_summary,
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("=== Confidence Summary from SHAP JSON files ===")
    print(f"Input SHAP dir     : {shap_dir}")
    print(f"Reference SHAP dir : {reference_shap_dir}")
    print(f"Loaded input samples     : {len(confidence_df)}")
    print(f"Loaded reference samples : {len(reference_df)}")

    print("\n=== Reference calibrated confidence 기준 ===")
    for row in warning_class_stats:
        print(
            f"{row['class_label']}: "
            f"reference_n={row['reference_sample_count']}, "
            f"mean={row['reference_mean']:.4f}, "
            f"std={row['reference_std']:.4f}, "
            f"mean-1σ={row['reference_mean_minus_1sigma']:.4f}"
        )

    print("\n=== Warning type summary ===")
    for row in warning_summary:
        print(
            f"{row['warning_type']}: "
            f"n={row['sample_count']}, "
            f"misclassified={row['misclassified_count']}, "
            f"misclassified_sample_indices={row['misclassified_sample_indices']}"
        )

    print("\nBy predicted class:")
    for row in predicted_summary:
        print(
            f"{row['class_label']}: "
            f"n={row['sample_count']}, "
            f"mean={row['mean_confidence']:.4f}, "
            f"median={row['median_confidence']:.4f}, "
            f"std={row['std_confidence']:.4f}, "
            f"min={row['min_confidence']:.4f}, "
            f"max={row['max_confidence']:.4f}"
        )

    print("\nBy true class 참고용:")
    for row in true_summary:
        print(
            f"{row['class_label']}: "
            f"n={row['sample_count']}, "
            f"mean={row['mean_confidence']:.4f}, "
            f"median={row['median_confidence']:.4f}, "
            f"std={row['std_confidence']:.4f}, "
            f"min={row['min_confidence']:.4f}, "
            f"max={row['max_confidence']:.4f}"
        )

    print(f"\nSaved CSV       : {output_csv}")
    print(f"Saved CSV GOOD  : {good_csv_path}")
    print(f"Saved CSV BAD   : {bad_csv_path}")
    print(f"Saved JSON      : {output_json}")


if __name__ == "__main__":
    main()