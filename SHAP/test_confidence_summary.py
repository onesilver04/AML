import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from shap_rf_non_prefix import (
    DROP_COLUMNS,
    PROJECT_ROOT,
    evaluate_split_at_threshold,
    load_target,
    make_rf,
    resolve_output_path,
    to_risk_label,
    tune_rf,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute test-set confidence summaries using the same split and RF "
            "training logic as SHAP/shap_rf_non_prefix.py."
        )
    )
    parser.add_argument("--data-path", default="german21_ohe.csv")
    parser.add_argument(
        "--search-iter",
        type=int,
        default=32,
        help="Number of RF hyperparameter candidates evaluated during tuning.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds used for hyperparameter search and threshold tuning.",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.20,
        help="Minimum threshold value searched on out-of-fold train probabilities.",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.70,
        help="Maximum threshold value searched on out-of-fold train probabilities.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Threshold step size.",
    )
    parser.add_argument(
        "--output-csv",
        default="SHAP/test_confidence_by_sample.csv",
        help="CSV path for per-sample test confidence values.",
    )
    parser.add_argument(
        "--output-json",
        default="SHAP/test_confidence_summary.json",
        help="JSON path for aggregated confidence summaries.",
    )
    return parser.parse_args()


def build_confidence_dataframe(y_test, proba_bad, threshold):
    pred = (proba_bad >= threshold).astype(int)
    proba_good = 1.0 - proba_bad
    predicted_confidence = np.where(pred == 1, proba_bad, proba_good)
    true_class_confidence = np.where(np.asarray(y_test) == 1, proba_bad, proba_good)
    predicted_threshold_relative_confidence = np.where(
        pred == 1,
        np.clip((proba_bad - threshold) / (1.0 - threshold), 0.0, 1.0),
        np.clip((threshold - proba_bad) / threshold, 0.0, 1.0),
    )
    true_class_threshold_relative_confidence = np.where(
        np.asarray(y_test) == 1,
        np.clip((proba_bad - threshold) / (1.0 - threshold), 0.0, 1.0),
        np.clip((threshold - proba_bad) / threshold, 0.0, 1.0),
    )

    df = pd.DataFrame(
        {
            "sample_idx": np.arange(len(y_test)),
            "true_class": np.asarray(y_test, dtype=int),
            "predicted_class": pred,
            "bad_probability": proba_bad,
            "good_probability": proba_good,
            "predicted_confidence": predicted_confidence,
            "true_class_confidence": true_class_confidence,
            "predicted_threshold_relative_confidence": predicted_threshold_relative_confidence,
            "true_class_threshold_relative_confidence": true_class_threshold_relative_confidence,
        }
    )
    df["true_label"] = df["true_class"].map(to_risk_label)
    df["predicted_label"] = df["predicted_class"].map(to_risk_label)
    df["is_correct"] = df["true_class"] == df["predicted_class"]
    return df


def summarize_group(df, class_col, confidence_col):
    summary_rows = []
    for class_value in (0, 1):
        subset = df[df[class_col] == class_value]
        if subset.empty:
            continue
        summary_rows.append(
            {
                "class_value": class_value,
                "class_label": to_risk_label(class_value),
                "sample_count": int(len(subset)),
                "mean_confidence": float(subset[confidence_col].mean()),
                "median_confidence": float(subset[confidence_col].median()),
                "std_confidence": float(subset[confidence_col].std(ddof=0)),
                "min_confidence": float(subset[confidence_col].min()),
                "max_confidence": float(subset[confidence_col].max()),
            }
        )
    return summary_rows


def main():
    args = parse_args()

    df = pd.read_csv(args.data_path)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    y = load_target(df)
    X = df.drop(columns=["class"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    full_test_sample_count = len(X_test)

    tuned = tune_rf(X_train, y_train, args)
    rf = make_rf(**tuned["params"])
    rf.fit(X_train, y_train)

    proba_bad = rf.predict_proba(X_test)[:, 1]
    full_test_metrics = evaluate_split_at_threshold(
        y_test,
        proba_bad,
        tuned["threshold"],
    )
    confidence_df = build_confidence_dataframe(y_test, proba_bad, tuned["threshold"])

    predicted_summary = summarize_group(
        confidence_df,
        class_col="predicted_class",
        confidence_col="predicted_confidence",
    )
    predicted_threshold_relative_summary = summarize_group(
        confidence_df,
        class_col="predicted_class",
        confidence_col="predicted_threshold_relative_confidence",
    )
    true_summary = summarize_group(
        confidence_df,
        class_col="true_class",
        confidence_col="true_class_confidence",
    )
    true_threshold_relative_summary = summarize_group(
        confidence_df,
        class_col="true_class",
        confidence_col="true_class_threshold_relative_confidence",
    )

    overall_summary = {
        "sample_count": int(len(confidence_df)),
        "mean_predicted_confidence": float(confidence_df["predicted_confidence"].mean()),
        "mean_true_class_confidence": float(confidence_df["true_class_confidence"].mean()),
        "mean_predicted_threshold_relative_confidence": float(
            confidence_df["predicted_threshold_relative_confidence"].mean()
        ),
        "mean_true_class_threshold_relative_confidence": float(
            confidence_df["true_class_threshold_relative_confidence"].mean()
        ),
        "accuracy": float(confidence_df["is_correct"].mean()),
        "threshold": float(tuned["threshold"]),
    }

    output_csv = resolve_output_path(args.output_csv)
    output_json = resolve_output_path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    confidence_df.to_csv(output_csv, index=False)

    payload = {
        "data_path": args.data_path,
        "full_test_sample_count": int(full_test_sample_count),
        "selected_test_sample_count": int(len(confidence_df)),
        "tuned_params": tuned["params"],
        "cv_auc": float(tuned["cv_auc"]),
        "threshold": float(tuned["threshold"]),
        "full_test_metrics": {
            "accuracy": float(full_test_metrics["accuracy"]),
            "f1": float(full_test_metrics["f1"]),
            "auc": float(full_test_metrics["auc"]),
            "sensitivity": float(full_test_metrics["sensitivity"]),
            "specificity": float(full_test_metrics["specificity"]),
            "confusion": [int(v) for v in full_test_metrics["confusion"]],
        },
        "overall": overall_summary,
        "by_predicted_class": predicted_summary,
        "by_predicted_class_threshold_relative": predicted_threshold_relative_summary,
        "by_true_class": true_summary,
        "by_true_class_threshold_relative": true_threshold_relative_summary,
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("=== Test Confidence Summary ===")
    print(f"Threshold              : {tuned['threshold']:.2f}")
    print(
        "Full test metrics      : "
        f"acc={full_test_metrics['accuracy']:.4f}, "
        f"f1={full_test_metrics['f1']:.4f}, "
        f"auc={full_test_metrics['auc']:.4f}"
    )
    print(f"Test sample count      : {len(confidence_df)}")
    print(
        "Overall mean confidence: "
        f"predicted={overall_summary['mean_predicted_confidence']:.4f}, "
        f"true_class={overall_summary['mean_true_class_confidence']:.4f}"
    )
    print(
        "Overall threshold-relative confidence: "
        f"predicted={overall_summary['mean_predicted_threshold_relative_confidence']:.4f}, "
        f"true_class={overall_summary['mean_true_class_threshold_relative_confidence']:.4f}"
    )

    print("\nBy predicted class:")
    for row in predicted_summary:
        print(
            f"{row['class_label']}: n={row['sample_count']}, "
            f"mean={row['mean_confidence']:.4f}, median={row['median_confidence']:.4f}, "
            f"min={row['min_confidence']:.4f}, max={row['max_confidence']:.4f}"
        )

    print("\nBy predicted class (threshold-relative):")
    for row in predicted_threshold_relative_summary:
        print(
            f"{row['class_label']}: n={row['sample_count']}, "
            f"mean={row['mean_confidence']:.4f}, median={row['median_confidence']:.4f}, "
            f"min={row['min_confidence']:.4f}, max={row['max_confidence']:.4f}"
        )

    print("\nBy true class:")
    for row in true_summary:
        print(
            f"{row['class_label']}: n={row['sample_count']}, "
            f"mean={row['mean_confidence']:.4f}, median={row['median_confidence']:.4f}, "
            f"min={row['min_confidence']:.4f}, max={row['max_confidence']:.4f}"
        )

    print("\nBy true class (threshold-relative):")
    for row in true_threshold_relative_summary:
        print(
            f"{row['class_label']}: n={row['sample_count']}, "
            f"mean={row['mean_confidence']:.4f}, median={row['median_confidence']:.4f}, "
            f"min={row['min_confidence']:.4f}, max={row['max_confidence']:.4f}"
        )

    print(f"\nSaved CSV : {output_csv}")
    print(f"Saved JSON: {output_json}")


if __name__ == "__main__":
    main()
