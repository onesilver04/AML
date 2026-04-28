import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from shap_rf_non_prefix import (
    DROP_COLUMNS,
    evaluate_split_at_threshold,
    make_rf,
    resolve_output_path,
    select_sample_indices,
    tune_rf,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute test-set confidence summaries using "
            "predicted_confidence as the main confidence criterion."
        )
    )
    parser.add_argument("--data-path", default="german21_ohe.csv")
    parser.add_argument("--search-iter", type=int, default=32)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-indices", type=str, default=None)
    parser.add_argument("--sample-index-file", type=str, default=None)
    parser.add_argument("--threshold-min", type=float, default=0.20)
    parser.add_argument("--threshold-max", type=float, default=0.70)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--output-csv",
        default="SHAP/confidence.csv",
    )
    parser.add_argument(
        "--output-json",
        default="SHAP/test_confidence_summary_bad_positive.json",
    )
    return parser.parse_args()


def load_bad_positive_target(df: pd.DataFrame) -> pd.Series:
    if "class" not in df.columns:
        raise ValueError("타깃 컬럼 'class'를 찾지 못했습니다.")

    if df["class"].dtype == bool:
        return (~df["class"]).astype(int)

    lowered = pd.Series(df["class"]).dropna().astype(str).str.lower()
    if set(lowered.unique()).issubset({"good", "bad"}):
        return (df["class"].astype(str).str.lower() == "bad").astype(int)

    vals = set(pd.Series(df["class"]).dropna().unique())
    if vals.issubset({1, 2}):
        return (df["class"] == 2).astype(int)
    if vals.issubset({0, 1}):
        return df["class"].astype(int)

    raise ValueError(f"예상치 못한 class 값들: {vals}")


def to_bad_positive_label(value: int) -> str:
    return "BAD CREDIT RISK" if int(value) == 1 else "GOOD CREDIT RISK"


def build_threshold_grid(args):
    if args.threshold_step <= 0:
        raise ValueError("--threshold-step must be > 0.")
    if args.threshold_min >= args.threshold_max:
        raise ValueError("--threshold-min must be smaller than --threshold-max.")

    return np.round(
        np.arange(
            args.threshold_min,
            args.threshold_max + args.threshold_step / 2,
            args.threshold_step,
        ),
        4,
    )


def find_best_threshold(y_true, proba, threshold_grid):
    best_threshold_metrics = None

    for threshold in threshold_grid:
        metrics = evaluate_split_at_threshold(y_true, proba, threshold)

        score = (
            metrics["accuracy"],
            metrics["f1"],
            metrics["sensitivity"],
            -abs(threshold - 0.5),
        )

        if best_threshold_metrics is None or score > best_threshold_metrics["score"]:
            best_threshold_metrics = {
                "score": score,
                "threshold": threshold,
                "metrics": metrics,
            }

    return best_threshold_metrics


def build_confidence_dataframe(y_test, proba_bad, threshold, sample_indices=None):
    y_test_array = np.asarray(y_test, dtype=int)
    if sample_indices is None:
        sample_indices = np.arange(len(y_test_array))

    pred = (proba_bad >= threshold).astype(int)
    proba_good = 1.0 - proba_bad

    predicted_confidence = np.where(pred == 1, proba_bad, proba_good)

    df = pd.DataFrame(
        {
            "sample_idx": np.asarray(sample_indices, dtype=int),
            "true_class": y_test_array,
            "predicted_class": pred,
            "raw_bad_probability": proba_bad,
            "predicted_confidence": predicted_confidence,
        }
    )

    df["true_label"] = df["true_class"].map(to_bad_positive_label)
    df["predicted_label"] = df["predicted_class"].map(to_bad_positive_label)
    df["is_correct"] = df["true_class"] == df["predicted_class"]

    return df


def add_predicted_confidence_categories(confidence_df):
    mean_conf = confidence_df["predicted_confidence"].mean()
    std_conf = confidence_df["predicted_confidence"].std(ddof=0)
    mean_minus_1sigma = mean_conf - std_conf

    confidence_df["confidence_category"] = "middle_confidence"

    confidence_df.loc[
        confidence_df["predicted_confidence"] <= mean_minus_1sigma,
        "confidence_category",
    ] = "low_confidence"

    confidence_df.loc[
        confidence_df["predicted_confidence"] >= mean_conf,
        "confidence_category",
    ] = "high_confidence"

    return confidence_df, {
        "mean": float(mean_conf),
        "std": float(std_conf),
        "mean_minus_1sigma": float(mean_minus_1sigma),
    }


def add_class_based_confidence_categories(confidence_df):
    """
    Add confidence categories based on each class's own distribution.
    - predicted_class_confidence_category: based on predicted_class's confidence distribution
    
    Categories (2 levels):
    - 위험 (risk): confidence <= (class_mean - 1σ) - very low confidence
    - 주의 (caution): confidence > (class_mean - 1σ) - below average but not very low
    """
    # For predicted_class only
    for class_val in [0, 1]:
        mask = confidence_df["predicted_class"] == class_val
        subset = confidence_df.loc[mask, "predicted_confidence"]
        if not subset.empty:
            class_mean = subset.mean()
            class_std = subset.std(ddof=0)
            class_mean_minus_1sigma = class_mean - class_std

            confidence_df.loc[
                mask & (confidence_df["predicted_confidence"] <= class_mean_minus_1sigma),
                "predicted_class_confidence_category",
            ] = "위험"
            confidence_df.loc[
                mask & (confidence_df["predicted_confidence"] > class_mean_minus_1sigma),
                "predicted_class_confidence_category",
            ] = "주의"

    return confidence_df


def summarize_confidence_categories(confidence_df):
    categories = ["low_confidence", "middle_confidence", "high_confidence"]
    summary = []

    for category in categories:
        subset = confidence_df[confidence_df["confidence_category"] == category]
        misclassified = subset[subset["is_correct"] == False]

        summary.append(
            {
                "category": category,
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


def summarize_group(df, class_col, confidence_col):
    summary_rows = []

    for class_value in (0, 1):
        subset = df[df[class_col] == class_value]
        confidence_values = subset[confidence_col].dropna()

        if confidence_values.empty:
            continue

        summary_rows.append(
            {
                "class_value": class_value,
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

    df = pd.read_csv(args.data_path)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    y = load_bad_positive_target(df)
    X = df.drop(columns=["class"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    full_test_sample_count = len(X_test)

    if args.random_samples is not None or args.sample_indices or args.sample_index_file:
        sample_indices = select_sample_indices(args, full_test_sample_count)
    else:
        sample_indices = list(range(full_test_sample_count))

    tuned = tune_rf(X_train, y_train, args)

    raw_rf = make_rf(**tuned["params"])
    raw_rf.fit(X_train, y_train)

    raw_proba_test = raw_rf.predict_proba(X_test)[:, 1]

    raw_test_metrics = evaluate_split_at_threshold(
        y_test,
        raw_proba_test,
        tuned["threshold"],
    )

    y_test_selected = y_test.iloc[sample_indices]
    raw_proba_selected = raw_proba_test[sample_indices]

    raw_selected_metrics = evaluate_split_at_threshold(
        y_test_selected,
        raw_proba_selected,
        tuned["threshold"],
    )

    confidence_df = build_confidence_dataframe(
        y_test_selected,
        raw_proba_selected,
        tuned["threshold"],
        sample_indices=sample_indices,
    )

    confidence_df["raw_bad_probability"] = raw_proba_selected

    confidence_df, pred_conf_stats = add_predicted_confidence_categories(
        confidence_df
    )
    confidence_df = add_class_based_confidence_categories(confidence_df)
    category_summary = summarize_confidence_categories(confidence_df)

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
        "mean_predicted_confidence": pred_conf_stats["mean"],
        "std_predicted_confidence": pred_conf_stats["std"],
        "pred_confidence_mean_minus_1sigma": pred_conf_stats["mean_minus_1sigma"],
        "accuracy": float(confidence_df["is_correct"].mean()),
    }

    output_csv = resolve_output_path(args.output_csv)
    output_json = resolve_output_path(args.output_json)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    confidence_df.to_csv(output_csv, index=False)

    # Split by true_class and save separate CSV files
    good_df = confidence_df[confidence_df["true_class"] == 0]
    bad_df = confidence_df[confidence_df["true_class"] == 1]

    good_csv_path = output_csv.parent / "confidence_good.csv"
    bad_csv_path = output_csv.parent / "confidence_bad.csv"

    good_df.to_csv(good_csv_path, index=False)
    bad_df.to_csv(bad_csv_path, index=False)

    payload = {
        "data_path": args.data_path,
        "positive_class": {"value": 1, "label": "BAD CREDIT RISK"},
        "full_test_sample_count": int(full_test_sample_count),
        "tuned_params": tuned["params"],
        "cv_auc": float(tuned["cv_auc"]),
        "overall": overall_summary,
        "confidence_category_summary": category_summary,
        "raw_test_metrics": {
            "accuracy": float(raw_test_metrics["accuracy"]),
            "f1": float(raw_test_metrics["f1"]),
            "auc": float(raw_test_metrics["auc"]),
            "sensitivity": float(raw_test_metrics["sensitivity"]),
            "specificity": float(raw_test_metrics["specificity"]),
            "confusion": [int(v) for v in raw_test_metrics["confusion"]],
        },
        "raw_selected_metrics": {
            "accuracy": float(raw_selected_metrics["accuracy"]),
            "f1": float(raw_selected_metrics["f1"]),
            "auc": float(raw_selected_metrics["auc"]),
            "sensitivity": float(raw_selected_metrics["sensitivity"]),
            "specificity": float(raw_selected_metrics["specificity"]),
            "confusion": [int(v) for v in raw_selected_metrics["confusion"]],
        },
        "by_predicted_class": predicted_summary,
        "by_true_class": true_summary,
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("=== Predicted Confidence Summary ===")
    print("Positive class         : BAD CREDIT RISK (1)")

    print(
        "\nTest metrics: "
        f"acc={raw_test_metrics['accuracy']:.4f}, "
        f"f1={raw_test_metrics['f1']:.4f}, "
        f"auc={raw_test_metrics['auc']:.4f}"
    )

    print(f"\nTest sample count: {len(confidence_df)}")
    if len(confidence_df) != full_test_sample_count:
        print(f"Full test sample count: {full_test_sample_count}")
        print(f"Selected sample indices: {sample_indices}")

    print("\n=== Predicted confidence 기준 ===")
    print(f"Mean predicted_confidence        : {pred_conf_stats['mean']:.4f}")
    print(f"Std predicted_confidence         : {pred_conf_stats['std']:.4f}")
    print(f"Mean - 1σ threshold              : {pred_conf_stats['mean_minus_1sigma']:.4f}")

    print("\n=== Confidence category summary ===")
    for row in category_summary:
        print(
            f"{row['category']}: "
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

    print(f"\nSaved CSV : {output_csv}")
    print(f"Saved CSV (GOOD): {good_csv_path}")
    print(f"Saved CSV (BAD): {bad_csv_path}")
    print(f"Saved JSON: {output_json}")


if __name__ == "__main__":
    main()
