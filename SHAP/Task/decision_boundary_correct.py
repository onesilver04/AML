# 정상 분류된 샘플 중에서 decision boundary 기준 거리 구분

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = "SHAP/Task/Classified/correct_classified_test_samples.csv"
DEFAULT_SIGMA_INPUT = "SHAP/test_confidence_by_sample_bad_positive.csv"
DEFAULT_X_TEST = PROJECT_ROOT / "X_test.csv"
DEFAULT_WITHIN_OUTPUT = PROJECT_ROOT / "SHAP/Task/Classified/decision_boundary_within_1sigma.csv"
DEFAULT_OUTSIDE_OUTPUT = PROJECT_ROOT / "SHAP/Task/Classified/decision_boundary_outside_1sigma.csv"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "SHAP/Task/Classified/decision_boundary_summary.json"
DEFAULT_CLOSE_JSON_OUTPUT = PROJECT_ROOT / "SHAP/Task/Classified/decision_boundary_closest_51.json"
DEFAULT_FAR_JSON_OUTPUT = PROJECT_ROOT / "SHAP/Task/Classified/decision_boundary_farthest_51.json"
DEFAULT_CLOSE_CSV_OUTPUT = PROJECT_ROOT / "SHAP/Task/Classified/decision_boundary_closest_51.csv"
DEFAULT_FAR_CSV_OUTPUT = PROJECT_ROOT / "SHAP/Task/Classified/decision_boundary_farthest_51.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split test samples by whether raw_bad_probability falls within "
            "decision_boundary +/- sigma."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--sigma-input",
        type=Path,
        default=DEFAULT_SIGMA_INPUT,
        help=(
            "CSV used to calculate sigma. Defaults to the full test confidence "
            "file; --input defaults to the correctly classified samples file."
        ),
    )
    parser.add_argument("--x-test", type=Path, default=DEFAULT_X_TEST)
    parser.add_argument("--decision-boundary", type=float, default=0.56)
    parser.add_argument(
        "--sigma-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to the raw_bad_probability standard deviation.",
    )
    parser.add_argument("--within-output", type=Path, default=DEFAULT_WITHIN_OUTPUT)
    parser.add_argument("--outside-output", type=Path, default=DEFAULT_OUTSIDE_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--close-json-output", type=Path, default=DEFAULT_CLOSE_JSON_OUTPUT)
    parser.add_argument("--far-json-output", type=Path, default=DEFAULT_FAR_JSON_OUTPUT)
    parser.add_argument("--close-csv-output", type=Path, default=DEFAULT_CLOSE_CSV_OUTPUT)
    parser.add_argument("--far-csv-output", type=Path, default=DEFAULT_FAR_CSV_OUTPUT)
    parser.add_argument("--selection-count", type=int, default=51)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_predictions(path: Path) -> pd.DataFrame:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    df = pd.read_csv(path)
    required_cols = {"sample_idx", "raw_bad_probability"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            "input file is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )
    return df


def load_x_test(path: Path) -> pd.DataFrame:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"X_test file does not exist: {path}")

    x_test = pd.read_csv(path)
    x_test = x_test.copy()
    x_test.insert(0, "sample_idx", range(len(x_test)))
    return x_test


def clean_json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def row_to_clean_dict(row: pd.Series) -> dict:
    return {key: clean_json_value(value) for key, value in row.items()}


def build_sample_records(samples_df: pd.DataFrame, x_test: pd.DataFrame) -> list[dict]:
    feature_lookup = x_test.set_index("sample_idx")
    probability_cols = [
        "raw_bad_probability",
        "raw_good_probability",
        "predicted_bad_probability",
        "predicted_good_probability",
        "predicted_confidence",
        "true_class_confidence",
        "predicted_threshold_relative_confidence",
        "true_class_threshold_relative_confidence",
    ]

    records = []
    for _, row in samples_df.iterrows():
        sample_idx = int(row["sample_idx"])
        if sample_idx not in feature_lookup.index:
            raise ValueError(f"sample_idx not found in X_test: {sample_idx}")

        probability_values = {
            col: clean_json_value(row[col])
            for col in probability_cols
            if col in samples_df.columns
        }

        records.append(
            {
                "sample_idx": sample_idx,
                "selection_group": clean_json_value(row["selection_group"]),
                "sigma_group": clean_json_value(row["decision_boundary_sigma_group"]),
                "decision_boundary_distance": clean_json_value(
                    row["decision_boundary_distance"]
                ),
                "decision_boundary_abs_distance": clean_json_value(
                    row["decision_boundary_abs_distance"]
                ),
                "true_class": clean_json_value(row.get("true_class")),
                "true_label": clean_json_value(row.get("true_label")),
                "predicted_class": clean_json_value(row.get("predicted_class")),
                "predicted_label": clean_json_value(row.get("predicted_label")),
                "is_correct": clean_json_value(row.get("is_correct")),
                "confidence_category": clean_json_value(
                    row.get("confidence_category")
                ),
                "prediction_probabilities": probability_values,
                "original_features": row_to_clean_dict(feature_lookup.loc[sample_idx]),
            }
        )

    return records


def flatten_sample_records(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {}
        for key, value in record.items():
            if key == "prediction_probabilities":
                row.update(
                    {
                        f"probability_{prob_key}": prob_value
                        for prob_key, prob_value in value.items()
                    }
                )
            elif key == "original_features":
                row.update(
                    {
                        f"feature_{feature_key}": feature_value
                        for feature_key, feature_value in value.items()
                    }
                )
            else:
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def write_selection_outputs(
    records: list[dict],
    output_json: Path,
    output_csv: Path,
    metadata: dict,
):
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        **metadata,
        "selected_count": int(len(records)),
        "samples": records,
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    flatten_sample_records(records).to_csv(output_csv, index=False)


def summarize_split(name: str, df: pd.DataFrame) -> dict:
    summary = {
        "group": name,
        "sample_count": int(len(df)),
        "sample_indices": [int(idx) for idx in df["sample_idx"].tolist()],
    }

    if "is_correct" in df.columns:
        misclassified = df[df["is_correct"] == False]
        summary["misclassified_count"] = int(len(misclassified))
        summary["misclassified_sample_indices"] = [
            int(idx) for idx in misclassified["sample_idx"].tolist()
        ]

    return summary


def main():
    args = parse_args()
    if args.sigma_multiplier <= 0:
        raise ValueError("--sigma-multiplier must be > 0.")
    if args.selection_count < 1:
        raise ValueError("--selection-count must be >= 1.")

    predictions = load_predictions(args.input)
    sigma_source = load_predictions(args.sigma_input)
    x_test = load_x_test(args.x_test)

    proba = sigma_source["raw_bad_probability"]
    sigma = float(proba.std(ddof=0))
    margin = sigma * args.sigma_multiplier
    lower_bound = args.decision_boundary - margin
    upper_bound = args.decision_boundary + margin

    result = predictions.copy()
    result["decision_boundary"] = float(args.decision_boundary)
    result["raw_bad_probability_sigma"] = sigma
    result["decision_boundary_lower_bound"] = lower_bound
    result["decision_boundary_upper_bound"] = upper_bound
    result["decision_boundary_distance"] = (
        result["raw_bad_probability"] - args.decision_boundary
    )
    result["decision_boundary_abs_distance"] = result[
        "decision_boundary_distance"
    ].abs()
    sigma_label = f"{args.sigma_multiplier:g}".replace(".", "_")
    within_group = f"within_{sigma_label}sigma"
    outside_group = f"outside_{sigma_label}sigma"

    result["decision_boundary_sigma_group"] = outside_group

    within_mask = result["raw_bad_probability"].between(
        lower_bound,
        upper_bound,
        inclusive="both",
    )
    result.loc[within_mask, "decision_boundary_sigma_group"] = within_group

    within = result[within_mask].copy()
    outside = result[~within_mask].copy()

    within_output = resolve_path(args.within_output)
    outside_output = resolve_path(args.outside_output)
    summary_output = resolve_path(args.summary_output)
    close_json_output = resolve_path(args.close_json_output)
    far_json_output = resolve_path(args.far_json_output)
    close_csv_output = resolve_path(args.close_csv_output)
    far_csv_output = resolve_path(args.far_csv_output)

    within_output.parent.mkdir(parents=True, exist_ok=True)
    outside_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    within.to_csv(within_output, index=False)
    outside.to_csv(outside_output, index=False)

    closest = result.sort_values(
        "decision_boundary_abs_distance",
        ascending=True,
    ).head(args.selection_count).copy()
    closest["selection_group"] = "closest_to_decision_boundary"

    farthest = result.sort_values(
        "decision_boundary_abs_distance",
        ascending=False,
    ).head(args.selection_count).copy()
    farthest["selection_group"] = "farthest_from_decision_boundary"

    selection_metadata = {
        "input": str(resolve_path(args.input)),
        "sigma_input": str(resolve_path(args.sigma_input)),
        "x_test": str(resolve_path(args.x_test)),
        "decision_boundary": float(args.decision_boundary),
        "sigma_source_column": "raw_bad_probability",
        "sigma_ddof": 0,
        "sigma_multiplier": float(args.sigma_multiplier),
        "sigma": sigma,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "selection_count": int(args.selection_count),
        "total_sample_count": int(len(result)),
    }

    closest_records = build_sample_records(closest, x_test)
    farthest_records = build_sample_records(farthest, x_test)

    write_selection_outputs(
        closest_records,
        close_json_output,
        close_csv_output,
        {
            **selection_metadata,
            "selection_rule": (
                f"closest {args.selection_count} correctly classified samples "
                "to the decision boundary"
            ),
        },
    )
    write_selection_outputs(
        farthest_records,
        far_json_output,
        far_csv_output,
        {
            **selection_metadata,
            "selection_rule": (
                f"farthest {args.selection_count} correctly classified samples "
                "from the decision boundary"
            ),
        },
    )

    summary = {
        "input": str(resolve_path(args.input)),
        "sigma_input": str(resolve_path(args.sigma_input)),
        "decision_boundary": float(args.decision_boundary),
        "sigma_source_column": "raw_bad_probability",
        "sigma_ddof": 0,
        "sigma_multiplier": float(args.sigma_multiplier),
        "sigma": sigma,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "total_sample_count": int(len(result)),
        "sigma_source_sample_count": int(len(sigma_source)),
        within_group: summarize_split(within_group, within),
        outside_group: summarize_split(outside_group, outside),
    }

    with summary_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Decision boundary: {args.decision_boundary:.4f}")
    print(f"Sigma: {sigma:.4f}")
    print(f"Sigma source samples: {len(sigma_source)}")
    print(f"Split target samples: {len(result)}")
    print(f"Within range: {lower_bound:.4f} ~ {upper_bound:.4f}")
    print(f"Within +/- {args.sigma_multiplier:g} sigma: {len(within)}")
    print(f"Outside +/- {args.sigma_multiplier:g} sigma: {len(outside)}")
    print(f"Closest selected: {len(closest_records)}")
    print(f"Farthest selected: {len(farthest_records)}")
    print(f"Saved within CSV : {within_output}")
    print(f"Saved outside CSV: {outside_output}")
    print(f"Saved summary    : {summary_output}")
    print(f"Saved closest JSON: {close_json_output}")
    print(f"Saved closest CSV : {close_csv_output}")
    print(f"Saved farthest JSON: {far_json_output}")
    print(f"Saved farthest CSV : {far_csv_output}")


if __name__ == "__main__":
    main()
