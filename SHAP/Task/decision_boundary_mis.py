import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "SHAP/Task/Misclassified/misclassified_test_samples.csv"
DEFAULT_SIGMA_INPUT = PROJECT_ROOT / "SHAP/test_confidence_by_sample_bad_positive.csv"
DEFAULT_X_TEST = PROJECT_ROOT / "X_test.csv"
DEFAULT_WITHIN_OUTPUT = PROJECT_ROOT / "SHAP/Task/Misclassified/decision_boundary_within_1sigma.csv"
DEFAULT_OUTSIDE_OUTPUT = PROJECT_ROOT / "SHAP/Task/Misclassified/decision_boundary_outside_1sigma.csv"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "SHAP/Task/Misclassified/decision_boundary_summary.json"
DEFAULT_SELECTED_JSON_OUTPUT = (
    PROJECT_ROOT / "SHAP/Task/Misclassified/decision_boundary_selected_18.json"
)


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
            "file; --input defaults to the misclassified samples file."
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
    parser.add_argument(
        "--selected-json-output",
        type=Path,
        default=DEFAULT_SELECTED_JSON_OUTPUT,
    )
    parser.add_argument("--top-within-count", type=int, default=9)
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


def build_selected_samples_payload(
    outside: pd.DataFrame,
    within: pd.DataFrame,
    x_test: pd.DataFrame,
    args,
    sigma: float,
    lower_bound: float,
    upper_bound: float,
) -> dict:
    if args.top_within_count < 0:
        raise ValueError("--top-within-count must be >= 0.")

    outside_selected = outside.copy()
    outside_selected["selection_group"] = "outside_1sigma_all"

    within_selected = within.sort_values(
        "decision_boundary_abs_distance",
        ascending=False,
    ).head(args.top_within_count).copy()
    within_selected["selection_group"] = "within_1sigma_farthest"

    selected = pd.concat(
        [outside_selected, within_selected],
        ignore_index=True,
    ).sort_values(
        ["selection_group", "decision_boundary_abs_distance"],
        ascending=[True, False],
    )

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

    samples = []
    for _, row in selected.iterrows():
        sample_idx = int(row["sample_idx"])
        if sample_idx not in feature_lookup.index:
            raise ValueError(f"sample_idx not found in X_test: {sample_idx}")

        feature_values = row_to_clean_dict(feature_lookup.loc[sample_idx])
        prediction_probabilities = {
            col: clean_json_value(row[col])
            for col in probability_cols
            if col in selected.columns
        }

        samples.append(
            {
                "sample_idx": sample_idx,
                "selection_group": row["selection_group"],
                "sigma_group": row["decision_boundary_sigma_group"],
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
                "prediction_probabilities": prediction_probabilities,
                "original_features": feature_values,
            }
        )

    return {
        "input": str(resolve_path(args.input)),
        "sigma_input": str(resolve_path(args.sigma_input)),
        "x_test": str(resolve_path(args.x_test)),
        "decision_boundary": float(args.decision_boundary),
        "sigma": sigma,
        "sigma_multiplier": float(args.sigma_multiplier),
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "selection_rule": (
            "all outside-1sigma misclassified samples plus the farthest "
            f"{args.top_within_count} within-1sigma misclassified samples"
        ),
        "outside_selected_count": int(len(outside_selected)),
        "within_selected_count": int(len(within_selected)),
        "selected_count": int(len(samples)),
        "samples": samples,
    }


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
    selected_json_output = resolve_path(args.selected_json_output)

    within_output.parent.mkdir(parents=True, exist_ok=True)
    outside_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    selected_json_output.parent.mkdir(parents=True, exist_ok=True)

    within.to_csv(within_output, index=False)
    outside.to_csv(outside_output, index=False)

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

    selected_payload = build_selected_samples_payload(
        outside=outside,
        within=within,
        x_test=x_test,
        args=args,
        sigma=sigma,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    with selected_json_output.open("w", encoding="utf-8") as f:
        json.dump(selected_payload, f, ensure_ascii=False, indent=2)

    print(f"Decision boundary: {args.decision_boundary:.4f}")
    print(f"Sigma: {sigma:.4f}")
    print(f"Sigma source samples: {len(sigma_source)}")
    print(f"Split target samples: {len(result)}")
    print(f"Within range: {lower_bound:.4f} ~ {upper_bound:.4f}")
    print(f"Within +/- {args.sigma_multiplier:g} sigma: {len(within)}")
    print(f"Outside +/- {args.sigma_multiplier:g} sigma: {len(outside)}")
    print(f"Selected JSON samples: {selected_payload['selected_count']}")
    print(f"Saved within CSV : {within_output}")
    print(f"Saved outside CSV: {outside_output}")
    print(f"Saved summary    : {summary_output}")
    print(f"Saved selected   : {selected_json_output}")


if __name__ == "__main__":
    main()
