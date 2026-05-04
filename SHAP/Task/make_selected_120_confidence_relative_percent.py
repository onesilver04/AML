import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORRECT_INPUT = PROJECT_ROOT / "SHAP/Task/correct_102.csv"
DEFAULT_WRONG_INPUT = PROJECT_ROOT / "SHAP/Task/wrong_18.csv"
DEFAULT_X_TEST = PROJECT_ROOT / "X_test.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "SHAP/Task/selected_120_confidence_relative_percent.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create confidence percentile positions for the selected 120 samples, "
            "grouped by true class."
        )
    )
    parser.add_argument("--correct-input", type=Path, default=DEFAULT_CORRECT_INPUT)
    parser.add_argument("--wrong-input", type=Path, default=DEFAULT_WRONG_INPUT)
    parser.add_argument(
        "--x-test",
        type=Path,
        default=DEFAULT_X_TEST,
        help=(
            "X_test.csv saved by shap_rf_non_prefix.py. Its columns are the RF "
            "training features used to build the prefixed feature columns."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_selected_samples(path: Path) -> pd.DataFrame:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    df = pd.read_csv(path)
    required_cols = {
        "sample_idx",
        "true_class",
        "true_label",
        "predicted_class",
        "predicted_label",
        "is_correct",
        "probability_raw_bad_probability",
        "probability_raw_good_probability",
        "sigma_group",
        "decision_boundary_abs_distance",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{path} is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    df = df.copy()
    return df


def add_predicted_confidence(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    predicted_class = df["predicted_class"].astype(str)

    df["predicted_confidence"] = pd.NA
    bad_mask = predicted_class == "1"
    good_mask = predicted_class == "0"

    df.loc[bad_mask, "predicted_confidence"] = df.loc[
        bad_mask, "probability_raw_bad_probability"
    ]
    df.loc[good_mask, "predicted_confidence"] = df.loc[
        good_mask, "probability_raw_good_probability"
    ]

    if df["predicted_confidence"].isna().any():
        bad_values = df.loc[df["predicted_confidence"].isna(), "predicted_class"]
        raise ValueError(
            "Unexpected predicted_class values: "
            + ", ".join(sorted(bad_values.astype(str).unique()))
        )

    df["predicted_confidence"] = df["predicted_confidence"].astype(float)
    return df


def percentile_rank_0_to_100(values: pd.Series) -> pd.Series:
    if len(values) == 1:
        return pd.Series([100.0], index=values.index)

    ranks = values.rank(method="average", ascending=True) - 1
    return ranks / (len(values) - 1) * 100.0


def ohe_prefix(name: str) -> str:
    if "_" in name:
        return name.rsplit("_", 1)[0]
    return name


def is_true_value(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def add_prefixed_features(df: pd.DataFrame, x_test_path: Path) -> tuple[pd.DataFrame, list[str]]:
    x_test_path = resolve_path(x_test_path)
    if not x_test_path.exists():
        raise FileNotFoundError(f"X_test file does not exist: {x_test_path}")

    x_test = pd.read_csv(x_test_path)
    x_test = x_test.copy()
    x_test.insert(0, "sample_idx", range(len(x_test)))

    prefix_to_features: dict[str, list[str]] = {}
    for feature in x_test.columns:
        if feature == "sample_idx":
            continue
        prefix_to_features.setdefault(ohe_prefix(feature), []).append(feature)

    feature_rows = []
    for _, row in x_test.iterrows():
        prefixed = {"sample_idx": int(row["sample_idx"])}
        for prefix, features in prefix_to_features.items():
            col_name = f"feature_{prefix}"
            if len(features) == 1:
                prefixed[col_name] = row[features[0]]
                continue

            active_values = []
            for feature in features:
                if is_true_value(row[feature]):
                    active_values.append(feature[len(prefix) + 1 :])
            prefixed[col_name] = " | ".join(active_values)
        feature_rows.append(prefixed)

    prefixed_features = pd.DataFrame(feature_rows)
    feature_cols = [col for col in prefixed_features.columns if col != "sample_idx"]

    df = df.drop(columns=[col for col in feature_cols if col in df.columns])
    merged = df.merge(prefixed_features, on="sample_idx", how="left")
    if merged[feature_cols].isna().any(axis=None):
        missing = merged.loc[merged[feature_cols].isna().any(axis=1), "sample_idx"]
        raise ValueError(
            "Some selected sample_idx values were not found in X_test: "
            + ", ".join(missing.astype(str).tolist())
        )

    return merged, feature_cols


def add_class_relative_confidence(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["class_confidence_relative_percent"] = (
        df.groupby("true_class", group_keys=False)["predicted_confidence"]
        .apply(percentile_rank_0_to_100)
        .astype(float)
    )
    df["class_confidence_group_size"] = df.groupby("true_class")[
        "sample_idx"
    ].transform("size")
    return df


def main():
    args = parse_args()

    selected = pd.concat(
        [
            load_selected_samples(args.correct_input, "correct_102"),
            load_selected_samples(args.wrong_input, "wrong_18"),
        ],
        ignore_index=True,
    )

    selected = add_predicted_confidence(selected)
    selected = add_class_relative_confidence(selected)
    selected, prefixed_feature_cols = add_prefixed_features(selected, args.x_test)

    output_cols = [
        "sample_idx",
        "true_class",
        "true_label",
        "predicted_class",
        "predicted_label",
        "is_correct",
        "probability_raw_bad_probability",
        "probability_raw_good_probability",
        "predicted_confidence",
        "class_confidence_relative_percent",
        "class_confidence_group_size",
        "sigma_group",
        "decision_boundary_abs_distance",
    ] + prefixed_feature_cols

    output_df = selected[output_cols].copy()
    output_df["predicted_confidence"] = output_df["predicted_confidence"].map(
        "{:.16g}".format
    )
    output_df["class_confidence_relative_percent"] = output_df[
        "class_confidence_relative_percent"
    ].map("{:.6f}".format)

    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output, index=False)

    print(f"Selected samples: {len(selected)}")
    for true_class, group in selected.groupby("true_class", sort=True):
        labels = ", ".join(sorted(group["true_label"].astype(str).unique()))
        min_percent = group["class_confidence_relative_percent"].min()
        max_percent = group["class_confidence_relative_percent"].max()
        print(
            f"true_class={true_class} ({labels}): "
            f"count={len(group)}, range={min_percent:.6f}..{max_percent:.6f}"
        )
    print(f"Saved CSV: {output}")


if __name__ == "__main__":
    main()
