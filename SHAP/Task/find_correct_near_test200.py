import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "SHAP/confidence.csv"
DEFAULT_EXISTING_JSON_DIR = PROJECT_ROOT / "SHAP/120_samples_individual_json"
DEFAULT_OUTPUT = PROJECT_ROOT / "SHAP/Task/Classified/correct_near_missing_from_120.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find test-set samples where is_correct is True and distance is near. "
            "If distance is not already present, near is calculated as "
            "raw_bad_probability within decision_boundary +/- sigma."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--existing-json-dir",
        type=Path,
        default=DEFAULT_EXISTING_JSON_DIR,
        help="Directory containing existing sample_*.json files.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--decision-boundary", type=float, default=0.56)
    parser.add_argument("--sigma-multiplier", type=float, default=1.0)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(["true", "1", "yes", "y"])


def load_existing_sample_indices(json_dir: Path) -> set[int]:
    json_dir = resolve_path(json_dir)
    if not json_dir.exists():
        raise FileNotFoundError(f"Existing JSON directory does not exist: {json_dir}")

    sample_indices = set()
    for path in json_dir.glob("sample_*.json"):
        try:
            sample_indices.add(int(path.stem.removeprefix("sample_")))
        except ValueError:
            continue
    return sample_indices


def add_distance_if_needed(
    df: pd.DataFrame,
    decision_boundary: float,
    sigma_multiplier: float,
) -> pd.DataFrame:
    result = df.copy()

    if "distance" in result.columns:
        result["distance"] = result["distance"].astype(str).str.strip().str.lower()
        return result

    group_col = None
    for candidate in ("decision_boundary_sigma_group", "sigma_group"):
        if candidate in result.columns:
            group_col = candidate
            break

    if group_col:
        result["distance"] = (
            result[group_col].astype(str).str.strip().str.lower().str.startswith("within_")
        ).map({True: "near", False: "far"})
        return result

    if "raw_bad_probability" not in result.columns:
        raise ValueError(
            "input must contain either distance, sigma_group, "
            "decision_boundary_sigma_group, or raw_bad_probability."
        )

    sigma = float(result["raw_bad_probability"].std(ddof=0))
    margin = sigma * sigma_multiplier
    lower_bound = decision_boundary - margin
    upper_bound = decision_boundary + margin

    result["decision_boundary"] = float(decision_boundary)
    result["raw_bad_probability_sigma"] = sigma
    result["decision_boundary_lower_bound"] = lower_bound
    result["decision_boundary_upper_bound"] = upper_bound
    result["decision_boundary_distance"] = (
        result["raw_bad_probability"] - decision_boundary
    )
    result["decision_boundary_abs_distance"] = result[
        "decision_boundary_distance"
    ].abs()

    near_mask = result["raw_bad_probability"].between(
        lower_bound,
        upper_bound,
        inclusive="both",
    )
    result["distance"] = near_mask.map({True: "near", False: "far"})
    return result


def main():
    args = parse_args()
    if args.sigma_multiplier <= 0:
        raise ValueError("--sigma-multiplier must be > 0.")

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    existing_json_dir = resolve_path(args.existing_json_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    df = pd.read_csv(input_path)
    required_cols = {"sample_idx", "is_correct"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            "input file is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    df = add_distance_if_needed(
        df,
        decision_boundary=args.decision_boundary,
        sigma_multiplier=args.sigma_multiplier,
    )

    correct_near = df[
        parse_bool_series(df["is_correct"]) & (df["distance"] == "near")
    ].copy()

    if "decision_boundary_abs_distance" in correct_near.columns:
        correct_near = correct_near.sort_values(
            ["decision_boundary_abs_distance", "sample_idx"],
            ascending=[True, True],
        )
    else:
        correct_near = correct_near.sort_values("sample_idx")

    existing_sample_indices = load_existing_sample_indices(existing_json_dir)
    missing_from_120 = correct_near[
        ~correct_near["sample_idx"].astype(int).isin(existing_sample_indices)
    ].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing_from_120.to_csv(output_path, index=False)

    sample_indices = correct_near["sample_idx"].astype(int).tolist()
    missing_sample_indices = missing_from_120["sample_idx"].astype(int).tolist()
    print(f"Input samples: {len(df)}")
    print(f"Correct + near samples: {len(correct_near)}")
    print(f"Existing JSON samples: {len(existing_sample_indices)}")
    print(f"Correct + near sample indices: {sample_indices}")
    print(f"Missing from 120 JSON sample count: {len(missing_from_120)}")
    print(f"Missing from 120 JSON sample indices: {missing_sample_indices}")
    print(f"Saved CSV: {output_path}")


if __name__ == "__main__":
    main()
