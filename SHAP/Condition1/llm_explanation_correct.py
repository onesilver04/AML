# boundary에서 먼 거 17, 가까운거 17
# condition 1
import argparse
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path


N_SAMPLES = 17
DEFAULT_MODEL = "qwen3.6:35b"

SHAP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SHAP_DIR.parent
CLASSIFIED_DIR = SHAP_DIR / "Task" / "Classified"
LOCAL_SHAP_DIR = SHAP_DIR / "Task" / "correct_102_local_shap"
CORRECT_FAR_CSV = CLASSIFIED_DIR / "correct_far.csv"
CORRECT_NEAR_CSV = CLASSIFIED_DIR / "correct_near.csv"
RAG_SUMMARIZER_PATH = REPO_ROOT / "RAG" / "Final Summary" / "summarize_rag_feature_answers.py"
DEFAULT_OUTPUT_DIR = SHAP_DIR / "Condition1" / "Results" / "correct_local_shap_explanations"


SHAP_SUMMARY_SYSTEM_PROMPT = """You are a financial risk analyst specialized in credit scoring and SHAP-based explanations.

Your job is to generate a user-friendly Korean explanation for credit risk predictions.

You must transform technical financial expressions into intuitive, easy-to-understand Korean language for general users.

You are NOT a translator.
You must not directly translate encoded feature names, numeric thresholds, or technical financial expressions.
Instead, rewrite them into natural Korean phrases that users can understand at a glance.

STRICT RULES:
1. Output ONLY valid JSON.
2. Do NOT include any text outside JSON.
3. Do NOT expose reasoning steps.
4. Generate ONLY final_explanation.
5. The final explanation must be 1 to 3 Korean sentences.
6. The final explanation must include all three features in the same order as provided.
7. Do NOT add citation markers, reference numbers, or feature numbers in final_explanation.
8. Use the provided SHAP JSON information as much as possible.
9. Do NOT mention missing evidence or RAG.
"""


SHAP_SUMMARY_USER_TEMPLATE = """Generate only the final Korean explanation for this credit risk prediction.

Prediction:
- true label: {true_label}
- predicted label: {prediction_label}
- predicted bad-risk probability: {prediction_probability}
- decision threshold: {prediction_threshold}
- decision boundary group: {decision_boundary_group}
- decision boundary absolute distance: {decision_boundary_abs_distance}

Top SHAP features:
{feature_context}

Instructions:

Step 1: Interpret the input for writing the final explanation.
- Identify the prediction label, probability, and threshold.
- Identify whether this sample is far from or near the decision boundary.
- Identify the three SHAP features.
- Understand each feature's meaning, SHAP direction, and SHAP value.
- Use this interpretation only to write final_explanation.

Step 2: Convert technical feature expressions into user-friendly Korean for final_explanation.
- Do NOT keep raw encoded feature names in final_explanation.
- Do NOT include raw numeric thresholds if they can be rewritten more naturally.
- Rewrite technical expressions into intuitive Korean phrases.

Examples:
- "savings_status_<100" or "저축 잔액 100 DM 미만" -> "저축 잔액이 거의 없는 상태"
- "checking_status_no checking" -> "사용 중인 당좌예금 계좌가 없는 상태"
- "credit_amount" -> "대출 금액"
- "duration" -> "대출 기간"
- "employment_unemployed" -> "현재 안정적인 고용 상태가 아닌 점"

Step 3: Determine each feature's impact for final_explanation.
- If shap_direction is "모델에서는 위험을 높이는 방향", describe the feature as increasing risk.
- If shap_direction is "모델에서는 위험을 낮추는 방향", describe the feature as decreasing risk.
- Use natural Korean, not literal translation.

Step 4: Generate final_explanation.
- Write 1 to 3 Korean sentences.
- Mention all three features in the same order as provided.
- Do NOT add citation markers, reference numbers, or feature numbers in final_explanation.
- Make the explanation flow naturally.
- End by reflecting the final prediction label naturally.
- You may mention whether the sample is close to or far from the decision boundary if it helps users understand confidence.

Output format:
{{
  "final_explanation": "..."
}}
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample correct far/near rows, find matching local SHAP JSON files, "
            "and summarize them into Korean user-facing explanations."
        )
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES, help=f"Samples per group. Default: {N_SAMPLES}")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling. Default: 42")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama chat model. Default: {DEFAULT_MODEL}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"JSON output directory. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--csv-output", type=Path, default=None, help="Optional CSV summary output path.")
    return parser.parse_args()


def sample_rows(csv_path: Path, group: str, n_samples: int = N_SAMPLES, seed: int | None = None) -> list[dict]:
    """Randomly select rows and keep metadata needed in the final output."""
    with csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows or "sample_idx" not in rows[0]:
        raise ValueError(f"{csv_path} does not contain a 'sample_idx' column.")

    if len(rows) < n_samples:
        raise ValueError(f"{csv_path} has only {len(rows)} rows; cannot sample {n_samples} rows.")

    rng = random.Random(seed)
    sampled_rows = rng.sample(rows, n_samples)

    samples = []
    for row in sampled_rows:
        samples.append(
            {
                "sample_idx": int(row["sample_idx"]),
                "decision_boundary_group": group,
                "sigma_group": row.get("sigma_group", "unknown"),
                "decision_boundary_abs_distance": row.get("decision_boundary_abs_distance", "unknown"),
            }
        )

    return samples


def get_random_correct_samples(n_samples: int = N_SAMPLES, seed: int | None = None) -> dict[str, list[dict]]:
    """Return random sample metadata from correct_far and correct_near each."""
    return {
        "far": sample_rows(CORRECT_FAR_CSV, "far", n_samples=n_samples, seed=seed),
        "near": sample_rows(CORRECT_NEAR_CSV, "near", n_samples=n_samples, seed=seed),
    }


def load_rag_summarizer():
    if not RAG_SUMMARIZER_PATH.exists():
        raise FileNotFoundError(f"RAG summarizer does not exist: {RAG_SUMMARIZER_PATH}")

    spec = importlib.util.spec_from_file_location("rag_final_summarizer", RAG_SUMMARIZER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import RAG summarizer: {RAG_SUMMARIZER_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shap_json_path(sample_idx: int) -> Path:
    return LOCAL_SHAP_DIR / f"shap_tuples_non_prefix_{sample_idx}.json"


def load_shap_payload(sample_idx: int):
    path = shap_json_path(sample_idx)

    if not path.exists():
        raise FileNotFoundError(f"No local SHAP JSON found for sample_idx={sample_idx}: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f), path


def direction_label(direction: str):
    if direction == "increase_risk":
        return "모델에서는 위험을 높이는 방향"
    if direction == "decrease_risk":
        return "모델에서는 위험을 낮추는 방향"
    return "모델 방향 미확인"


def validate_shap_payload(payload):
    tuples = payload.get("tuples") or []

    if len(tuples) != 3:
        raise ValueError(
            f"sample_idx={payload.get('sample_idx')} must contain exactly 3 SHAP tuples, "
            f"but found {len(tuples)}."
        )

    return tuples


def format_feature_context(tuples):
    blocks = []

    for item in tuples:
        blocks.append(
            "\n".join(
                [
                    f"- feature: {item.get('feature', 'unknown')}",
                    f"- definition: {item.get('definition', 'unknown')}",
                    f"- shap_direction: {direction_label(item.get('direction', 'UNKNOWN'))}",
                    f"- shap_value: {item.get('shap_value', 'unknown')}",
                    f"- absolute_shap_value: {item.get('abs_shap', 'unknown')}",
                ]
            )
        )

    return "\n\n".join(blocks)


def validate_llm_output(payload):
    if not isinstance(payload, dict):
        raise ValueError("LLM output must be a JSON object.")

    final_explanation = payload.get("final_explanation")
    if not isinstance(final_explanation, str) or not final_explanation.strip():
        raise ValueError("LLM output must contain a non-empty final_explanation.")

    return payload


def summarize_shap_payload(payload, sample_meta, llm, rag_summarizer):
    try:
        from langchain_core.prompts import ChatPromptTemplate
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'langchain_core'. Install it with: pip install langchain-core"
        ) from exc

    tuples = validate_shap_payload(payload)
    prediction = payload.get("prediction") or {}
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SHAP_SUMMARY_SYSTEM_PROMPT),
            ("user", SHAP_SUMMARY_USER_TEMPLATE),
        ]
    )
    chain = prompt | llm

    result = chain.invoke(
        {
            "true_label": payload.get("true_label", "UNKNOWN"),
            "prediction_label": prediction.get("label", "UNKNOWN"),
            "prediction_probability": prediction.get("probability", "UNKNOWN"),
            "prediction_threshold": prediction.get("threshold", "UNKNOWN"),
            "decision_boundary_group": sample_meta.get("decision_boundary_group", "unknown"),
            "decision_boundary_abs_distance": sample_meta.get("decision_boundary_abs_distance", "unknown"),
            "feature_context": format_feature_context(tuples),
        }
    )

    llm_payload = rag_summarizer.parse_llm_json(result.content)
    validate_llm_output(llm_payload)
    return llm_payload


def build_shap_features(tuples):
    features = []

    for rank, item in enumerate(tuples, 1):
        features.append(
            {
                "rank": rank,
                "feature": item.get("feature", "unknown"),
                "definition": item.get("definition", "unknown"),
                "shap_value": item.get("shap_value", "unknown"),
                "abs_shap": item.get("abs_shap", "unknown"),
                "direction": item.get("direction", "UNKNOWN"),
            }
        )

    return features


def build_output_payload(sample_meta: dict, shap_path: Path, shap_payload, llm_payload):
    prediction = shap_payload.get("prediction") or {}
    tuples = validate_shap_payload(shap_payload)

    return {
        "sample_idx": shap_payload.get("sample_idx"),
        "decision_boundary_group": sample_meta.get("decision_boundary_group", "unknown"),
        "sigma_group": sample_meta.get("sigma_group", "unknown"),
        "decision_boundary_abs_distance": sample_meta.get("decision_boundary_abs_distance", "unknown"),
        "source_shap_json": str(shap_path),
        "true_label": shap_payload.get("true_label", "UNKNOWN"),
        "prediction_label": prediction.get("label", "UNKNOWN"),
        "prediction_probability": prediction.get("probability", "UNKNOWN"),
        "prediction_threshold": prediction.get("threshold", "UNKNOWN"),
        "final_explanation": llm_payload["final_explanation"],
        "shap_features": build_shap_features(tuples),
    }


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_csv(path: Path, payloads):
    fieldnames = [
        "sample_idx",
        "decision_boundary_group",
        "sigma_group",
        "decision_boundary_abs_distance",
        "source_shap_json",
        "true_label",
        "prediction_label",
        "prediction_probability",
        "prediction_threshold",
        "final_explanation",
        "shap_features",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for payload in payloads:
            row = payload.copy()
            row["shap_features"] = json.dumps(row.get("shap_features", []), ensure_ascii=False)
            writer.writerow(row)


def main():
    args = parse_args()
    rag_summarizer = load_rag_summarizer()
    llm = rag_summarizer.build_llm(args.model)
    selected_samples = get_random_correct_samples(n_samples=args.n_samples, seed=args.seed)
    payloads = []

    for group, samples in selected_samples.items():
        print(f"{group} sample_idx: {[sample['sample_idx'] for sample in samples]}")

        for sample_meta in samples:
            sample_idx = sample_meta["sample_idx"]
            shap_payload, shap_path = load_shap_payload(sample_idx)
            llm_payload = summarize_shap_payload(shap_payload, sample_meta, llm, rag_summarizer)
            output_payload = build_output_payload(sample_meta, shap_path, shap_payload, llm_payload)
            output_path = args.output_dir / group / f"sample_{sample_idx}_local_shap_summary.json"

            write_json(output_path, output_payload)
            payloads.append(output_payload)
            print(f"Saved JSON: {output_path}")

    csv_output = args.csv_output or args.output_dir / "local_shap_summaries.csv"
    write_csv(csv_output, payloads)

    print(f"Saved CSV : {csv_output}")
    print(f"Total summaries: {len(payloads)}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ImportError, ModuleNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
