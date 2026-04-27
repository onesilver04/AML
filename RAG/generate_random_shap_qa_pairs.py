import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from RAG.make_rag_query import generate_rag_queries


DEFAULT_OUTPUT = Path("RAG/QA/random_120_local_shap_feature_qa_seed42.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate query-answer pairs from random local SHAP samples."
    )
    parser.add_argument(
        "--shap-dir",
        default="SHAP/120 Local Shap",
        help="Directory containing shap_tuples_non_prefix_{sample_idx}.json files.",
    )
    parser.add_argument(
        "--sample-index-file",
        default=None,
        help="File containing sample indices separated by commas or newlines.",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=None,
        help="Single sample index to process manually. Overrides random sampling.",
    )
    parser.add_argument(
        "--sample-indices",
        default=None,
        help="Comma-separated sample indices. Overrides random sampling.",
    )
    parser.add_argument(
        "--y-test",
        default="y_test.csv",
        help="Used to infer the test population size when sampling randomly.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-output", default=None)
    parser.add_argument(
        "--query-output-dir",
        default="RAG/QA/Queries",
        help="Directory where per-sample feature query lists are saved.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--min_docs", type=int, default=3)
    parser.add_argument("--max_docs", type=int, default=20)
    parser.add_argument("--min_gap", type=float, default=0.001)
    parser.add_argument("--max_chunks_per_source", type=int, default=2)
    parser.add_argument(
        "--evidence-sentences-per-source",
        type=int,
        default=1,
        help="Number of verbatim evidence sentences to keep from each selected source.",
    )
    return parser.parse_args()


def parse_index_text(text: str):
    return [
        int(part.strip())
        for part in text.replace("\n", ",").split(",")
        if part.strip()
    ]


def infer_population_size(y_test_path: Path) -> int:
    with y_test_path.open("r", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def select_sample_indices(args):
    if args.sample_idx is not None:
        sample_indices = [args.sample_idx]
    elif args.sample_indices:
        sample_indices = parse_index_text(args.sample_indices)
    elif args.sample_index_file:
        with Path(args.sample_index_file).open("r", encoding="utf-8") as f:
            sample_indices = parse_index_text(f.read())
    else:
        population_size = infer_population_size(Path(args.y_test))
        if args.count > population_size:
            raise ValueError(
                f"--count={args.count} exceeds available test samples "
                f"({population_size})."
            )
        rng = random.Random(args.seed)
        sample_indices = rng.sample(range(population_size), args.count)

    if args.sample_idx is None and args.count and len(sample_indices) != args.count:
        print(
            f"[WARN] selected {len(sample_indices)} indices, "
            f"but --count is {args.count}."
        )
    return sample_indices


def shap_json_path(shap_dir: Path, sample_idx: int) -> Path:
    return shap_dir / f"shap_tuples_non_prefix_{sample_idx}.json"


def load_shap_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing SHAP JSON: {path}. Run SHAP/shap_rf_non_prefix.py "
            f"with matching sample indices first."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def top_feature_payload(shap_data, top_k: int = 3):
    tuples = sorted(
        shap_data.get("tuples", []),
        key=lambda item: abs(float(item.get("abs_shap", item.get("shap_value", 0.0)))),
        reverse=True,
    )

    features = []
    for item in tuples[:top_k]:
        feature = item.get("feature", "UNKNOWN_FEATURE")
        direction = item.get("direction", "UNKNOWN")
        definition = item.get("definition")
        if not definition:
            raise ValueError(
                f"Missing definition for feature '{feature}'. "
                "Run the SHAP step that writes integrated JSON with definitions first."
            )
        features.append(
            {
                "feature": feature,
                "definition": definition,
                "direction": direction,
                "shap_value": float(item.get("shap_value", 0.0)),
                "abs_shap": float(item.get("abs_shap", abs(float(item.get("shap_value", 0.0))))),
            }
        )
    return features


def validate_integrated_shap_json(shap_data):
    for item in shap_data.get("tuples", []):
        feature = item.get("feature", "UNKNOWN_FEATURE")
        if not item.get("definition"):
            raise ValueError(
                f"Missing definition for feature '{feature}'. "
                "Run the SHAP step that writes integrated JSON with definitions first."
            )


def generated_queries_by_feature(shap_data, top_features):
    validate_integrated_shap_json(shap_data)
    generated_queries = generate_rag_queries(shap_data)

    if len(generated_queries) < len(top_features):
        raise ValueError(
            f"Generated {len(generated_queries)} queries for "
            f"{len(top_features)} top features."
        )

    queries_by_feature = {}
    for feature_info, query in zip(top_features, generated_queries):
        queries_by_feature[feature_info["feature"]] = query

    return generated_queries, queries_by_feature


def selected_sources_payload(selected_docs):
    sources = []
    for doc, score in selected_docs:
        sources.append(
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": page_display(doc.metadata.get("page", "NA")),
                "score": float(score),
            }
        )
    return sources


def tokenize_for_evidence(text: str):
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "associated",
        "be",
        "by",
        "credit",
        "default",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "probability",
        "risk",
        "statistical",
        "the",
        "to",
        "with",
    }
    return {
        token
        for token in re.findall(r"[A-Za-z0-9_<>/=+-]+", text.lower())
        if len(token) > 2 and token not in stopwords
    }


def split_sentences(text: str):
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'([])", normalized)
        if sentence.strip()
    ]


def evidence_sentences_payload(
    selected_docs,
    question: str,
    feature_info: dict,
    sentences_per_source: int = 1,
):
    if sentences_per_source < 1:
        return []

    query_terms = tokenize_for_evidence(
        " ".join(
            [
                question,
                feature_info.get("feature", ""),
                feature_info.get("definition", ""),
                feature_info.get("direction", ""),
            ]
        )
    )

    evidence_rows = []

    for doc, score in selected_docs:
        sentences = split_sentences(doc.page_content)
        if not sentences:
            continue

        ranked_sentences = sorted(
            sentences,
            key=lambda sentence: (
                len(tokenize_for_evidence(sentence) & query_terms),
                -len(sentence),
            ),
            reverse=True,
        )

        for sentence in ranked_sentences[:sentences_per_source]:
            evidence_rows.append(
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": page_display(doc.metadata.get("page", "NA")),
                    "score": float(score),
                    "sentence": sentence,
                }
            )

    return evidence_rows


def page_display(meta_page):
    if isinstance(meta_page, int):
        return meta_page + 1
    return meta_page


def summary_output_path(output_path: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    if output_path.suffix == ".jsonl":
        return output_path.with_name(f"{output_path.stem}_summary.csv")
    return output_path.with_suffix(".csv")


def resolve_output_path(args) -> Path:
    if args.output != str(DEFAULT_OUTPUT):
        return Path(args.output)
    if args.sample_idx is not None:
        return Path("RAG/QA") / f"sample_{args.sample_idx}_feature_qa.jsonl"
    return Path(args.output)


def uses_manual_sample_selection(args) -> bool:
    return (
        args.sample_idx is not None
        or args.sample_indices is not None
        or args.sample_index_file is not None
    )


def per_sample_output_path(sample_idx: int) -> Path:
    return Path("RAG/QA") / f"sample_{sample_idx}_feature_qa.jsonl"


def write_summary_csv(path: Path, records):
    fieldnames = [
        "sample_idx",
        "shap_json_path",
        "prediction_label",
        "prediction_probability",
        "feature_rank",
        "feature",
        "feature_definition",
        "feature_direction",
        "feature_shap_value",
        "feature_abs_shap",
        "question",
        "answer",
        "selected_sources",
        "evidence_sentences",
        "retrieval_params",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = record.copy()
            row["selected_sources"] = json.dumps(row["selected_sources"], ensure_ascii=False)
            row["evidence_sentences"] = json.dumps(
                row["evidence_sentences"],
                ensure_ascii=False,
            )
            row["retrieval_params"] = json.dumps(row["retrieval_params"], ensure_ascii=False)
            writer.writerow(row)


def write_sample_queries(path: Path, sample_idx: int, shap_path: Path, queries):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_idx": sample_idx,
        "shap_json_path": str(shap_path),
        "queries": queries,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    from score_gap_nomic import ask_rag_with_score_gap # 지연 임포트로 RAG 관련 의존성 최소화

    shap_dir = Path(args.shap_dir)
    output_path = resolve_output_path(args)
    query_output_dir = Path(args.query_output_dir)

    sample_indices = select_sample_indices(args)
    retrieval_params = {
        "min_docs": args.min_docs,
        "max_docs": args.max_docs,
        "min_gap": args.min_gap,
        "max_chunks_per_source": args.max_chunks_per_source,
        "inspect_all": False,
    }
    use_per_sample_outputs = (
        args.output == str(DEFAULT_OUTPUT) and uses_manual_sample_selection(args)
    )

    all_records = []

    def process_sample(sample_idx: int, sample_position: int, total_samples: int):
        path = shap_json_path(shap_dir, sample_idx)
        shap_data = load_shap_json(path)
        top_features = top_feature_payload(shap_data, top_k=3)
        pred = shap_data.get("prediction", {})
        sample_queries = []
        _, feature_queries = generated_queries_by_feature(
            shap_data,
            top_features,
        )
        sample_records = []

        for feature_rank, feature_info in enumerate(top_features, 1):
            question = feature_queries[feature_info["feature"]]
            sample_queries.append(
                {
                    "feature_rank": feature_rank,
                    "feature": feature_info["feature"],
                    "feature_definition": feature_info["definition"],
                    "feature_direction": feature_info["direction"],
                    "question": question,
                }
            )

            print(
                f"[sample {sample_position}/{total_samples} | "
                f"feature {feature_rank}/{len(top_features)}] "
                f"sample_idx={sample_idx} feature={feature_info['feature']}"
            )
            answer, selected_docs, _, _, _, _ = ask_rag_with_score_gap(
                question,
                min_docs=args.min_docs,
                max_docs=args.max_docs,
                min_gap=args.min_gap,
                max_chunks_per_source=args.max_chunks_per_source,
                inspect_all=False,
            )
            evidence_sentences = evidence_sentences_payload(
                selected_docs,
                question=question,
                feature_info=feature_info,
                sentences_per_source=args.evidence_sentences_per_source,
            )

            record = {
                "sample_idx": sample_idx,
                "shap_json_path": str(path),
                "prediction_label": pred.get("label", "UNKNOWN"),
                "prediction_probability": pred.get("probability", "UNKNOWN"),
                "feature_rank": feature_rank,
                "feature": feature_info["feature"],
                "feature_definition": feature_info["definition"],
                "feature_direction": feature_info["direction"],
                "feature_shap_value": feature_info["shap_value"],
                "feature_abs_shap": feature_info["abs_shap"],
                "question": question,
                "answer": answer,
                "selected_sources": selected_sources_payload(selected_docs),
                "evidence_sentences": evidence_sentences,
                "retrieval_params": retrieval_params,
            }
            sample_records.append(record)

        write_sample_queries(
            query_output_dir / f"sample_{sample_idx}_queries.json",
            sample_idx,
            path,
            sample_queries,
        )
        return sample_records

    if use_per_sample_outputs:
        for idx, sample_idx in enumerate(sample_indices, 1):
            sample_records = process_sample(sample_idx, idx, len(sample_indices))
            sample_output_path = per_sample_output_path(sample_idx)
            sample_summary_path = summary_output_path(sample_output_path, None)
            sample_output_path.parent.mkdir(parents=True, exist_ok=True)
            with sample_output_path.open("w", encoding="utf-8") as jsonl_file:
                for record in sample_records:
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            write_summary_csv(sample_summary_path, sample_records)
            all_records.extend(sample_records)
            print(f"\nSaved JSONL: {sample_output_path}")
            print(f"Saved CSV  : {sample_summary_path}")
    else:
        summary_path = summary_output_path(output_path, args.summary_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as jsonl_file:
            for idx, sample_idx in enumerate(sample_indices, 1):
                sample_records = process_sample(sample_idx, idx, len(sample_indices))
                all_records.extend(sample_records)
                for record in sample_records:
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    jsonl_file.flush()

        write_summary_csv(summary_path, all_records)
        print(f"\nSaved JSONL: {output_path}")
        print(f"Saved CSV  : {summary_path}")

    print(f"Saved per-sample queries to: {query_output_dir}")
    print(f"Total QA pairs: {len(all_records)}")


if __name__ == "__main__":
    main()
