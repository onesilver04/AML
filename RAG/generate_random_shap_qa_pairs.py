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


DEFAULT_SHAP_DIR = Path("SHAP/Test Dataset Local Shap 25")
DEFAULT_OUTPUT = Path("RAG/QA/true_102_feature_qa.jsonl")
DEFAULT_QUERY_OUTPUT_DIR = Path("RAG/QA/Queries/250 Queries")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate query-answer pairs from local SHAP samples."
    )
    parser.add_argument(
        "--shap-dir",
        default=str(DEFAULT_SHAP_DIR),
        help="Directory containing shap_tuples_non_prefix_{sample_idx}.json files.",
    )
    parser.add_argument(
        "--input-json",
        type=str,
        required=False,
        help="Single JSONL file to process"
    )
    parser.add_argument(
        "--sample-index-file",
        default=None,
        help="File containing sample indices separated by commas or newlines.",
    )
    parser.add_argument(
        "--sample-indices-from-shap-dir",
        action="store_true",
        help=(
            "Use every shap_tuples_non_prefix_{sample_idx}.json file in --shap-dir "
            "as the sample list. This is now the default when no explicit sample "
            "selection is provided."
        ),
    )
    parser.add_argument(
        "--random-samples",
        action="store_true",
        help="Use the older random sampling behavior based on --y-test and --count.",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=None,
        help="Single sample index to process manually. Overrides random sampling.",
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        help="Comma-separated sample indices"
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
        default=str(DEFAULT_QUERY_OUTPUT_DIR),
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
    elif args.sample_indices_from_shap_dir or not args.random_samples:
        sample_indices = sample_indices_from_shap_dir(Path(args.shap_dir))
    else:
        population_size = infer_population_size(Path(args.y_test))
        if args.count > population_size:
            raise ValueError(
                f"--count={args.count} exceeds available test samples "
                f"({population_size})."
            )
        rng = random.Random(args.seed)
        sample_indices = rng.sample(range(population_size), args.count)

    if (
        args.sample_idx is None
        and args.random_samples
        and args.count
        and len(sample_indices) != args.count
    ):
        print(
            f"[WARN] selected {len(sample_indices)} indices, "
            f"but --count is {args.count}."
        )
    return sample_indices


def sample_indices_from_shap_dir(shap_dir: Path):
    if not shap_dir.exists():
        raise FileNotFoundError(f"SHAP directory does not exist: {shap_dir}")

    sample_indices = []
    pattern = re.compile(r"shap_tuples_non_prefix_(\d+)\.json$")
    for path in shap_dir.glob("shap_tuples_non_prefix_*.json"):
        match = pattern.match(path.name)
        if match:
            sample_indices.append(int(match.group(1)))

    if not sample_indices:
        raise FileNotFoundError(
            f"No shap_tuples_non_prefix_*.json files found in: {shap_dir}"
        )

    return sorted(sample_indices)


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
    
def load_jsonl_file(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records    


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


def resolve_output_path(args, sample_idx=None) -> Path:
    if args.output != str(DEFAULT_OUTPUT):
        output_path = Path(args.output)

        # 여러 sample을 돌릴 때 output이 디렉토리처럼 쓰이도록 처리
        if args.sample_indices or args.sample_index_file:
            if output_path.suffix:
                return output_path.parent / f"sample_{sample_idx}_feature_qa.jsonl"
            return output_path / f"sample_{sample_idx}_feature_qa.jsonl"

        return output_path

    if sample_idx is not None:
        return Path("RAG/QA/Summary") / f"sample_{sample_idx}_feature_qa.jsonl"

    if args.input_json:
        sample_idx = load_shap_json(Path(args.input_json)).get("sample_idx", "unknown")
        return Path("RAG/QA/Summary") / f"sample_{sample_idx}_feature_qa.jsonl"

    return Path("RAG/QA/Summary") / "default_output.jsonl"

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
    from score_gap_nomic import ask_rag_with_score_gap

    shap_dir = Path(args.shap_dir)
    query_output_dir = Path(args.query_output_dir)

    retrieval_params = {
        "min_docs": args.min_docs,
        "max_docs": args.max_docs,
        "min_gap": args.min_gap,
        "max_chunks_per_source": args.max_chunks_per_source,
        "inspect_all": False,
    }

    # 입력 파일 목록 만들기
    if args.sample_indices or args.sample_index_file:
        sample_indices = select_sample_indices(args)

        input_json_paths = [
            shap_json_path(shap_dir, sample_idx)
            for sample_idx in sample_indices
        ]

    elif args.input_json:
        input_json_paths = [Path(args.input_json)]

    else:
        sample_indices = sample_indices_from_shap_dir(shap_dir)

        input_json_paths = [
            shap_json_path(shap_dir, sample_idx)
            for sample_idx in sample_indices
        ]

    total_records = 0

    for file_idx, input_json_path in enumerate(input_json_paths, 1):
        shap_data = load_shap_json(input_json_path)
        sample_idx = shap_data.get("sample_idx", "unknown")
        pred = shap_data.get("prediction", {})

        output_path = resolve_output_path(args, sample_idx=sample_idx)
        summary_path = summary_output_path(output_path, args.summary_output)

        top_features = top_feature_payload(shap_data, top_k=3)

        _, feature_queries = generated_queries_by_feature(
            shap_data,
            top_features,
        )

        sample_queries = []
        all_records = []

        print(f"\n=== Processing {file_idx}/{len(input_json_paths)} | sample_idx={sample_idx} ===")

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
                f"[sample_idx={sample_idx} | feature {feature_rank}/{len(top_features)}] "
                f"feature={feature_info['feature']}"
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
                "shap_json_path": str(input_json_path),
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

            all_records.append(record)

        write_sample_queries(
            query_output_dir / f"sample_{sample_idx}_queries.json",
            sample_idx,
            input_json_path,
            sample_queries,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as jsonl_file:
            for record in all_records:
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        write_summary_csv(summary_path, all_records)

        total_records += len(all_records)

        print(f"Saved JSONL  : {output_path}")
        print(f"Saved CSV    : {summary_path}")
        print(f"Saved queries: {query_output_dir / f'sample_{sample_idx}_queries.json'}")

    print(f"\nTotal QA pairs: {total_records}")
        
if __name__ == "__main__":
    main()
