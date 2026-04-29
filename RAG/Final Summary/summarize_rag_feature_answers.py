import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_INPUT = Path("RAG/QA/test_correct_5.jsonl")
DEFAULT_INPUT_DIR = Path("RAG/QA/Answers")
DEFAULT_OUTPUT_DIR = Path("RAG/Final Summary/Results")
DEFAULT_MODEL = "qwen3.6:35b"
EXPECTED_FEATURE_COUNT = 3


SUMMARY_SYSTEM_PROMPT = """You are a concise Korean financial risk explanation writer.
Your job is to compress three feature-level RAG answers into exactly one Korean sentence for a credit-risk UI."""


SUMMARY_USER_TEMPLATE = """Create one final Korean explanation sentence for this credit-risk prediction.

Prediction:
- label: {prediction_label}
- probability: {prediction_probability}

Feature evidence:
{feature_evidence}

Rules:
1. Output exactly one Korean sentence only.
2. Mention all three features in the same order as provided.
3. Put the feature's citation marker immediately after the clause for that feature, using [1], [2], and [3].
4. Do not use bullets, line breaks, headings, markdown, or extra commentary.
5. Keep the sentence short enough for a UI explanation.
6. If the model SHAP direction and the RAG literature direction conflict, briefly say that they differ.
7. End by reflecting the final prediction label naturally in Korean.
"""

def tokenize_for_evidence(text: str):
    stopwords = {
        "a", "an", "and", "are", "as", "associated", "be", "by",
        "credit", "default", "for", "from", "in", "is", "it",
        "of", "on", "or", "probability", "risk", "statistical",
        "the", "to", "with",
    }
    return {
        token
        for token in re.findall(r"[A-Za-z0-9_<>/=+-]+", (text or "").lower())
        if len(token) > 2 and token not in stopwords
    }


def split_sentences(text: str):
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'([])", normalized)
        if sentence.strip()
    ]


def page_display(meta_page):
    if isinstance(meta_page, int):
        return meta_page + 1
    return meta_page


def extract_rag_core_sentences_for_record(
    record,
    min_docs=3,
    max_docs=20,
    min_gap=0.001,
    max_chunks_per_source=2,
    sentences_per_feature=1,
):
    from score_gap_nomic import ask_rag_with_score_gap

    question = record.get("question", "")
    feature_text = " ".join(
        [
            record.get("feature", ""),
            record.get("feature_definition", ""),
            record.get("feature_direction", ""),
            question,
        ]
    )

    answer, selected_docs, _, _, _, _ = ask_rag_with_score_gap(
        question,
        min_docs=min_docs,
        max_docs=max_docs,
        min_gap=min_gap,
        max_chunks_per_source=max_chunks_per_source,
        inspect_all=False,
    )

    query_terms = tokenize_for_evidence(feature_text)
    candidate_sentences = []

    for doc, score in selected_docs:
        for sentence in split_sentences(doc.page_content):
            sentence_terms = tokenize_for_evidence(sentence)
            overlap = len(sentence_terms & query_terms)

            if overlap == 0:
                continue

            candidate_sentences.append(
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": page_display(doc.metadata.get("page", "NA")),
                    "score": float(score),
                    "overlap": overlap,
                    "sentence": sentence,
                }
            )

    candidate_sentences = sorted(
        candidate_sentences,
        key=lambda item: (
            -item["overlap"],
            item["score"],
            len(item["sentence"]),
        ),
    )

    return {
        "feature": record.get("feature", "unknown"),
        "question": question,
        "rag_answer_for_sentence_search": answer,
        "core_sentences": candidate_sentences[:sentences_per_feature],
        "selected_sources": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": page_display(doc.metadata.get("page", "NA")),
                "score": float(score),
            }
            for doc, score in selected_docs
        ],
    }


def build_rag_core_evidence(records):
    core_evidence = []

    for ref_number, record in enumerate(records, 1):
        retrieval_params = record.get("retrieval_params") or {}

        core = extract_rag_core_sentences_for_record(
            record,
            min_docs=retrieval_params.get("min_docs", 3),
            max_docs=retrieval_params.get("max_docs", 20),
            min_gap=retrieval_params.get("min_gap", 0.001),
            max_chunks_per_source=retrieval_params.get("max_chunks_per_source", 2),
            sentences_per_feature=1,
        )

        core_evidence.append(
            {
                "ref": ref_number,
                **core,
            }
        )

    return core_evidence


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize per-feature RAG answers into one Korean sentence per sample."
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Single feature QA JSONL file to summarize. Default: {DEFAULT_INPUT}",
    )
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help=f"Directory of sample_*_feature_qa.jsonl files to summarize. Default batch dir: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for final summary JSON and CSV outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path for single-file mode.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV path for batch output.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama chat model for summarization. Default: {DEFAULT_MODEL}",
    )
    return parser.parse_args()


def load_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return records


def validate_records(records, path: Path):
    if len(records) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"{path} must contain exactly {EXPECTED_FEATURE_COUNT} records, "
            f"but found {len(records)}."
        )

    sample_ids = {record.get("sample_idx") for record in records}
    if len(sample_ids) != 1:
        raise ValueError(f"{path} must contain records for exactly one sample: {sample_ids}")

    return sorted(records, key=lambda record: int(record.get("feature_rank", 0)))


def representative_reference(record, ref_number: int):
    sources = record.get("selected_sources") or []
    first_source = sources[0] if sources else {}
    return {
        "ref": ref_number,
        "feature": record.get("feature", "unknown"),
        "source": first_source.get("source", "unknown"),
        "page": first_source.get("page", "unknown"),
    }


def direction_label(direction: str):
    if direction == "increase_risk":
        return "모델에서는 위험을 높이는 방향"
    if direction == "decrease_risk":
        return "모델에서는 위험을 낮추는 방향"
    return "모델 방향 미확인"


def compact_answer(answer: str, max_chars: int = 1200):
    normalized = " ".join((answer or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rsplit(" ", 1)[0] + "..."


def format_feature_evidence(records):
    blocks = []
    for ref_number, record in enumerate(records, 1):
        blocks.append(
            "\n".join(
                [
                    f"[{ref_number}]",
                    f"- feature: {record.get('feature', 'unknown')}",
                    f"- definition: {record.get('feature_definition', 'unknown')}",
                    f"- shap_direction: {direction_label(record.get('feature_direction', 'UNKNOWN'))}",
                    f"- shap_value: {record.get('feature_shap_value', 'unknown')}",
                    f"- rag_answer: {compact_answer(record.get('answer', ''))}",
                ]
            )
        )
    return "\n\n".join(blocks)


def clean_summary(text: str):
    summary = " ".join((text or "").split())
    summary = re.sub(r"^[\"'“”]+|[\"'“”]+$", "", summary).strip()
    return summary


def validate_summary(summary: str):
    missing_refs = [ref for ref in ("[1]", "[2]", "[3]") if ref not in summary]
    if missing_refs:
        raise ValueError(
            f"LLM summary is missing citation marker(s): {', '.join(missing_refs)}. "
            f"Summary: {summary}"
        )
    if "\n" in summary:
        raise ValueError(f"LLM summary must not contain line breaks: {summary}")


def build_llm(model: str):
    try:
        from langchain_ollama import ChatOllama
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'langchain_ollama'. Install the same RAG runtime "
            "dependencies used by score_gap_nomic.py before running this script."
        ) from exc

    return ChatOllama(model=model, temperature=0.0)


def summarize_records(records, llm):
    try:
        from langchain_core.prompts import ChatPromptTemplate
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'langchain_core'. Install the same RAG runtime "
            "dependencies used by score_gap_nomic.py before running this script."
        ) from exc

    first = records[0]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUMMARY_SYSTEM_PROMPT),
            ("user", SUMMARY_USER_TEMPLATE),
        ]
    )
    chain = prompt | llm
    result = chain.invoke(
        {
            "prediction_label": first.get("prediction_label", "UNKNOWN"),
            "prediction_probability": first.get("prediction_probability", "UNKNOWN"),
            "feature_evidence": format_feature_evidence(records),
        }
    )
    summary = clean_summary(result.content)
    validate_summary(summary)
    return summary


def build_payload(records, final_explanation):
    first = records[0]
    references = [
        representative_reference(record, ref_number)
        for ref_number, record in enumerate(records, 1)
    ]

    rag_core_evidence = build_rag_core_evidence(records)

    return {
        "sample_idx": first.get("sample_idx"),
        "prediction_label": first.get("prediction_label", "UNKNOWN"),
        "prediction_probability": first.get("prediction_probability", "UNKNOWN"),
        "final_explanation": final_explanation,
        "references": references,
        "rag_core_evidence_sentences": rag_core_evidence,
    }

def output_path_for_input(input_path: Path, output_dir: Path):
    match = re.search(r"sample_(\d+)_feature_qa\.jsonl$", input_path.name)
    if match:
        return output_dir / f"sample_{match.group(1)}_final_summary.json"
    return output_dir / f"{input_path.stem}_final_summary.json"


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_csv(path: Path, payloads):
    fieldnames = [
        "sample_idx",
        "prediction_label",
        "prediction_probability",
        "final_explanation",
        "references",
        "rag_core_evidence_sentences",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for payload in payloads:
            row = payload.copy()
            row["references"] = json.dumps(row["references"], ensure_ascii=False)
            row["rag_core_evidence_sentences"] = json.dumps(
                row.get("rag_core_evidence_sentences", []),
                ensure_ascii=False
            )
            writer.writerow(row)


def summarize_file(input_path: Path, output_path: Path, llm):
    records = validate_records(load_jsonl(input_path), input_path)
    final_explanation = summarize_records(records, llm)
    payload = build_payload(records, final_explanation)
    write_json(output_path, payload)
    return payload


def iter_input_files(input_dir: Path):
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    files = sorted(input_dir.glob("sample_*_feature_qa.jsonl"))
    if not files:
        raise FileNotFoundError(f"No sample_*_feature_qa.jsonl files found in: {input_dir}")
    return files


def main():
    args = parse_args()
    llm = build_llm(args.model)

    if args.input_dir:
        input_files = iter_input_files(args.input_dir)
        payloads = []
        for input_path in input_files:
            output_path = output_path_for_input(input_path, args.output_dir)
            payload = summarize_file(input_path, output_path, llm)
            payloads.append(payload)
            print(f"Saved JSON: {output_path}")

        csv_output = args.csv_output or args.output_dir / "final_summaries.csv"
        write_csv(csv_output, payloads)
        print(f"Saved CSV : {csv_output}")
        print(f"Total summaries: {len(payloads)}")
        return

    output_path = args.output or output_path_for_input(args.input, args.output_dir)
    payload = summarize_file(args.input, output_path, llm)
    print(f"Saved JSON: {output_path}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ModuleNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
