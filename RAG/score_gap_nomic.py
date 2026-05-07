import argparse
from datetime import datetime
import os
from typing import List

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

from prompts import RAG_SYSTEM, RAG_USER_TEMPLATE


PERSIST_DIR = "RAG/rag_db_nomic"
COLLECTION_NAME = "finance_papers"
RESULT_DIR = "RAG/Result"

LLM_MODEL = "qwen3.6:35b"
EMBED_MODEL = "nomic-embed-text"
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

class PrefixedOllamaEmbeddings(OllamaEmbeddings):
    """Apply retrieval prefixes recommended by nomic-embed-text."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"{DOC_PREFIX}{t}" for t in texts]
        return super().embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(f"{QUERY_PREFIX}{text}")


DEFAULT_QUERIES = [
  "Empirical analysis of the relationship between present employment since less than one year and increased credit default probability.",
  "Statistical impact of the status of existing checking account being no checking account on elevated credit risk.",
  "Determinants of credit default associated with the status of existing checking account between 0 and 200 Deutsche Mark."
]


def load_db():
    embeddings = PrefixedOllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"} # cosine
    )
    return db


def page_display(meta_page):
    if isinstance(meta_page, int):
        return meta_page + 1
    return meta_page


def format_context(scored_docs, max_chars=5000):
    """
    scored_docs: [(Document, distance), ...]
    distance는 작을수록 더 관련 있음
    """
    parts = []
    total = 0

    for d, score in scored_docs:
        src = d.metadata.get("source", "unknown")
        page = page_display(d.metadata.get("page", "NA"))
        chunk = (
            f"[source={src} page={page} raw_score={score:.6f}]\n"
            f"{d.page_content}\n"
        )

        if total + len(chunk) > max_chars:
            break

        parts.append(chunk)
        total += len(chunk)

    return "\n---\n".join(parts)


def translate_pair(llm, question, answer):
    translate_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a precise translation engine. Do not add commentary."),
        ("user",
         "Translate the following question and answer into Korean.\n\n"
         "Question:\n{question}\n\nAnswer:\n{answer}")
    ])
    chain = translate_prompt | llm
    result = chain.invoke({"question": question, "answer": answer})
    return result.content.strip()


def get_all_scored_docs(db, question: str):
    results = db.similarity_search_with_score(question, k=100)
    results = sorted(results, key=lambda x: x[1])  # 작은 거리 우선
    return results


def find_score_gap_cutoff(
    scored_docs,
    min_docs=3,
    max_docs=100,
    min_gap=0.001,
):
    """
    scored_docs: [(Document, distance), ...]
    distance는 작을수록 관련성 높음.
    distance가 급격히 커지는 지점(gap)을 cutoff로 사용.
    """
    if not scored_docs:
        return [], 0, []

    # 1. distance가 작은 순(오름차순) 정렬
    scored_docs = sorted(scored_docs, key=lambda x: x[1])

    limited = scored_docs[:max_docs]
    scores = [score for _, score in limited]

    if len(scores) <= min_docs:
        return limited, len(limited), []

    # 2. 인접 거리 차이 계산
    # distance가 갑자기 커지는 지점 = 관련 문서군이 끝나는 지점
    gaps = []
    for i in range(len(scores) - 1):
        gap = scores[i + 1] - scores[i]
        gaps.append(gap)

    # 3. 최소 문서 수 이후 가장 큰 gap 탐색
    search_start = max(min_docs - 1, 0)
    candidate_gaps = gaps[search_start:]

    if not candidate_gaps:
        return limited, len(limited), gaps

    relative_idx = max(range(len(candidate_gaps)), key=lambda i: candidate_gaps[i])
    best_gap_idx = search_start + relative_idx
    best_gap = gaps[best_gap_idx]

    if best_gap <= min_gap:
        cut_index = len(limited)
    else:
        cut_index = best_gap_idx + 1

    selected = limited[:cut_index]
    return selected, cut_index, gaps


def apply_source_cap(scored_docs, max_chunks_per_source=2):
    """
    같은 source의 chunk가 너무 많이 포함되지 않도록 source당 최대 N개만 유지
    """
    selected = []
    source_counter = {}

    for d, score in scored_docs:
        src = d.metadata.get("source", "unknown")
        count = source_counter.get(src, 0)

        if count < max_chunks_per_source:
            selected.append((d, score))
            source_counter[src] = count + 1

    return selected


def print_source_distribution(scored_docs):
    print("\n===== Source Distribution =====\n")
    counter = {}
    for d, _ in scored_docs:
        src = d.metadata.get("source", "unknown")
        counter[src] = counter.get(src, 0) + 1

    for src, cnt in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"{src}: {cnt}")


def print_score_sequence(scored_docs, max_rows=30):
    """
    distance 흐름을 직관적으로 보기 위한 출력
    """
    print("\n===== Score Sequence =====\n")

    scores = [score for _, score in scored_docs]
    limit = min(len(scores), max_rows)

    for i in range(limit):
        print(f"{i+1:03d}: {scores[i]:.6f}")

    print("\n===== Score Gaps =====\n")

    for i in range(limit - 1):
        gap = scores[i + 1] - scores[i]
        print(
            f"{i+1:03d}->{i+2:03d} | "
            f"{scores[i]:.6f} -> {scores[i+1]:.6f} | "
            f"gap={gap:.6f}"
        )


def ask_rag_with_score_gap(
    question: str,
    min_docs: int = 3,
    max_docs: int = 100,
    min_gap: float = 0.0,
    max_chunks_per_source: int = 2,
    inspect_all: bool = True,
):
    db = load_db()
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM),
        ("user", RAG_USER_TEMPLATE),
    ])

    # 1) query에 대한 후보 chunk 전체 조회
    all_scored_docs = get_all_scored_docs(db, question)

    # 필요하면 거리 흐름 직접 확인
    if inspect_all:
        print_score_sequence(all_scored_docs, max_rows=min(len(all_scored_docs), 30))

    # 2) score-gap으로 상위 relevant cluster 선택
    gap_selected_docs, cut_index, gaps = find_score_gap_cutoff(
        all_scored_docs,
        min_docs=min_docs,
        max_docs=max_docs,
        min_gap=min_gap,
    )

    # 3) 동일 source 편중 방지
    final_selected_docs = apply_source_cap(
        gap_selected_docs,
        max_chunks_per_source=max_chunks_per_source,
    )

    # 4) 디버깅 출력
    if inspect_all:
        print("\n===== Before Source Cap =====")
        print(f"Selected {len(gap_selected_docs)} docs by score-gap")
        print_source_distribution(gap_selected_docs)

        print("\n===== After Source Cap =====")
        print(f"Selected {len(final_selected_docs)} docs after source cap")
        print_source_distribution(final_selected_docs)

    # 5) 최종 선택 문서를 context로 만들어 LLM에 전달
    context = format_context(final_selected_docs)

    chain = prompt | llm
    result = chain.invoke({
        "question": question,
        "context": context
    })

    return result.content, final_selected_docs, all_scored_docs, cut_index, gaps, gap_selected_docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--q", "--question",
        dest="question",
        type=str,
        required=False,
        help="Your question (English recommended). If omitted, DEFAULT_QUERIES will be used."
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="If set, translate LLM answer into Korean and print it."
    )
    parser.add_argument(
        "--min_docs",
        type=int,
        default=3,
        help="Minimum number of docs to keep before applying score gap cutoff."
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=20,
        help="Maximum number of docs to inspect for score gap cutoff."
    )
    parser.add_argument(
        "--min_gap",
        type=float,
        default=0.001,
        help="Minimum gap required to apply cutoff."
    )
    parser.add_argument(
        "--max_chunks_per_source",
        type=int,
        default=2,
        help="Maximum number of chunks allowed per source."
    )
    parser.add_argument(
        "--no_inspect_all",
        action="store_true",
        help="If set, do not print all retrieved scores."
    )
    args = parser.parse_args()

    questions = [args.question] if args.question else DEFAULT_QUERIES
    translate_llm = ChatOllama(model=LLM_MODEL, temperature=0.0) if args.translate else None

    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    file_name = now.strftime("%Y%m%d_%H%M%S.txt")

    final_output_buffer = []
    final_output_buffer.append(f"### RAG Experiment Result - {timestamp_str} ###\n")
    final_output_buffer.append(
        f"Parameters: min_docs={args.min_docs}, max_docs={args.max_docs}, "
        f"min_gap={args.min_gap}, max_chunks_per_source={args.max_chunks_per_source}\n"
    )
    final_output_buffer.append("=" * 50 + "\n")

    for idx, q in enumerate(questions, 1):
        answer, selected_docs, all_docs, cut_index, gaps, gap_selected_docs = ask_rag_with_score_gap(
            q,
            min_docs=args.min_docs,
            max_docs=args.max_docs,
            min_gap=args.min_gap,
            max_chunks_per_source=args.max_chunks_per_source,
            inspect_all=not args.no_inspect_all,
        )

        query_header = f"\n========== Query {idx} ==========\n"
        question_text = f"Question: {q}\n"

        selected_info = f"\n[Score-Gap Selection] cut_index={cut_index}, total_before_cap={len(gap_selected_docs)}\n"
        selected_info += f"===== Final Selected Docs ({len(selected_docs)} chunks) =====\n"

        doc_list = ""
        for i, (d, score) in enumerate(selected_docs, 1):
            src = d.metadata.get("source", "unknown")
            page = page_display(d.metadata.get("page", "NA"))
            doc_list += f"{i}. score={score:.6f} | {src} (page={page})\n"

        llm_header = "\n===== LLM Generation Text =====\n"
        query_result = query_header + question_text + selected_info + doc_list + llm_header + answer + "\n"

        if args.translate:
            translated = translate_pair(translate_llm, q, answer)
            query_result += f"\n===== Translation (KO) =====\n{translated}\n"

        print(query_result)
        final_output_buffer.append(query_result)

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    save_path = os.path.join(RESULT_DIR, file_name)
    with open(save_path, "w", encoding="utf-8") as f:
        f.writelines(final_output_buffer)

    print(f"\n[INFO] 모든 결과가 저장되었습니다: {save_path}")


if __name__ == "__main__":
    main()