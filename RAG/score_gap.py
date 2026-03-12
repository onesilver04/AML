import argparse
from datetime import datetime
import json
import math

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

from prompts import RAG_SYSTEM, RAG_USER_TEMPLATE


PERSIST_DIR = "RAG/rag_db"
COLLECTION_NAME = "finance_papers"

LLM_MODEL = "qwen3.5:27b" # 모델 변경해가면서 시도 -- qwen2.5:14b, qwen3.5:27b, glm-5:cloud, gemma3:latest, deepseek-r1:latest
EMBED_MODEL = "nomic-embed-text"

# ✅ "스크립트에 그냥 박아두고" 쓰고 싶으면 여기만 편집하면 됨
DEFAULT_QUERIES = [
    "Higher loan duration is associated with higher default risk.",
    "Higher checking account balance is associated with lower default risk.",
    "Having no known savings is associated with higher default risk.",
    "What is the impact of music genre on listener engagement?",
]

# Ollama 임베딩 모델을 사용해 Chroma 벡터DB(PERSIST_DIR) 로드
def load_db():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    return db

# evidence 쪽수 표시
def page_display(meta_page):
    if isinstance(meta_page, int):
        return meta_page + 1
    return meta_page

# 최종 선택된 문서 chunk들을 LLM 입력용 context 문자열로 합치는 함수
def format_context(scored_docs, max_chars=12000):
    """
    scored_docs: [(Document, score), ...]
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

# 생성된 질문-답변 쌍을 한국어로 번역하는 함수
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

# 벡터 DB에서 query와 유사한 후보 chunk들을 score와 함께 가져오는 함수
def get_all_scored_docs(db, question: str):
    """
    Chroma의 score는 보통 similarity가 아니라 distance에 가까워서
    '작을수록 더 관련 있음'으로 해석하는 것이 일반적.
    """
    results = db.similarity_search_with_score(question, k=100) # k: candidate pool size
    results = sorted(results, key=lambda x: x[1])  # 작은 score(distance) 우선
    return results

# score 분포의 gap을 이용해 상위 relevant cluster를 고르는 함수
def find_score_gap_cutoff(
    scored_docs,
    min_docs=3,
    max_docs=100,
    min_gap=0.0,
):
    """
    scored_docs: [(Document, raw_score), ...]
    raw_score는 distance라고 가정 (작을수록 관련성 높음)

    반환:
      selected_scored_docs, cut_index, gaps
    """
    if not scored_docs:
        return [], 0, []

    limited = scored_docs[:max_docs]
    scores = [score for _, score in limited]

    if len(scores) <= min_docs:
        return limited, len(limited), []

    gaps = []
    for i in range(len(scores) - 1):
        gap = scores[i + 1] - scores[i]
        gaps.append(gap)

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

# # 같은 source의 chunk가 너무 많이 포함되지 않도록 source당 최대 N개만 남기는 함수
# retrieval 편중이 줄어듦
def apply_source_cap(scored_docs, max_chunks_per_source=2): # source당 최대 N개 유지
    """
    score-gap으로 선택된 결과 안에서 같은 source당 최대 N개 chunk만 유지.
    완전 dedup 대신 '균형 전략'으로 사용.
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

# 현재 선택된 chunk들이 source별로 몇 개씩 포함되어 있는지 출력하는 함수
def print_source_distribution(scored_docs):
    print("\n===== Source Distribution =====\n")
    counter = {}
    for d, _ in scored_docs:
        src = d.metadata.get("source", "unknown")
        counter[src] = counter.get(src, 0) + 1

    for src, cnt in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"{src}: {cnt}")

# retrieval score의 흐름과 인접 score 간 gap을 눈으로 확인하기 위한 함수
def print_score_sequence(scored_docs, max_rows=30):
    """
    similarity score 흐름을 직관적으로 보기 위한 출력
    """
    print("\n===== Score Sequence =====\n")

    scores = [score for _, score in scored_docs]

    limit = min(len(scores), max_rows)

    for i in range(limit):
        print(f"{i+1:03d}: {scores[i]:.6f}")

    print("\n===== Score Gaps =====\n")

    for i in range(limit - 1):
        gap = scores[i+1] - scores[i]
        print(
            f"{i+1:03d}->{i+2:03d} | "
            f"{scores[i]:.6f} -> {scores[i+1]:.6f} | "
            f"gap={gap:.6f}"
        )

# retrieval → score-gap → source cap → LLM generation 전체 파이프라인을 수행하는 함수        
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

    # 2) score-gap으로 상위 relevant cluster 선택
    gap_selected_docs, cut_index, gaps = find_score_gap_cutoff(
        all_scored_docs,
        min_docs=min_docs,
        max_docs=max_docs,
        min_gap=min_gap,
    )

    # 3) 동일 source 편중을 막기 위해 source당 최대 N개로 제한
    final_selected_docs = apply_source_cap(
        gap_selected_docs,
        max_chunks_per_source=max_chunks_per_source,
    )

    # 4) 디버깅/분석용 출력
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

# CLI 인자를 받아 전체 실험을 실행하고 결과를 출력하는 메인 함수
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", "--question", dest="question", type=str, required=False,
                        help="Your question (English recommended). If omitted, DEFAULT_QUERIES will be used.")
    parser.add_argument("--translate", action="store_true",
                        help="If set, translate LLM answer into Korean and print it.")

    parser.add_argument("--min_docs", type=int, default=3,
                        help="Minimum number of docs to keep before applying score gap cutoff.")
    parser.add_argument("--max_docs", type=int, default=100,
                        help="Maximum number of docs to inspect for score gap cutoff.")
    parser.add_argument("--min_gap", type=float, default=0.0,
                        help="Minimum gap required to apply cutoff. If best gap <= min_gap, keep all inspected docs.")
    parser.add_argument("--max_chunks_per_source", type=int, default=2,
                        help="Maximum number of chunks allowed per source after score-gap selection.")
    parser.add_argument("--no_inspect_all", action="store_true",
                        help="If set, do not print all retrieved scores.")
    args = parser.parse_args()

    questions = [args.question] if args.question else DEFAULT_QUERIES
    translate_llm = ChatOllama(model=LLM_MODEL, temperature=0.0) if args.translate else None

    for idx, q in enumerate(questions, 1):
        answer, selected_docs, all_docs, cut_index, gaps, gap_selected_docs = ask_rag_with_score_gap(
            q,
            min_docs=args.min_docs,
            max_docs=args.max_docs,
            min_gap=args.min_gap,
            max_chunks_per_source=args.max_chunks_per_source,
            inspect_all=not args.no_inspect_all,
        )

        print(f"\n========== Query {idx} ==========\n")
        print("Question:", q)

        print("\n===== Score-Gap Selected Docs (Before Source Cap) =====\n")
        print(f"Selected {len(gap_selected_docs)} docs (cut_index={cut_index})")
        for i, (d, score) in enumerate(gap_selected_docs, 1):
            src = d.metadata.get("source", "unknown")
            page = page_display(d.metadata.get("page", "NA"))
            print(f"{i}. score={score:.6f} | {src} (page={page})")

        print("\n===== Final Selected Docs (After Source Cap) =====\n")
        print(f"Selected {len(selected_docs)} docs")
        for i, (d, score) in enumerate(selected_docs, 1):
            src = d.metadata.get("source", "unknown")
            page = page_display(d.metadata.get("page", "NA"))
            print(f"{i}. score={score:.6f} | {src} (page={page})")

        # 최종 LLM 답변 출력
        print("\n===== LLM Generation Text =====\n")
        print(answer)

        # 한국어 번역도 출력
        if args.translate:
            translated = translate_pair(translate_llm, q, answer)
            print("\n===== Translation (KO) =====\n")
            print(translated)

if __name__ == "__main__":
    main()