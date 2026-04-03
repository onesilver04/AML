import argparse
from datetime import datetime
import json
import os
import re
from typing import List

import torch
from transformers import AutoModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

# 프로프트 파일이 따로 없다면 아래 주석을 해제하고 직접 정의하세요.
# RAG_SYSTEM = "You are a helpful assistant..."
# RAG_USER_TEMPLATE = "Context: {context}\n\nQuestion: {question}"
from prompts import RAG_SYSTEM, RAG_USER_TEMPLATE

# =========================
# 설정 (PPLX 전용)
# =========================
PERSIST_DIR = "RAG/rag_db_pplx"  # pplx DB 경로
COLLECTION_NAME = "finance_papers"
RESULT_DIR = "RAG/Result"

LLM_MODEL = "qwen3.5:27b"
# 질의 시에는 컨텍스트 모델이 아닌 일반 임베딩 모델을 사용합니다.
QUERY_EMBED_MODEL = "perplexity-ai/pplx-embed-v1-4B"

# =========================
# Perplexity 전용 임베딩 클래스
# =========================
class PplxRAGEmbeddings:
    def __init__(self, model_name: str):
        print(f"임베딩 모델 로드 중: {model_name}")
        # trust_remote_code=True는 pplx 모델 로드에 필수입니다.
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval() # 추론 모드

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # DB 로드 시에는 저장된 벡터를 쓰므로 실제 호출되지는 않지만 인터페이스 유지를 위해 정의
        return []

    def embed_query(self, text: str) -> List[float]:
        with torch.no_grad():
            # pplx 모델의 encode 메서드를 호출
            emb = self.model.encode(text)
        return emb.tolist()

# =========================
# 기본 쿼리 및 유틸리티
# =========================
DEFAULT_QUERIES = [
    "Empirical analysis of the relationship between negative checking account balances and increased credit default probability.",
    "Statistical impact of the absence of an existing checking account on elevated credit risk.",
    "Empirical evidence on the correlation between extended credit duration in months and increased credit risk."
]

def load_db():
    # PPLX 임베딩 함수 생성
    embeddings = PplxRAGEmbeddings(model_name=QUERY_EMBED_MODEL)
    
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return db

def page_display(meta_page):
    if isinstance(meta_page, int):
        return meta_page + 1
    return meta_page

def format_context(scored_docs, max_chars=5000):
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

# =========================
# 검색 로직 (Score-Gap & Source Cap)
# =========================
def get_all_scored_docs(db, question: str):
    # k를 100정도로 충분히 가져와서 Gap을 분석합니다.
    results = db.similarity_search_with_score(question, k=100)
    # 점수가 높은(유사도가 높은) 순으로 정렬
    return sorted(results, key=lambda x: x[1], reverse=True)

def find_score_gap_cutoff(scored_docs, min_docs=3, max_docs=20, min_gap=0.001):
    if not scored_docs:
        return [], 0, []

    limited = scored_docs[:max_docs]
    scores = [score for _, score in limited]

    if len(scores) <= min_docs:
        return limited, len(limited), []

    gaps = []
    for i in range(len(scores) - 1):
        gaps.append(scores[i] - scores[i + 1])

    search_start = max(min_docs - 1, 0)
    candidate_gaps = gaps[search_start:]
    
    if not candidate_gaps:
        return limited, len(limited), gaps

    relative_idx = max(range(len(candidate_gaps)), key=lambda i: candidate_gaps[i])
    best_gap_idx = search_start + relative_idx
    
    if gaps[best_gap_idx] <= min_gap:
        cut_index = len(limited)
    else:
        cut_index = best_gap_idx + 1

    return limited[:cut_index], cut_index, gaps

def apply_source_cap(scored_docs, max_chunks_per_source=1):
    selected = []
    source_counter = {}
    for d, score in scored_docs:
        src = d.metadata.get("source", "unknown")
        count = source_counter.get(src, 0)
        if count < max_chunks_per_source:
            selected.append((d, score))
            source_counter[src] = count + 1
    return selected

# =========================
# 메인 실행부
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=str, help="질문 내용 (생략 시 기본 쿼리 사용)")
    parser.add_argument("--min_docs", type=int, default=3)
    parser.add_argument("--max_docs", type=int, default=20)
    parser.add_argument("--min_gap", type=float, default=0.001)
    parser.add_argument("--max_chunks_per_source", type=int, default=1)
    args = parser.parse_args()

    # DB 및 모델 로드
    db = load_db()
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
    
    questions = [args.q] if args.q else DEFAULT_QUERIES
    
    # 결과 저장 준비
    now = datetime.now()
    os.makedirs(RESULT_DIR, exist_ok=True)
    save_path = os.path.join(RESULT_DIR, now.strftime("%Y%m%d_%H%M%S_pplx.txt"))
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"### PPLX RAG Experiment - {now} ###\n\n")

        for idx, q in enumerate(questions, 1):
            print(f"\nProcessing Query {idx}: {q}")
            
            # 1. 검색 및 컷오프
            all_scored = get_all_scored_docs(db, q)
            gap_docs, cut_idx, _ = find_score_gap_cutoff(
                all_scored, min_docs=args.min_docs, max_docs=args.max_docs, min_gap=args.min_gap
            )
            final_docs = apply_source_cap(gap_docs, max_chunks_per_source=args.max_chunks_per_source)
            
            # 2. 답변 생성
            context = format_context(final_docs, max_chars=5000)
            prompt = ChatPromptTemplate.from_messages([("system", RAG_SYSTEM), ("user", RAG_USER_TEMPLATE)])
            chain = prompt | llm
            answer = chain.invoke({"question": q, "context": context}).content

            # 3. 출력 및 저장
            output = f"== Query {idx} ==\nQ: {q}\n"
            output += f"Selected {len(final_docs)} chunks (Cutoff at {cut_idx})\n"
            for i, (d, s) in enumerate(final_docs, 1):
                output += f" {i}. [{s:.4f}] {d.metadata['source']} (p.{page_display(d.metadata['page'])})\n"
            output += f"\n[Answer]\n{answer}\n\n"
            
            print(output)
            f.write(output)

    print(f"실험 완료! 결과 저장됨: {save_path}")

if __name__ == "__main__":
    main()