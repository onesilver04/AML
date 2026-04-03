import argparse
from datetime import datetime
import math
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from prompts import RAG_SYSTEM, RAG_USER_TEMPLATE

# =========================
# 설정 및 로그인
# =========================
# ⚠️ 실제로는 토큰 하드코딩하지 말고 환경변수 사용 권장
HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN:
    login(token=HF_TOKEN)

PERSIST_DIR = "RAG/rag_db_gemma"
COLLECTION_NAME = "finance_papers"
RESULT_DIR = "RAG/Result"

LLM_MODEL = "qwen3.5:27b"
EMBED_MODEL_NAME = "google/embeddinggemma-300m"

# =========================
# Gemma 전용 임베딩 클래스
# =========================
class GemmaRAGEmbeddings:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Loading Gemma Embedding model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state  # [B, T, H]
            attention_mask = inputs["attention_mask"]  # [B, T]

            # 마지막 유효 토큰 위치 계산
            last_token_idx = attention_mask.sum(dim=1) - 1  # [B]
            batch_indices = torch.arange(hidden.size(0), device=self.device)
            embeddings = hidden[batch_indices, last_token_idx, :]  # [B, H]

            # cosine similarity 안정화를 위해 정규화
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

# =========================
# 기본 질의
# =========================
DEFAULT_QUERIES = [
"Having no checking account is associated with higher credit risk."
]

# =========================
# DB 로드
# =========================
def load_db():
    embeddings = GemmaRAGEmbeddings(model_name=EMBED_MODEL_NAME)
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return db, embeddings

# =========================
# 유틸
# =========================
def page_display(meta_page):
    if isinstance(meta_page, int):
        return meta_page + 1
    return meta_page

def is_valid_number(x) -> bool:
    try:
        return x is not None and not math.isnan(float(x)) and not math.isinf(float(x))
    except Exception:
        return False

def safe_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)

    if a.ndim != 1 or b.ndim != 1:
        return float("nan")
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    if len(a) != len(b):
        return float("nan")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return float("nan")

    score = float(np.dot(a, b) / (norm_a * norm_b))
    return score

def format_context(scored_docs: List[Tuple[object, float]], max_chars=5000):
    parts = []
    total = 0
    for d, score in scored_docs:
        src = d.metadata.get("source", "unknown")
        page = page_display(d.metadata.get("page", "NA"))
        chunk = f"[source={src} page={page} cosine_similarity={score:.6f}]\n{d.page_content}\n"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n---\n".join(parts)

# =========================
# cosine similarity 직접 계산
# =========================
def get_all_scored_docs(db, embeddings, question: str, k: int = 100):
    # 1) 문서 검색은 일단 relevance search로 가져오고
    # 2) 점수는 query/doc embedding으로 직접 cosine similarity 계산
    docs = db.similarity_search(question, k=k)
    query_emb = embeddings.embed_query(question)

    scored_results = []
    for d in docs:
        doc_text = d.page_content.strip()
        if not doc_text:
            continue

        doc_emb = embeddings.embed_query(doc_text)
        score = safe_cosine_similarity(query_emb, doc_emb)

        if is_valid_number(score):
            scored_results.append((d, score))

    # cosine similarity는 클수록 더 유사
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return scored_results

# =========================
# score-gap cutoff
# =========================
def find_score_gap_cutoff(scored_docs, min_docs=3, max_docs=50, min_gap=0.001):
    if not scored_docs:
        return [], 0, []

    limited = [
        (doc, score) for doc, score in scored_docs[:max_docs]
        if is_valid_number(score)
    ]

    if not limited:
        return [], 0, []

    scores = [score for _, score in limited]

    if len(scores) <= min_docs:
        return limited, len(limited), []

    # 높은 점수 -> 낮은 점수 순이므로 앞 - 뒤
    gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
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

# =========================
# source cap
# =========================
def apply_source_cap(scored_docs, max_chunks_per_source=3):
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
# RAG 질의
# =========================
def ask_rag_with_score_gap(question, min_docs, max_docs, min_gap, max_chunks_per_source, db, embeddings):
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM),
        ("user", RAG_USER_TEMPLATE)
    ])

    all_scored_docs = get_all_scored_docs(db, embeddings, question, k=100)
    gap_docs, cut_idx, gaps = find_score_gap_cutoff(
        all_scored_docs,
        min_docs=min_docs,
        max_docs=max_docs,
        min_gap=min_gap
    )
    final_docs = apply_source_cap(gap_docs, max_chunks_per_source=max_chunks_per_source)

    context = format_context(final_docs)
    chain = prompt | llm
    answer = chain.invoke({
        "question": question,
        "context": context
    }).content

    return answer, final_docs, all_scored_docs, cut_idx, gaps, gap_docs

# =========================
# 디버그 출력
# =========================
def format_debug_scores(scored_docs, limit=10):
    lines = []
    for i, (d, s) in enumerate(scored_docs[:limit], 1):
        src = d.metadata.get("source", "unknown")
        page = page_display(d.metadata.get("page", "NA"))
        lines.append(f"{i:02d}. score={s:.6f} | {src} (p.{page})")
    return "\n".join(lines)

def format_gaps(gaps, limit=10):
    if not gaps:
        return "No gaps"
    lines = []
    for i, g in enumerate(gaps[:limit], 1):
        lines.append(f"{i:02d}. gap={g:.6f}")
    return "\n".join(lines)

# =========================
# 메인 실행부
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=str, help="Question")
    parser.add_argument("--min_docs", type=int, default=3)
    parser.add_argument("--max_docs", type=int, default=20)
    parser.add_argument("--min_gap", type=float, default=0.001)
    parser.add_argument("--max_chunks_per_source", type=int, default=2)  # 기본값 2로 변경
    parser.add_argument("--debug_topk", type=int, default=10)
    args = parser.parse_args()

    db, embeddings = load_db()

    questions = [args.q] if args.q else DEFAULT_QUERIES
    now = datetime.now()
    os.makedirs(RESULT_DIR, exist_ok=True)
    save_path = os.path.join(RESULT_DIR, now.strftime("%Y%m%d_%H%M%S_gemma_cosine.txt"))

    final_output_buffer = [f"### Gemma RAG Result - {now} ###\n"]

    for idx, q in enumerate(questions, 1):
        print(f"\n[Processing Query {idx}] {q}")

        answer, selected_docs, all_docs, cut_idx, gaps, gap_selected = ask_rag_with_score_gap(
            q,
            args.min_docs,
            args.max_docs,
            args.min_gap,
            args.max_chunks_per_source,
            db,
            embeddings
        )

        res_text = f"\n========== Query {idx} ==========\n"
        res_text += f"Question: {q}\n"
        res_text += f"Cut-off Index: {cut_idx}, Selected Chunks: {len(selected_docs)}\n\n"

        res_text += "[Top Retrieved Chunks by Cosine Similarity]\n"
        res_text += format_debug_scores(all_docs, limit=args.debug_topk) + "\n\n"

        res_text += "[Score Gaps]\n"
        res_text += format_gaps(gaps, limit=args.debug_topk) + "\n\n"

        res_text += "[Final Selected Chunks]\n"
        for i, (d, s) in enumerate(selected_docs, 1):
            src = d.metadata.get("source", "unknown")
            page = page_display(d.metadata.get("page", "NA"))
            res_text += f"{i}. score={s:.6f} | {src} (p.{page})\n"

        res_text += f"\n[LLM Answer]\n{answer}\n"

        print(res_text)
        final_output_buffer.append(res_text)

    with open(save_path, "w", encoding="utf-8") as f:
        f.writelines(final_output_buffer)

    print(f"\n[INFO] 결과 저장 완료: {save_path}")

if __name__ == "__main__":
    main()