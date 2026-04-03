import os
from glob import glob
from typing import List, Tuple
from uuid import uuid4
from collections import defaultdict
import torch

from tqdm import tqdm
from transformers import AutoModel
import chromadb

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# 설정
# =========================
PDF_DIR = "papers"
PERSIST_DIR = "RAG/rag_db_pplx"
COLLECTION_NAME = "finance_papers"

# 문서용 / 질의용 모델 분리
DOC_EMBED_MODEL = "perplexity-ai/pplx-embed-context-v1-4B"
QUERY_EMBED_MODEL = "perplexity-ai/pplx-embed-v1-4B"

# =========================
# PDF 로드
# =========================
def load_pdfs(pdf_dir: str):
    pdf_paths = sorted(glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"PDF 파일이 {pdf_dir} 폴더에 없습니다.")

    docs = []
    for path in tqdm(pdf_paths, desc="PDF 로딩 중"):
        loader = PyPDFLoader(path)
        file_docs = loader.load()
        for d in file_docs:
            d.metadata["source"] = os.path.basename(path)
            # page 정보가 없을 수도 있으니 안전하게 보정
            d.metadata["page"] = d.metadata.get("page", 0)
        docs.extend(file_docs)
    return docs

# =========================
# 청크 분할
# =========================
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)

# =========================
# context 모델용 그룹핑
# 같은 source/page 단위로 묶어서
# "주변 청크 문맥"을 함께 반영
# =========================
def group_chunks_for_context(chunks):
    groups = defaultdict(list)

    for chunk in chunks:
        key = (
            chunk.metadata.get("source", "unknown"),
            chunk.metadata.get("page", 0),
        )
        groups[key].append(chunk)

    # 정렬된 순서로 반환
    grouped = []
    for key in sorted(groups.keys()):
        grouped.append(groups[key])

    return grouped

# =========================
# 문서 임베딩 생성
# pplx-embed-context-v1-4B는
# List[List[str]] 형태로 넣어야
# 각 청크가 주변 청크를 반영해 임베딩됨
# =========================
def build_contextual_embeddings(grouped_chunks):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 중인 디바이스: {device}")
    model_ctx = AutoModel.from_pretrained(
        DOC_EMBED_MODEL,
        trust_remote_code=True,
    ).to(device)

    all_texts = []
    all_metadatas = []
    all_ids = []
    all_embeddings = []

    for group in tqdm(grouped_chunks, desc="문서 임베딩 생성 중"):
        texts = [c.page_content for c in group]

        # 반환 형태:
        # group_embeddings.shape == (len(texts), 2560)
        group_embeddings = model_ctx.encode(texts)

        for chunk, emb in zip(group, group_embeddings):
            all_texts.append(chunk.page_content)
            all_metadatas.append(chunk.metadata)
            all_ids.append(str(uuid4()))
            all_embeddings.append(emb.tolist())

    return all_ids, all_texts, all_metadatas, all_embeddings

# =========================
# Chroma 저장
# Perplexity 모델은 cosine 사용 권장
# =========================
def save_to_chroma(ids, texts, metadatas, embeddings):
    os.makedirs(PERSIST_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=PERSIST_DIR)

    # 이미 있으면 삭제 후 재생성
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 128
    for i in tqdm(range(0, len(texts), batch_size), desc="Chroma 저장 중"):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=texts[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
        )

    return collection

# =========================
# 질의 임베더
# =========================
class PplxQueryEmbedder:
    def __init__(self, model_name=QUERY_EMBED_MODEL):
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

    def embed_query(self, text: str) -> List[float]:
        emb = self.model.encode(text)
        return emb.tolist()

# =========================
# 메인
# =========================
def main():
    docs = load_pdfs(PDF_DIR)
    chunks = split_docs(docs)
    print(f"로드된 페이지: {len(docs)}, 생성된 청크: {len(chunks)}")

    grouped_chunks = group_chunks_for_context(chunks)
    ids, texts, metadatas, embeddings = build_contextual_embeddings(grouped_chunks)

    save_to_chroma(ids, texts, metadatas, embeddings)
    print(f"완료. 벡터 DB가 다음 위치에 저장되었습니다: {PERSIST_DIR}")

if __name__ == "__main__":
    main()