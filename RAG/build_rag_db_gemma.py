import os
import torch
from glob import glob
from uuid import uuid4
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import chromadb
from huggingface_hub import login
login(token="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")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# 설정
# =========================
PDF_DIR = "papers"
PERSIST_DIR = "RAG/rag_db_gemma"
COLLECTION_NAME = "finance_papers"
GEMMA_MODEL = "google/embeddinggemma-300m"  # 접근 권한 필요 (huggingface-cli login)

# =========================
# 헬퍼 함수: Gemma 임베딩 추출
# =========================
def get_gemma_embeddings(texts, model, tokenizer, device):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # 전용 모델은 보통 마지막 hidden state(또는 특정 풀링)를 바로 사용하도록 설계됨
        embeddings = outputs.last_hidden_state[:, -1, :] # 마지막 토큰 활용
    return embeddings.cpu().numpy()

# =========================
# 메인 프로세스
# =========================
def main():
    # 1. PDF 로드 및 분할
    pdf_paths = glob(os.path.join(PDF_DIR, "*.pdf"))
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    for path in tqdm(pdf_paths, desc="PDF 로딩"):
        loader = PyPDFLoader(path)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = os.path.basename(path)
        all_chunks.extend(splitter.split_documents(docs))

    print(f"총 청크 수: {len(all_chunks)}")

    # 2. Gemma 모델 로드 (GPU 사용)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL)
    model = AutoModel.from_pretrained(GEMMA_MODEL, torch_dtype=torch.float16).to(device)
    model.eval()

    # 3. 임베딩 생성 및 데이터 준비
    ids, texts, metadatas, embeddings = [], [], [], []
    batch_size = 8  # GPU 메모리에 맞게 조절
    
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Gemma 임베딩 생성"):
        batch = all_chunks[i : i + batch_size]
        batch_texts = [c.page_content for c in batch]
        
        batch_embs = get_gemma_embeddings(batch_texts, model, tokenizer, device)
        
        for chunk, emb in zip(batch, batch_embs):
            ids.append(str(uuid4()))
            texts.append(chunk.page_content)
            metadatas.append(chunk.metadata)
            embeddings.append(emb.tolist())

    # 4. ChromaDB 저장
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    try: client.delete_collection(COLLECTION_NAME)
    except: pass
    
    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    
    # 저장 시 배치 처리
    for i in tqdm(range(0, len(texts), 64), desc="DB 저장"):
        collection.add(
            ids=ids[i:i+64],
            documents=texts[i:i+64],
            metadatas=metadatas[i:i+64],
            embeddings=embeddings[i:i+64]
        )
    print(f"완료! DB 저장 위치: {PERSIST_DIR}")

if __name__ == "__main__":
    main()