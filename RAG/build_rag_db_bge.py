import os
import torch
import numpy as np
from glob import glob
from uuid import uuid4
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import chromadb

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# 설정
# =========================
PDF_DIR = "papers"
# 💡 BGE 전용 DB임을 구분하기 위해 경로 이름을 변경하는 것을 추천합니다.
PERSIST_DIR = "RAG/rag_db_bge" 
COLLECTION_NAME = "finance_papers"
BGE_MODEL = "BAAI/bge-m3"

# =========================
# 헬퍼 함수: BGE-M3 임베딩 추출 (Dense Retrieval 전용)
# =========================
def get_bge_embeddings(texts, model, tokenizer, device):
    # BGE-M3는 최대 8192 토큰까지 지원하지만, 안전하게 1024~2048 정도로 제한하여 사용합니다.
    encoded_input = tokenizer(texts, padding=True, truncation=True, max_length=1024, return_tensors='pt').to(device)
    
    with torch.no_grad():
        model_output = model(**encoded_input)
        # BGE-M3 공식 가이드에 따라 [CLS] 토큰(첫 번째 토큰)의 벡터를 사용합니다.
        # 이후 코사인 유사도 계산을 위해 정규화(Normalization)를 수행합니다.
        embeddings = model_output[0][:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
    return embeddings.cpu().numpy()

# =========================
# 메인 프로세스
# =========================
def main():
    # 1. PDF 로드 및 분할
    pdf_paths = glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_paths:
        print(f"Error: {PDF_DIR} 폴더에 PDF 파일이 없습니다.")
        return

    all_chunks = []
    # BGE-M3는 문맥 파악 능력이 좋으므로 chunk_size를 약간 늘려도(예: 1000) 잘 작동합니다.
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)

    for path in tqdm(pdf_paths, desc="PDF 로딩"):
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            for d in docs:
                # 메타데이터에 파일명과 페이지 번호를 확실히 저장
                d.metadata["source"] = os.path.basename(path)
            all_chunks.extend(splitter.split_documents(docs))
        except Exception as e:
            print(f"Error loading {path}: {e}")

    print(f"총 청크 수: {len(all_chunks)}")

    # 2. BGE-M3 모델 로드 (GPU 사용)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {BGE_MODEL} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL)
    model = AutoModel.from_pretrained(BGE_MODEL).to(device)
    model.eval()

    # 3. 임베딩 생성 및 데이터 준비
    ids, texts, metadatas, embeddings = [], [], [], []
    batch_size = 12  # BGE-M3는 Gemma보다 가벼워 배치를 조금 더 키울 수 있습니다.
    
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="BGE 임베딩 생성"):
        batch = all_chunks[i : i + batch_size]
        batch_texts = [c.page_content for c in batch]
        
        batch_embs = get_bge_embeddings(batch_texts, model, tokenizer, device)
        
        for chunk, emb in zip(batch, batch_embs):
            ids.append(str(uuid4()))
            texts.append(chunk.page_content)
            metadatas.append(chunk.metadata)
            embeddings.append(emb.tolist())

    # 4. ChromaDB 저장
    if not os.path.exists(os.path.dirname(PERSIST_DIR)):
        os.makedirs(os.path.dirname(PERSIST_DIR), exist_ok=True)
        
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    
    # 기존 컬렉션 삭제 (초기화)
    try: 
        client.delete_collection(COLLECTION_NAME)
        print(f"기존 컬렉션 '{COLLECTION_NAME}' 삭제 완료.")
    except: 
        pass
    
    collection = client.create_collection(
        name=COLLECTION_NAME, 
        metadata={"hnsw:space": "cosine"} # 코사인 유사도 사용
    )
    
    # 저장 시 배치 처리 (ChromaDB 안정성을 위해 100개 단위)
    save_batch_size = 100
    for i in tqdm(range(0, len(texts), save_batch_size), desc="DB 저장"):
        collection.add(
            ids=ids[i : i + save_batch_size],
            documents=texts[i : i + save_batch_size],
            metadatas=metadatas[i : i + save_batch_size],
            embeddings=embeddings[i : i + save_batch_size]
        )
        
    print(f"완료! DB 저장 위치: {PERSIST_DIR}")

if __name__ == "__main__":
    main()