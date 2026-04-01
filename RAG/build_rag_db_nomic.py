import os
from glob import glob
from typing import List
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# 설정값 최적화
PDF_DIR = "papers"
PERSIST_DIR = "RAG/rag_db"
COLLECTION_NAME = "finance_papers"
EMBED_MODEL = "nomic-embed-text"
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

class PrefixedOllamaEmbeddings(OllamaEmbeddings):
    """Nomic-embed-text의 성능을 최적화하기 위한 프리픽스 적용 클래스"""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"{DOC_PREFIX}{t}" for t in texts]
        return super().embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(f"{QUERY_PREFIX}{text}")

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
        docs.extend(file_docs)
    return docs

def split_docs(docs):
    # 제안 3: 금융 문서의 문맥 보존을 위해 chunk_size를 800으로 상향
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)

def get_vector_db(chunks=None):
    embeddings = PrefixedOllamaEmbeddings(model=EMBED_MODEL)

    # 코사인 유사도 설정을 위한 메타데이터
    # hnsw:space를 'cosine'으로 설정해야 코사인 유사도 기반으로 인덱싱됩니다.
    collection_metadata = {"hnsw:space": "cosine"}

    # DB 폴더가 이미 존재하고 새로운 청크가 인자로 들어오지 않은 경우 (로드 모드)
    if os.path.exists(PERSIST_DIR) and chunks is None:
        print("기존 Vector DB를 로드합니다. (Cosine Similarity 설정 확인 권장)")
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
            collection_metadata=collection_metadata # 로드 시에도 명시
        )
    
    # 새로운 Vector DB를 생성하거나 업데이트하는 경우
    print("새로운 Vector DB를 생성/업데이트합니다. (공간: Cosine Similarity)")
    
    # 만약 기존에 L2로 생성된 DB가 있다면, 정확한 테스트를 위해 
    # PERSIST_DIR 폴더를 직접 삭제한 후 실행하는 것을 권장합니다.
    
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        collection_metadata=collection_metadata # 생성 시 코사인 유사도 적용
    )
    return vectordb

def main():
    # 1. 문서 로드 및 분할
    docs = load_pdfs(PDF_DIR)
    chunks = split_docs(docs)
    print(f"로드된 페이지: {len(docs)}, 생성된 청크: {len(chunks)}")

    # 2. 벡터 DB 생성 및 저장
    db = get_vector_db(chunks)
    print(f"완료. 벡터 DB가 다음 위치에 저장되었습니다: {PERSIST_DIR}")

if __name__ == "__main__":
    main()