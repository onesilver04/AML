import os
import torch
from glob import glob
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 설정 값
PDF_DIR = "papers"
PERSIST_DIR = "RAG/rag_db"
COLLECTION_NAME = "finance_papers"
# 최신 gte-Qwen2-7B-instruct 모델 경로
EMBED_MODEL_NAME = "Alibaba-NLP/gte-Qwen2-7B-instruct"

from langchain_huggingface import HuggingFaceEmbeddings

class QwenInstructionEmbeddings(HuggingFaceEmbeddings):
    def embed_query(self, text: str) -> list[float]:
        # 쿼리 시에만 인스트럭션 추가
        instruction = "Given a web search query, retrieve relevant passages that answer the query."
        return super().embed_query(f"Instruction: {instruction}\nQuery: {text}")

# build_chroma 함수 내에서 사용 시:
def build_chroma(chunks):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    embeddings = QwenInstructionEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={'device': device, 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vectordb.add_documents(chunks)
    return vectordb

def load_pdfs(pdf_dir: str):
    pdf_paths = sorted(glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    docs = []
    for path in tqdm(pdf_paths, desc="Loading PDFs"):
        loader = PyPDFLoader(path)
        file_docs = loader.load()
        for d in file_docs:
            d.metadata["source"] = os.path.basename(path)
        docs.extend(file_docs)
    return docs

def split_docs(docs):
    # gte-Qwen2는 긴 문맥에 강하므로 chunk_size를 1000~1500 정도로 키워도 성능이 잘 유지됩니다.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def main():
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
        print(f"Directory {PDF_DIR} created. Please add PDFs.")
        return

    docs = load_pdfs(PDF_DIR)
    chunks = split_docs(docs)
    print(f"Loaded pages: {len(docs)}, Chunks: {len(chunks)}")

    db = build_chroma(chunks)
    print("Done. Vector DB persisted to:", PERSIST_DIR)

if __name__ == "__main__":
    main()