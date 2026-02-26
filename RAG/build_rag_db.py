import os
from glob import glob
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


PDF_DIR = "papers"
PERSIST_DIR = "rag_db"
COLLECTION_NAME = "finance_papers"

# 임베딩 모델(권장: nomic-embed-text / bge-m3 등)
EMBED_MODEL = "nomic-embed-text"


def load_pdfs(pdf_dir: str):
    pdf_paths = sorted(glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    docs = []
    for path in tqdm(pdf_paths, desc="Loading PDFs"):
        loader = PyPDFLoader(path)
        file_docs = loader.load()  # page 단위 Document 리스트
        # 메타데이터에 파일명 넣기 (추적/검증에 중요)
        for d in file_docs:
            d.metadata["source"] = os.path.basename(path)
        docs.extend(file_docs)
    return docs


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_chroma(chunks):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # persist 폴더가 이미 있으면 "추가"로 이어붙이기 가능
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    vectordb.add_documents(chunks)
    vectordb.persist()
    return vectordb


def main():
    docs = load_pdfs(PDF_DIR)
    chunks = split_docs(docs)
    print(f"Loaded pages: {len(docs)}, Chunks: {len(chunks)}")

    db = build_chroma(chunks)
    print("Done. Vector DB persisted to:", PERSIST_DIR)


if __name__ == "__main__":
    main()