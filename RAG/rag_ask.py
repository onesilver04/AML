# rag_ask.py
import argparse
from datetime import datetime
import json

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

from prompts import RAG_SYSTEM, RAG_USER_TEMPLATE


PERSIST_DIR = "RAG/rag_db"
COLLECTION_NAME = "finance_papers"

LLM_MODEL = "qwen2.5:14b" # 모델 변경해가면서 시도 -- qwen2.5:14b, glm-5:cloud, gemma3:4b
EMBED_MODEL = "nomic-embed-text"


# ✅ "스크립트에 그냥 박아두고" 쓰고 싶으면 여기만 편집하면 됨
DEFAULT_QUERIES = [
    "Higher loan duration is associated with higher default risk.",
    "Higher checking account balance is associated with lower default risk.",
    "Having no known savings is associated with higher default risk.",
]


def load_db():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    return db


def format_context(docs, max_chars=12000):
    parts = []
    total = 0
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "NA")
        chunk = f"[source={src} page={page}]\n{d.page_content}\n"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n---\n".join(parts)

def ask_rag(question: str, k: int = 6):
    db = load_db()
    retriever = db.as_retriever(search_kwargs={"k": k})

    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM),
        ("user", RAG_USER_TEMPLATE),
    ])

    docs = retriever.get_relevant_documents(question)
    context = format_context(docs)

    chain = prompt | llm
    result = chain.invoke({"question": question, "context": context})

    return result.content, docs

def main():
    parser = argparse.ArgumentParser()
    # required=True -> required=False 로 변경
    parser.add_argument("--q", "--question", dest="question", type=str, required=False,
                        help="Your question (English recommended). If omitted, DEFAULT_QUERIES will be used.")
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    # ✅ 입력 우선순위: CLI 질문이 있으면 1개만 실행, 없으면 기본 질문 리스트 전부 실행
    questions = [args.question] if args.question else DEFAULT_QUERIES

    for idx, q in enumerate(questions, 1):
        answer, docs = ask_rag(q, k=args.k)

        print(f"\n========== Query {idx} ==========\n")
        print("Question:", q)

        print("\n===== RAG Answer =====\n")
        print(answer)

        print("\n===== Retrieved Sources =====\n")
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", "NA")
            print(f"{i}. {src} (page={page})")
            
if __name__ == "__main__":
    main()