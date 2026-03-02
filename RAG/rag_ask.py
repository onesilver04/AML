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

# Ollama 임베딩 모델을 사용해 Chroma 벡터DB(PERSIST_DIR) 로드
def load_db():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    return db

# Retriever가 가져온 문서(docs)를 LLM 프롬프트에 넣기 좋은 형태로 합침
def format_context(docs, max_chars=12000):
    parts = []
    total = 0
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "NA")
        chunk = f"[source={src} page={page}]\n{d.page_content}\n"
        if total + len(chunk) > max_chars: # 프롬프트 길이 제한
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n---\n".join(parts)

# # 생성된 답변을 target_lang으로 번역
# def translate_text(llm: ChatOllama, text: str, target_lang: str = "Korean") -> str:
#     translate_prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are a precise translation engine. Do not add commentary."),
#         ("user", "Translate the following text into {target_lang}.\n\n{text}")
#     ])

#     chain = translate_prompt | llm
#     result = chain.invoke({"target_lang": target_lang, "text": text})
#     return result.content.strip()

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

# 메인 함수
def ask_rag(question: str, k: int = 6):
    db = load_db() # 벡터DB 로드
    retriever = db.as_retriever(search_kwargs={"k": k}) # 질문(question)으로 관련 문서 k개 검색(retrieve)

    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM),
        ("user", RAG_USER_TEMPLATE),
    ])

    docs = retriever.get_relevant_documents(question) # DEFAULT_QUERIES 벡터로 변환
    context = format_context(docs) # LLM에 넣을 컨텍스트 텍스트 구성
    chain = prompt | llm # 체인으로 연결
    result = chain.invoke({"question": question, "context": context}) # LLM 모델에 넘겨 답변 문장 생성

    return result.content, docs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", "--question", dest="question", type=str, required=False,
                        help="Your question (English recommended). If omitted, DEFAULT_QUERIES will be used.")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument(
        "--translate",
        action="store_true",
        help="If set, translate LLM answer into Korean and print it."
    )
    args = parser.parse_args()

    # ✅ 입력 우선순위: CLI 질문이 있으면 1개만 실행, 없으면 기본 질문 리스트 전부 실행
    questions = [args.question] if args.question else DEFAULT_QUERIES
    
    translate_llm = ChatOllama(model=LLM_MODEL, temperature=0.0) if args.translate else None
    for idx, q in enumerate(questions, 1):
        answer, docs = ask_rag(q, k=args.k)

        print(f"\n========== Query {idx} ==========\n")
        print("Question:", q)
            
        print("\n===== LLM Generation Text =====\n")
        print(answer)
        
        if args.translate:
            translated = translate_pair(translate_llm, q, answer)
            print("\n===== Translation (KO) =====\n")
            print(translated)
            
        print("\n===== Retrieved Sources =====\n")
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", "NA")
            print(f"{i}. {src} (page={page})")
            
if __name__ == "__main__":
    main()