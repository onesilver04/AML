import os
import json
import re
from typing import List

# 모델 및 경로 설정
MODEL_NAME = "qwen3.6:35b"
# 이제 definition이 포함된 통합 JSON 파일 하나만 사용합니다.
INPUT_JSON_PATH = "SHAP/Feature Importance/shap_tuples_non_prefix_41.json" 
OUTPUT_PATH = "RAG/Query/generated_rag_queries_41.txt"

def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일이 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_chain(model_name: str = MODEL_NAME):
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an expert financial risk analyst. Your task is to generate a LIST of independent RAG search queries based on the provided feature definitions.

TASK:
1. Look at the 'tuples' in the input JSON.
2. For EACH tuple, use the 'definition' and 'direction' to create ONE formal academic search query.
3. The query should focus on the relationship between that specific feature and credit risk.

STRATEGY:
- Create independent sentences (e.g., "Statistical impact of [Definition] on credit default probability").
- Use the 'direction' to refine the tone (e.g., if 'increase_risk', focus on its contribution to default).
- Use academic terms: 'empirical analysis', 'determinants of default', 'credit risk drivers'.
- Output exactly one sentence per feature in a list format.
- Do NOT mention SHAP, JSON, tuples, or specific codes like A11.
"""
        ),
        (
            "user",
            "Generate RAG queries for these specific features:\n\n{input_json}"
        )
    ])

    llm = ChatOllama(model=model_name, temperature=0.0)
    return prompt | llm

def postprocess_queries(text: str) -> List[str]:
    # 줄바꿈이나 불릿 기호를 기준으로 문장 분리
    raw_lines = text.strip().split('\n')
    queries = []
    for line in raw_lines:
        # 숫자, 불릿(-, *), Query: 등의 접두사 제거
        clean = re.sub(r'^(\d+\.|\-|\*|Query:)\s*', '', line).strip()
        if len(clean) > 20: 
            if clean[-1] not in ".!?":
                clean += "."
            queries.append(clean)
    return queries

def generate_rag_queries(input_data: dict, model_name: str = MODEL_NAME) -> List[str]: # 
    chain = build_chain(model_name)
    response = chain.invoke({
        "input_json": json.dumps(input_data, indent=2, ensure_ascii=False)
    })
    return postprocess_queries(response.content)

def main():
    print(f"데이터 로딩 중: {INPUT_JSON_PATH}")
    try:
        # 통합된 JSON 데이터 로드
        input_data = load_json(INPUT_JSON_PATH)
    except Exception as e:
        print(f"오류: {e}")
        return

    print(f"Generating queries using {MODEL_NAME}...")
    rag_queries = generate_rag_queries(input_data)
    
    print("\n===== Generated RAG Queries =====")
    for i, q in enumerate(rag_queries, 1):
        print(f"{i}. {q}")
    
    # 결과 저장
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rag_queries, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
