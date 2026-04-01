import os
import json
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# 모델 및 경로 설정
MODEL_NAME = "qwen3.5:27b"
MAPPING_JSON_PATH = "RAG/feature_mapping.json"
SHAP_JSON_PATH = "SHAP/Feature Importance/shap_tuples_non_prefix_14.json"
OUTPUT_PATH = "RAG/Query/generated_rag_queries_14.json" # 결과를 리스트 형식(JSON)으로 저장

def load_json_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_chain(model_name: str = MODEL_NAME):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an expert financial risk analyst. Your task is to generate a list of independent RAG search queries.

TASK:
1. Identify the top 3-4 most important features from the [SHAP RESULT].
2. For EACH feature, create ONE independent academic sentence describing its relationship with credit risk or default probability.
3. Use the [FEATURE MAPPING] to translate codes (e.g., A14) into precise financial terms.

STRATEGY FOR BETTER RAG:
- Each sentence must focus on ONLY ONE feature to ensure high retrieval accuracy.
- Use academic phrasing: "The relationship between [Specific Feature] and the probability of credit default."
- Avoid ML jargon like 'SHAP' or 'feature importance'.
- Output the sentences as a bulleted list or numbered list.
"""
        ),
        (
            "user",
            """
### FEATURE MAPPING:
{mapping}

### SHAP RESULT:
{shap_result}

Generate 3-4 independent academic queries for RAG:
"""
        )
    ])

    llm = ChatOllama(model=model_name, temperature=0.0)
    return prompt | llm

def postprocess_queries(text: str) -> list[str]:
    # 줄바꿈이나 불릿 기호를 기준으로 문장들을 분리
    lines = text.strip().split('\n')
    queries = []
    for line in lines:
        # 불릿, 숫자, 접두사 제거
        clean_line = re.sub(r'^(\d+\.|\-|\*|Query:)\s*', '', line).strip()
        if len(clean_line) > 20: # 의미 있는 길이의 문장만 포함
            if clean_line[-1] not in ".!?":
                clean_line += "."
            queries.append(clean_line)
    return queries

def main():
    print("데이터 로딩 중...")
    try:
        mapping_dict = load_json_data(MAPPING_JSON_PATH)
        shap_dict = load_json_data(SHAP_JSON_PATH)
        
        # LLM에 넘길 때는 문자열로 변환
        mapping_str = json.dumps(mapping_dict, indent=2)
        shap_str = json.dumps(shap_dict, indent=2)
    except Exception as e:
        print(f"오류 발생: {e}")
        return

    print(f"Generating independent queries using {MODEL_NAME}...")
    chain = build_chain()
    response = chain.invoke({
        "mapping": mapping_str,
        "shap_result": shap_str
    })

    # 여러 개의 쿼리를 리스트로 추출
    rag_queries = postprocess_queries(response.content)
    
    print("\n===== Generated RAG Queries =====")
    for i, q in enumerate(rag_queries, 1):
        print(f"{i}. {q}")
    
    # 결과를 RAG 스크립트에서 쓰기 좋게 JSON 리스트로 저장
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rag_queries, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(rag_queries)} queries to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()