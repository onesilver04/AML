# 논문을 참고해 프롬프트 수정한 버전
# 사람 친화적인 결과를 쿼리로 사용한다.

import os
import argparse

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

MODEL_NAME = "mistral:7b"

DEFAULT_INPUT_PATH = "SHAP/shap_fixed_template.txt"
DEFAULT_OUTPUT_PATH = "RAG/generated_rag_query_userfriendly.txt"


def load_input_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved: {path}")

# 랭체인
def build_chain(model_name: str = MODEL_NAME):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an expert in credit risk analysis and Machine Learning (ML) models.
Your task is to explain the decision of a credit risk prediction model for a given applicant.
The explanation should be concise, accurate, and easily understandable for a non-expert user.

Guidelines:
- Focus only on the most influential features.
- Explain how the features relate to the predicted credit risk outcome.
- Clearly reflect whether each feature increases or decreases credit risk.
- Write the explanation as a short natural paragraph in flowing prose.
- Do not use bullet points, numbering, section titles, labels, or line breaks.
- Do not list the features mechanically one by one.
- Integrate the important features into one coherent explanation.
- Use natural English.
- Do not mention SHAP, contribution scores, feature importance, or internal model mechanics.
""".strip()
        ),
        (
            "user",
            """
Applicant Features and Prediction:
{input_text}

Provide an explanation for this prediction focusing on the most influential features.
Write it as a natural paragraph, not as a list.
""".strip()
        )
    ])

    llm = ChatOllama(model=model_name, temperature=0)
    return prompt | llm

def generate_rag_query(input_text: str, model_name: str = MODEL_NAME) -> str:
    chain = build_chain(model_name=model_name)
    result = chain.invoke({"input_text": input_text})
    return result.content.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_PATH,
                        help="Path to the structured English explanation text.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH,
                        help="Path to save the generated RAG query.")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help="Ollama model name (default: mistral:7b)")
    args = parser.parse_args()

    input_text = load_input_text(args.input)
    rag_query = generate_rag_query(input_text, model_name=args.model)

    print("\n===== Generated RAG Query =====\n")
    print(rag_query)

    print("\n===== Copy for DEFAULT_QUERIES =====\n")
    print(f'DEFAULT_QUERIES = [\n    "{rag_query}",\n]')

    save_text(rag_query, args.output)


if __name__ == "__main__":
    main()