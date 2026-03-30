import os
import argparse

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

MODEL_NAME = "mistral-small3.2:24b" # model test: mistral:7b, llama3.1:8b, qwen3.5:27b, mistral-small3.2:24b

DEFAULT_INPUT_PATH = "SHAP/llm_input_with_definitions.txt"
DEFAULT_OUTPUT_PATH = "RAG/generated_rag_query.txt"


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
You are an expert in credit risk modeling.

Your task is to generate a single concise sentence describing the relationship between given features and credit risk.

You must strictly follow the provided feature definitions.
Do not infer or assume meanings beyond the definitions.
Do not generalize categorical variables using vague expressions like "increase in feature".
Use precise, definition-based wording.

Avoid hallucination at all costs.
"""
        ),
        (
            "user",
            """
### Task:
Generate a single concise English sentence suitable for retrieving supporting evidence from academic credit risk literature.

### Input:
{input_text}

### Requirements:
- Use only the provided feature definitions and directions
- Reflect the prediction direction (increase/decrease risk)
- Do not mention SHAP or contribution values
- Do not add explanations
- Do not use bullet points
- Do not generalize categorical features (e.g., avoid "increase in checking status")
- Be concise and factual

### Output:
""".strip()
        )
    ])

    llm = ChatOllama(model=model_name, temperature=0.1)
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