import os
import json
import argparse

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

MODEL_NAME = "mistral:7b"

DEFAULT_INPUT_PATH = "SHAP/shap_tuples.json"
DEFAULT_OUTPUT_PATH = "RAG/generated_rag_query_from_json.txt"


def load_input_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(text: str, path: str) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved: {path}")


def postprocess_text(text: str) -> str:
    text = " ".join(text.split())
    text = text.strip().strip('"').strip("'")
    if not text.endswith("."):
        text += "."
    return text


def build_chain(model_name: str = MODEL_NAME):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an assistant that converts structured credit risk model outputs into a single concise retrieval-friendly sentence.

Rules:
- Use only the information explicitly provided in the input JSON.
- Do not infer the real-world meaning of feature names.
- Do not add any external domain knowledge or assumptions.
- Reflect the prediction label and the feature directions accurately.
- Mention only the most influential features when relevant.
- Output exactly one declarative sentence in English.
- Do not use bullet points, numbering, labels, or line breaks.
- Do not mention SHAP, JSON, feature importance, contribution scores, or machine learning.
- Do not start with "Retrieve", "Find", "Search for", or "Look for".
- Keep it concise and suitable for retrieving supporting evidence from academic finance or credit risk papers.
""".strip()
        ),
        (
            "user",
            """
Input JSON:
{input_json}

Write one concise declarative sentence that expresses the relationship between the most influential features and the predicted credit risk outcome.
""".strip()
        )
    ])

    llm = ChatOllama(model=model_name, temperature=0.2)
    return prompt | llm


def generate_rag_query(input_json: dict, model_name: str = MODEL_NAME) -> str:
    chain = build_chain(model_name=model_name)
    json_text = json.dumps(input_json, ensure_ascii=False, indent=2)
    result = chain.invoke({"input_json": json_text})
    return postprocess_text(result.content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Path to shap_tuples.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the generated RAG query."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Ollama model name (default: mistral:7b)"
    )
    args = parser.parse_args()

    input_json = load_input_json(args.input)
    rag_query = generate_rag_query(input_json, model_name=args.model)

    print("\n===== Generated RAG Query =====\n")
    print(rag_query)

    print("\n===== Copy for DEFAULT_QUERIES =====\n")
    print(f'DEFAULT_QUERIES = [\n    "{rag_query}",\n]')

    save_text(rag_query, args.output)


if __name__ == "__main__":
    main()