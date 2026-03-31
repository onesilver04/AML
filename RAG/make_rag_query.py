import os
import argparse
import re

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

MODEL_NAME = "qwen3.5:27b"

DEFAULT_INPUT_PATH = "SHAP/llm_input_with_definitions_3.txt"
DEFAULT_OUTPUT_PATH = "RAG/generated_rag_query_3.txt"


def load_input_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_text(text: str, path: str) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved: {path}")


def postprocess_query(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(Query:|Sentence:|Output:)\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s+", " ", text).strip()
    text = text.split("\n")[0].strip()

    if text and text[-1] not in ".!?":
        text += "."

    return text


def build_chain(model_name: str = MODEL_NAME):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an expert assistant for credit risk literature retrieval.

Your task is to convert structured feature-risk statements into a single retrieval-friendly query sentence for RAG.

Rules:
- Use only the provided statements.
- Preserve the exact risk direction: "higher credit risk" or "lower credit risk".
- Keep the wording factual and suitable for matching academic finance or credit risk papers.
- Prefer explicit relationship wording such as "is associated with higher credit risk" or "is associated with lower credit risk".
- Combine related statements into one sentence using natural connectors such as "while".
- Do not mention SHAP, feature importance, prediction probability, machine learning, JSON, or contribution scores.
- Do not add explanations, hedging, or background context.
- Do not introduce new concepts such as default probability, creditworthiness, financial instability, or repayment behavior unless they already appear in the input.
- Output exactly one concise English sentence.
"""
        ),
        (
            "user",
            """
Convert the following input into one RAG-friendly query sentence for academic literature retrieval:

{input_text}

Return exactly one concise sentence.
""".strip()
        )
    ])

    llm = ChatOllama(
        model=model_name,
        temperature=0.0,
    )
    return prompt | llm


def generate_rag_query(input_text: str, model_name: str = MODEL_NAME) -> str:
    chain = build_chain(model_name=model_name)
    result = chain.invoke({"input_text": input_text})
    return postprocess_query(result.content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Path to the structured English explanation text."
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
        help="Ollama model name"
    )
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