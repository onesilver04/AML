import os
import argparse

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

MODEL_NAME = "mistral-small3.2:24b" # model test: mistral:7b, llama3.1:8b, qwen3.5:27b, mistral-small3.2:24b

DEFAULT_INPUT_PATH = "SHAP/shap_fixed_template.txt"
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
            "user",
            """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
The general context is credit risk prediction using machine learning models. Don’t deviate from this frame. Do not answer in lengthy form; be brief, concise, and precise at all costs.

In the input, you will be given a prediction from a machine learning model for credit risk classification, categorized as either GOOD CREDIT RISK or BAD CREDIT RISK, along with the most influential features, their contribution values, and their contribution directions.

Produce a single concise explanatory sentence in English that expresses the likely relationship between the most influential features and the predicted credit risk outcome.

The sentence should:    
- be written as a natural declarative statement, not a command or instruction
- be suitable for retrieving supporting evidence from academic finance or credit risk papers
- mention the most influential features when relevant
- reflect the prediction direction
- not mention SHAP
- not use bullet points
- not include explanations before or after the sentence
- not start with phrases such as "Retrieve", "Find", "Search for", or "Look for"

### Input:
{input_text}

### Response:
""".strip()
        )
    ])

    llm = ChatOllama(model=model_name, temperature=0.2)
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