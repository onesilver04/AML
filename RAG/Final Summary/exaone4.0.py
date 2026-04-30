import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_INPUT = Path("RAG/Final Summary/Wrong_Results/sample_3_final_summary.json")
DEFAULT_OUTPUT = None

# 👉 HuggingFace EXAONE 4.5 모델 (사용 가능한 repo로 수정 필요)
MODEL_NAME = "LGAI-EXAONE/EXAONE-4.0-1.2B"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate final_explanation using EXAONE 4.5 (HuggingFace)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSON path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path. Default: input filename + _ko.json",
    )
    return parser.parse_args()


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def output_path_for_input(input_path: Path):
    return input_path.with_name(f"{input_path.stem}_ko{input_path.suffix}")


# 🔥 EXAONE 번역 프롬프트
def build_prompt(text: str):
    return f"""You are a professional Korean translator for financial risk explanations.

Translate the following English explanation into natural Korean.

Rules:
- Preserve [1], [2], [3]
- Do NOT add explanation
- Keep structure identical
- Output ONLY Korean

Text:
{text}

Korean:
"""


def load_model():
    print("Loading EXAONE model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    return tokenizer, model


def translate(text, tokenizer, model):
    prompt = build_prompt(text)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 🔥 prompt 이후 부분만 추출
    translated = result.split("Korean:")[-1].strip()

    return translated


def main():
    args = parse_args()

    payload = load_json(args.input)

    if "final_explanation" not in payload:
        raise ValueError("final_explanation field missing.")

    tokenizer, model = load_model()

    translated = translate(payload["final_explanation"], tokenizer, model)

    payload["final_explanation_ko"] = translated

    output_path = args.output or output_path_for_input(args.input)
    write_json(output_path, payload)

    print(f"Saved: {output_path}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)