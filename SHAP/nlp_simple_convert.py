# llm 넣기 전 간단 틀에 맞춘 자연어 변환
import json

def load_shap_json(path="SHAP/shap_tuples.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_fixed_template_text(data: dict, top_n_summary: int = 3) -> str:
    tuples = data.get("tuples", [])
    prediction = data.get("prediction", {})

    if not tuples:
        return "No SHAP tuples available."

    # 상위 feature
    top_features = [item["feature"] for item in tuples[:top_n_summary]]
    top_features_text = ", ".join(top_features)

    lines = []

    # 1️⃣ 요약 문장
    lines.append(f"The most influential features are {top_features_text}.")

    # 2️⃣ 각 feature 설명
    for item in tuples:
        feature = item["feature"]
        shap_value = item["shap_value"]
        direction = item["direction"]

        lines.append(
            f"The contribution of '{feature}' is {shap_value:.4f}, and its direction is '{direction}'."
        )

    # 3️⃣ prediction 문장
    label = prediction.get("label", "UNKNOWN")
    probability = prediction.get("probability", None)

    if probability is not None:
        lines.append(
            f"The predicted credit risk for this individual is '{label}' with a probability of {probability:.4f}."
        )
    else:
        lines.append(
            f"The predicted credit risk for this individual is '{label}'."
        )

    return "\n".join(lines)

def save_text(text: str, path="SHAP/shap_fixed_template_en.txt"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved: {path}")

if __name__ == "__main__":
    data = load_shap_json("SHAP/shap_tuples.json")
    result_text = build_fixed_template_text(data, top_n_summary=3)

    print(result_text)
    save_text(result_text, "SHAP/shap_fixed_template.txt")