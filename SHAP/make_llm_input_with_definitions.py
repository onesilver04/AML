import os
import json
import argparse
from typing import Dict, Any, List


DEFAULT_JSON_PATH = "SHAP/shap_tuples_non_prefix.json"
DEFAULT_OUTPUT_PATH = "SHAP/llm_input_with_definitions.txt"


# ==========================================================
# 1) German Credit 변수 정의 사전
#    - 필요하면 너 데이터셋 컬럼명에 맞게 계속 추가/수정하면 됨
# ==========================================================
EXACT_FEATURE_DEFINITIONS: Dict[str, str] = {
    # numeric / ordinal
    "duration": "Loan duration in months.",
    "credit_amount": "Credit amount requested by the applicant.",
    "installment_commitment": "Installment rate as a percentage of disposable income.",
    "age": "Applicant age in years.",
    "existing_credits": "Number of existing credits at this bank.",
    "residence_since": "Present residence duration.",
    "num_dependents": "Number of dependents.",
    "people_liable": "Number of people liable for maintenance.",
    "foreign_worker": "Whether the applicant is a foreign worker.",
    "telephone": "Whether the applicant has a telephone.",
    "present_residence": "Years at current residence.",
    "number_credits": "Number of existing credits.",
    "employment_duration": "Present employment duration.",
    "credit_history": "Past credit history category.",
    "savings_status": "Savings account status category.",
    "checking_status": "Checking account status category.",
}

PREFIX_RULES = [
    # checking status
    ("checking_status_<0", "Checking account status: balance below 0 DM."),
    ("checking_status_0<=X<200", "Checking account status: balance between 0 and 200 DM."),
    ("checking_status_>=200", "Checking account status: balance at least 200 DM."),
    ("checking_status_no checking", "Checking account status: no checking account."),

    # savings
    ("savings_status_<100", "Savings account status: savings below 100 DM."),
    ("savings_status_100<=X<500", "Savings account status: savings between 100 and 500 DM."),
    ("savings_status_500<=X<1000", "Savings account status: savings between 500 and 1000 DM."),
    ("savings_status_>=1000", "Savings account status: savings at least 1000 DM."),
    ("savings_status_no known savings", "Savings account status: no known savings account."),

    # employment
    ("employment_unemployed", "Employment status: unemployed."),
    ("employment_<1", "Present employment duration: less than 1 year."),
    ("employment_1<=X<4", "Present employment duration: between 1 and 4 years."),
    ("employment_4<=X<7", "Present employment duration: between 4 and 7 years."),
    ("employment_>=7", "Present employment duration: 7 years or more."),

    # credit history
    ("credit_history_no credits/all paid", "Credit history: no credits taken or all previous credits paid back duly."),
    ("credit_history_all paid", "Credit history: all previous credits at this bank paid back duly."),
    ("credit_history_existing paid", "Credit history: existing credits paid back duly so far."),
    ("credit_history_delayed previously", "Credit history: delay in paying off previous credits."),
    ("credit_history_critical/other existing credit", "Credit history: critical account or other existing credits."),

    # purpose
    ("purpose_radio/tv", "Loan purpose: purchase of radio or TV."),
    ("purpose_used car", "Loan purpose: purchase of a used car."),
    ("purpose_new car", "Loan purpose: purchase of a new car."),
    ("purpose_furniture/equipment", "Loan purpose: furniture or equipment purchase."),
    ("purpose_business", "Loan purpose: business."),
    ("purpose_education", "Loan purpose: education."),
    ("purpose_repairs", "Loan purpose: repairs."),
    ("purpose_domestic appliances", "Loan purpose: domestic appliances."),
    ("purpose_retraining", "Loan purpose: retraining."),
    ("purpose_other", "Loan purpose: other."),

    # housing
    ("housing_own", "Housing status: owns housing."),
    ("housing_rent", "Housing status: rents housing."),
    ("housing_for free", "Housing status: lives for free."),

    # personal status / sex
    ("personal_status_male single", "Personal status: male and single."),
    ("personal_status_female div/dep/mar", "Personal status: female, divorced, dependent, or married."),
    ("personal_status_male mar/wid", "Personal status: male and married or widowed."),
    ("personal_status_male div/sep", "Personal status: male and divorced or separated."),

    # property
    ("property_magnitude_real estate", "Property: real estate."),
    ("property_magnitude_life insurance", "Property: life insurance."),
    ("property_magnitude_car", "Property: car or other non-real-estate property."),
    ("property_magnitude_no known property", "Property: no known property."),

    # job
    ("job_unemp/unskilled non res", "Job category: unemployed or unskilled and non-resident."),
    ("job_unskilled resident", "Job category: unskilled and resident."),
    ("job_skilled", "Job category: skilled employee or official."),
    ("job_high qualif/self emp/mgmt", "Job category: highly qualified, self-employed, or management."),

    # debtors
    ("other_parties_none", "Other debtors or guarantors: none."),
    ("other_parties_co applicant", "Other debtors or guarantors: co-applicant."),
    ("other_parties_guarantor", "Other debtors or guarantors: guarantor."),
]


# ==========================================================
# 2) 유틸
# ==========================================================
def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(text: str, path: str) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved: {path}")


def direction_to_text(direction: str) -> str:
    if direction == "increase_risk":
        return "increases credit risk"
    if direction == "decrease_risk":
        return "decreases credit risk"
    return "has an unclear effect on credit risk"


def get_feature_definition(feature_name: str) -> str:
    # exact match 먼저
    if feature_name in EXACT_FEATURE_DEFINITIONS:
        return EXACT_FEATURE_DEFINITIONS[feature_name]

    # prefix/패턴 매칭
    for prefix, definition in PREFIX_RULES:
        if feature_name == prefix:
            return definition

    # 부분 매칭 fallback
    for prefix, definition in PREFIX_RULES:
        if feature_name.startswith(prefix):
            return definition

    # 일반 fallback
    return f"No curated definition found for '{feature_name}'. Use the feature name as given without adding outside assumptions."


def build_llm_input_text(data: Dict[str, Any]) -> str:
    sample_idx = data.get("sample_idx", "NA")
    pred = data.get("prediction", {})
    label = pred.get("label", "UNKNOWN")
    prob = pred.get("probability", "UNKNOWN")
    tuples: List[Dict[str, Any]] = data.get("tuples", [])

    lines: List[str] = []
    lines.append("Credit Risk Prediction Instance")
    lines.append(f"Sample index: {sample_idx}")
    lines.append(f"Predicted label: {label}")
    lines.append(f"Predicted probability: {prob}")
    lines.append("")
    lines.append("Most influential raw features:")
    lines.append("")

    for i, item in enumerate(tuples, start=1):
        feature = item.get("feature", "UNKNOWN_FEATURE")
        shap_value = item.get("shap_value", "UNKNOWN")
        abs_shap = item.get("abs_shap", "UNKNOWN")
        direction = item.get("direction", "UNKNOWN")
        definition = get_feature_definition(feature)

        lines.append(f"{i}. Feature: {feature}")
        lines.append(f"   Definition: {definition}")
        lines.append(f"   SHAP value: {shap_value}")
        lines.append(f"   Absolute SHAP value: {abs_shap}")
        lines.append(f"   Direction: {direction} ({direction_to_text(direction)})")
        lines.append("")

    lines.append("Instruction for the LLM:")
    lines.append(
        "Using only the feature names, SHAP directions, and feature definitions above, "
        "write one concise English sentence suitable for retrieving supporting evidence "
        "from academic finance or credit risk papers."
    )
    lines.append(
        "Do not mention SHAP, JSON, contribution scores, or machine learning internals."
    )
    lines.append(
        "Do not add outside knowledge that is not grounded in the provided feature definitions."
    )
    lines.append(
        "If a feature is categorical, avoid vague expressions such as 'increase in the feature'; "
        "instead, describe the specific category in natural language."
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_JSON_PATH,
        help="Path to shap_tuples_non_prefix.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the LLM input text",
    )
    args = parser.parse_args()

    data = load_json(args.input)
    llm_input_text = build_llm_input_text(data)

    print("\n===== Generated LLM Input Text =====\n")
    print(llm_input_text)

    save_text(llm_input_text, args.output)


if __name__ == "__main__":
    main()