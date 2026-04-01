import os
import json
import argparse
from typing import Dict, Any, List, Optional


DEFAULT_JSON_PATH = "SHAP/Feature Importance/shap_tuples_non_prefix_14.json"
DEFAULT_OUTPUT_PATH = "SHAP/LLM Input with Definitions/llm_input_with_definitions_14.txt"


# ==========================================================
# 1) German Credit 변수 정의 사전
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
    "other_payment_plans": "Other installment plan category.",
    "housing": "Housing category.",
    "job": "Job category.",
    "property_magnitude": "Property category.",
    "purpose": "Loan purpose category.",
    "other_parties": "Other debtors or guarantors category.",
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
# 2) Canonical phrase rules
# ==========================================================
CATEGORICAL_CANONICAL_PHRASES: Dict[str, str] = {
    "checking_status_<0": "having a checking account balance below 0 DM",
    "checking_status_0<=X<200": "having a checking account balance between 0 and 200 DM",
    "checking_status_>=200": "having a checking account balance of at least 200 DM",
    "checking_status_no checking": "having no checking account",

    "savings_status_<100": "having savings below 100 DM",
    "savings_status_100<=X<500": "having savings between 100 and 500 DM",
    "savings_status_500<=X<1000": "having savings between 500 and 1000 DM",
    "savings_status_>=1000": "having savings of at least 1000 DM",
    "savings_status_no known savings": "having no known savings account",

    "employment_unemployed": "being unemployed",
    "employment_<1": "having been employed for less than 1 year",
    "employment_1<=X<4": "having been employed for between 1 and 4 years",
    "employment_4<=X<7": "having been employed for between 4 and 7 years",
    "employment_>=7": "having been employed for 7 years or more",

    "credit_history_no credits/all paid": "having no previous credits or having all previous credits paid back duly",
    "credit_history_all paid": "having all previous credits at this bank paid back duly",
    "credit_history_existing paid": "having existing credits paid back duly so far",
    "credit_history_delayed previously": "having delays in paying off previous credits",
    "credit_history_critical/other existing credit": "having a critical account or other existing credits",

    "purpose_radio/tv": "applying for credit for a radio or television",
    "purpose_used car": "applying for credit for a used car",
    "purpose_new car": "applying for credit for a new car",
    "purpose_furniture/equipment": "applying for credit for furniture or equipment",
    "purpose_business": "applying for credit for business purposes",
    "purpose_education": "applying for credit for education",
    "purpose_repairs": "applying for credit for repairs",
    "purpose_domestic appliances": "applying for credit for domestic appliances",
    "purpose_retraining": "applying for credit for retraining",
    "purpose_other": "applying for credit for other purposes",

    "housing_own": "living in owner-occupied housing",
    "housing_rent": "living in rented housing",
    "housing_for free": "living in housing free of charge",

    "personal_status_male single": "being male and single",
    "personal_status_female div/dep/mar": "being female and divorced, dependent, or married",
    "personal_status_male mar/wid": "being male and married or widowed",
    "personal_status_male div/sep": "being male and divorced or separated",

    "property_magnitude_real estate": "having real estate as property",
    "property_magnitude_life insurance": "having life insurance as property",
    "property_magnitude_car": "having a car or other non-real-estate property",
    "property_magnitude_no known property": "having no known property",

    "job_unemp/unskilled non res": "being unemployed or unskilled and non-resident",
    "job_unskilled resident": "being an unskilled resident worker",
    "job_skilled": "being a skilled employee or official",
    "job_high qualif/self emp/mgmt": "being highly qualified, self-employed, or in management",

    "other_parties_none": "having no other debtors or guarantors",
    "other_parties_co applicant": "having a co-applicant",
    "other_parties_guarantor": "having a guarantor",
}

NUMERIC_CANONICAL_TEMPLATES: Dict[str, Dict[str, str]] = {
    "duration": {
        "increase_risk": "A longer loan duration is associated with higher credit risk.",
        "decrease_risk": "A longer loan duration is associated with lower credit risk.",
    },
    "credit_amount": {
        "increase_risk": "A larger credit amount is associated with higher credit risk.",
        "decrease_risk": "A larger credit amount is associated with lower credit risk.",
    },
    "installment_commitment": {
        "increase_risk": "A higher installment rate relative to disposable income is associated with higher credit risk.",
        "decrease_risk": "A higher installment rate relative to disposable income is associated with lower credit risk.",
    },
    "age": {
        "increase_risk": "Older age is associated with higher credit risk.",
        "decrease_risk": "Older age is associated with lower credit risk.",
    },
    "existing_credits": {
        "increase_risk": "A larger number of existing credits at this bank is associated with higher credit risk.",
        "decrease_risk": "A larger number of existing credits at this bank is associated with lower credit risk.",
    },
    "number_credits": {
        "increase_risk": "A larger number of existing credits is associated with higher credit risk.",
        "decrease_risk": "A larger number of existing credits is associated with lower credit risk.",
    },
    "residence_since": {
        "increase_risk": "A longer residence duration is associated with higher credit risk.",
        "decrease_risk": "A longer residence duration is associated with lower credit risk.",
    },
    "present_residence": {
        "increase_risk": "A longer residence duration is associated with higher credit risk.",
        "decrease_risk": "A longer residence duration is associated with lower credit risk.",
    },
    "num_dependents": {
        "increase_risk": "A larger number of dependents is associated with higher credit risk.",
        "decrease_risk": "A larger number of dependents is associated with lower credit risk.",
    },
    "people_liable": {
        "increase_risk": "A larger number of people liable for maintenance is associated with higher credit risk.",
        "decrease_risk": "A larger number of people liable for maintenance is associated with lower credit risk.",
    },
}


# ==========================================================
# 3) 유틸
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


def get_feature_definition(feature_name: str) -> str:
    if feature_name in EXACT_FEATURE_DEFINITIONS:
        return EXACT_FEATURE_DEFINITIONS[feature_name]

    for prefix, definition in PREFIX_RULES:
        if feature_name == prefix:
            return definition

    for prefix, definition in PREFIX_RULES:
        if feature_name.startswith(prefix):
            return definition

    return f"No curated definition found for '{feature_name}'. Use the feature name as given without adding outside assumptions."


def get_categorical_phrase(feature_name: str) -> Optional[str]:
    if feature_name in CATEGORICAL_CANONICAL_PHRASES:
        return CATEGORICAL_CANONICAL_PHRASES[feature_name]

    for key, phrase in CATEGORICAL_CANONICAL_PHRASES.items():
        if feature_name.startswith(key):
            return phrase

    return None


def build_canonical_sentence(feature_name: str, direction: str) -> str:
    if feature_name in NUMERIC_CANONICAL_TEMPLATES:
        templates = NUMERIC_CANONICAL_TEMPLATES[feature_name]
        if direction in templates:
            return templates[direction]
        return f"The relationship between {feature_name.replace('_', ' ')} and credit risk is unclear."

    phrase = get_categorical_phrase(feature_name)
    if phrase is not None:
        if direction == "increase_risk":
            return f"{phrase.capitalize()} is associated with higher credit risk."
        if direction == "decrease_risk":
            return f"{phrase.capitalize()} is associated with lower credit risk."
        return f"The relationship between {phrase} and credit risk is unclear."

    if direction == "increase_risk":
        return f"The feature '{feature_name}' is associated with higher credit risk."
    if direction == "decrease_risk":
        return f"The feature '{feature_name}' is associated with lower credit risk."
    return f"The relationship between '{feature_name}' and credit risk is unclear."


def build_llm_input_text(data: Dict[str, Any]) -> str:
    sample_idx = data.get("sample_idx", "NA")
    pred = data.get("prediction", {})
    label = pred.get("label", "UNKNOWN")
    prob = pred.get("probability", "UNKNOWN")
    tuples: List[Dict[str, Any]] = data.get("tuples", [])

    canonical_sentences: List[str] = []

    for item in tuples:
        feature = item.get("feature", "UNKNOWN_FEATURE")
        direction = item.get("direction", "UNKNOWN")
        canonical_sentence = build_canonical_sentence(feature, direction)
        canonical_sentences.append(canonical_sentence)

    lines: List[str] = []
    lines.append("Credit Risk Prediction Instance")
    lines.append(f"Sample index: {sample_idx}")
    lines.append(f"Predicted label: {label}")
    lines.append(f"Predicted probability: {prob}")
    lines.append("")
    lines.append("feature-risk statements:")
    for sentence in canonical_sentences:
        lines.append(f"- {sentence}")
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