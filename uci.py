import pandas as pd

# 1. 데이터 로드
path = "statlog+german+credit+data/german.data-numeric"  # 실제 파일 경로로 수정
df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")

# 2. 표준 컬럼명 정의 (Statlog / 논문 / Kaggle 관행 기준)
column_names = [
    "checking_status",          # 1
    "duration_months",          # 2
    "credit_history",           # 3
    "purpose",                  # 4
    "credit_amount",            # 5
    "savings_status",           # 6
    "employment_since",         # 7
    "installment_rate",         # 8
    "personal_status_sex",      # 9
    "other_debtors",            # 10
    "residence_since",          # 11
    "property",                 # 12
    "age",                      # 13
    "other_installment_plans",  # 14
    "housing",                  # 15
    "existing_credits",         # 16
    "job",                      # 17
    "num_dependents",           # 18
    "telephone",                # 19
    "foreign_worker",           # 20

    # numeric 버전에서 추가된 파생/이진 변수들
    "has_checking_account",     # 21
    "has_savings_account",      # 22
    "is_employed",              # 23
    "is_single",                # 24

    # target
    "credit_risk"               # 25 (1=good, 2=bad)
]

# 3. 컬럼 수 검증
assert df.shape[1] == len(column_names), "컬럼 개수가 맞지 않습니다."

# 4. 컬럼명 적용
df.columns = column_names

# 5. 타겟 값 정리 (ML / AML 표준)
# 0 = good, 1 = bad
df["credit_risk"] = df["credit_risk"].map({1: 0, 2: 1})

# 6. CSV로 저장
output_path = "german_credit_numeric.csv"
df.to_csv(output_path, index=False)

print("저장 완료:", output_path)
print("shape:", df.shape)
print(df.head())
