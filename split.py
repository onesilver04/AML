import pandas as pd
from sklearn.model_selection import train_test_split

# 언더샘플링 끝난 파일 호출
df = pd.read_csv("german21_ohe_balanced.csv")

# 타깃: bad=1, good=0
y = df["class_bad"]
X = df.drop(columns=["class_bad", "class_good"])  # 타깃 관련 더미 컬럼 제거(중요)

# 80/20 split (클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("train:", X_train.shape, y_train.value_counts().to_dict())
print("test :", X_test.shape,  y_test.value_counts().to_dict())

# 필요하면 합쳐서 저장
train_df = pd.concat([X_train, y_train.rename("class_bad")], axis=1)
test_df  = pd.concat([X_test,  y_test.rename("class_bad")], axis=1)

train_df.to_csv("german21_train.csv", index=False)
test_df.to_csv("german21_test.csv", index=False)
