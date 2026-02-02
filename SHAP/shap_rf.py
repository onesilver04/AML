import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# =========================
# 0) 저장해둔 모델/데이터 로드
# =========================
rf = joblib.load("rf_model.joblib")

X_test = pd.read_csv("X_test.csv")
# y_test는 SHAP 계산에 필수는 아니지만, 필요하면 로드 가능
# y_test = pd.read_csv("y_test.csv").squeeze("columns")

print("Loaded:", type(rf))
print("X_test shape:", X_test.shape)

# =========================
# 1) TreeSHAP 계산
# =========================
explainer = shap.TreeExplainer(rf)
shap_exp = explainer(X_test)   # shap.Explanation

# =========================
# 2) 시각화
# =========================
if len(shap_exp.values.shape) == 3:
    shap_exp = shap_exp[:, :, 1]
# (A) Summary plot - beeswarm
shap.plots.beeswarm(shap_exp, max_display=20)
plt.show()

# (B) Summary plot - bar (mean |SHAP|)
shap.plots.bar(shap_exp, max_display=20)
plt.show()

# (C) Local explanation - waterfall (예: 첫 번째 샘플)
shap.plots.waterfall(shap_exp[0], max_display=20)
plt.show()

# =========================
# 3) 중요도 표 저장 (mean abs SHAP)
# =========================
sv = shap_exp.values
mean_abs_shap = np.mean(np.abs(sv), axis=0)

imp_df = (
    pd.DataFrame({"feature": X_test.columns, "mean_abs_shap": mean_abs_shap})
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)

print("\nTop 20 SHAP feature importance:")
print(imp_df.head(20).to_string(index=False))

imp_df.to_csv("rf_shap_feature_importance.csv", index=False)
print("\nSaved: rf_shap_feature_importance.csv")
