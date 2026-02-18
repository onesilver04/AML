import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize_iqr_outliers(X_df: pd.DataFrame, num_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """
    IQR(Interquartile Range) 기반 이상치 요약.
    각 수치형 컬럼별로 Q1, Q3, IQR, 하한/상한, 이상치 개수/비율을 계산하고,
    행 단위로는 "어느 수치형 컬럼에서든 이상치에 걸린 행인지"를 나타내는 시리즈를 반환.
    """
    if len(num_cols) == 0:
        empty = pd.DataFrame(
            columns=["col", "q1", "q3", "iqr", "lower", "upper", "outlier_count", "outlier_pct"]
        )
        return empty, pd.Series(dtype=bool)

    rows = []
    any_outlier = pd.Series(False, index=X_df.index)

    for col in num_cols:
        s = pd.to_numeric(X_df[col], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (s < lower) | (s > upper)
        any_outlier |= mask.fillna(False)

        rows.append(
            {
                "col": col,
                "q1": float(q1) if pd.notna(q1) else np.nan,
                "q3": float(q3) if pd.notna(q3) else np.nan,
                "iqr": float(iqr) if pd.notna(iqr) else np.nan,
                "lower": float(lower) if pd.notna(lower) else np.nan,
                "upper": float(upper) if pd.notna(upper) else np.nan,
                "outlier_count": int(mask.sum(skipna=True)),
                "outlier_pct": float(mask.mean(skipna=True) * 100.0),
            }
        )

    report = (
        pd.DataFrame(rows)
        .sort_values(["outlier_count", "outlier_pct"], ascending=False)
        .reset_index(drop=True)
    )
    return report, any_outlier


def main() -> None:
    # Matplotlib 설정 (Cursor/서버 환경에서도 에러 없이 저장되도록)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
    matplotlib.use("Agg")

    data_path = "german21.csv"
    df = pd.read_csv(data_path)

    print(f"데이터 shape: {df.shape}")

    # 수치형 컬럼만 선택
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    print(f"\n수치형 컬럼 ({len(numeric_cols)}개): {numeric_cols}")

    # IQR 기반 이상치 요약
    outlier_report, any_outlier = summarize_iqr_outliers(df, numeric_cols)
    if len(numeric_cols) > 0:
        print("\n[IQR 기반 이상치 요약 - 상위 10개 컬럼]")
        print(
            outlier_report.head(10)[
                ["col", "outlier_count", "outlier_pct", "lower", "upper"]
            ].to_string(index=False)
        )
        print(
            f"\n수치형 기준으로 '어떤 컬럼이든 이상치'를 포함하는 행: "
            f"{int(any_outlier.sum())} / {len(df)}"
        )
    else:
        print("\n수치형 컬럼이 없어 IQR 기반 이상치 탐지가 불가능합니다.")

    # 1) 전체 수치형 박스플롯 (컬럼별 나란히)
    if len(numeric_cols) > 0:
        plt.figure(figsize=(14, 6))
        df[numeric_cols].boxplot(rot=45)
        plt.title("Numeric Features Boxplot (All)")
        plt.tight_layout()
        plt.savefig("boxplot_numeric_all.png", dpi=160)
        plt.close()

    # 2) 컬럼별 개별 박스플롯 (그리드)
    if len(numeric_cols) > 0:
        import math

        n = len(numeric_cols)
        cols = 3
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
        axes = axes.flatten()

        for i, c in enumerate(numeric_cols):
            axes[i].boxplot(df[c].dropna(), vert=True)
            axes[i].set_title(c)

        # 남는 subplot은 비활성화
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig("boxplot_numeric_grid.png", dpi=160)
        plt.close()

    print("\n박스플롯 파일 생성:")
    print("- boxplot_numeric_all.png")
    if len(numeric_cols) > 0:
        print("- boxplot_numeric_grid.png")


if __name__ == "__main__":
    main()

