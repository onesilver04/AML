import argparse
import csv
import json
import shutil  # 파일 복사를 위해 추가
from pathlib import Path

DEFAULT_SHAP_DIR = Path("SHAP/Feature Importance")
DEFAULT_Y_TEST_PATH = Path("y_test.csv")
# 추출된 파일들이 저장될 목적지 폴더
OUTPUT_EXTRACT_DIR = Path("SHAP")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ambiguous-confidence or misclassified local SHAP outputs."
    )
    parser.add_argument("--shap-dir", default=str(DEFAULT_SHAP_DIR))
    parser.add_argument("--y-test", default=str(DEFAULT_Y_TEST_PATH))
    parser.add_argument("--low", type=float, default=0.40)
    parser.add_argument("--high", type=float, default=0.60)
    return parser.parse_args()

def load_y_test(y_test_path: Path):
    labels = []
    with y_test_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["class"]))
    return labels

def to_risk_label(value: int) -> str:
    return "BAD CREDIT RISK" if int(value) == 1 else "GOOD CREDIT RISK"

def load_and_filter_records(shap_dir: Path, y_true, low, high):
    records = []
    # 결과 저장을 위한 폴더 생성
    OUTPUT_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    
    for path in sorted(shap_dir.glob("shap_tuples_non_prefix_*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        sample_idx = int(data["sample_idx"])
        probability = float(data["prediction"]["probability"])
        pred_label = data["prediction"]["label"]
        y_true_value = y_true[sample_idx]
        true_label = to_risk_label(y_true_value)

        is_misclassified = pred_label != true_label
        is_ambiguous = low <= probability <= high
        
        # 오분류 혹은 애매한 경우에만 처리
        if is_misclassified or is_ambiguous:
            case_type = ""
            if is_misclassified and is_ambiguous: case_type = "misclassified_ambiguous"
            elif is_misclassified: case_type = "misclassified"
            else: case_type = "ambiguous_confidence"

            record = {
                "sample_idx": sample_idx,
                "file_name": path.name,
                "src_path": path,
                "case_type": case_type,
                "true_label": true_label,
                "pred_label": pred_label,
                "bad_probability": probability,
                "confidence_margin": abs(probability - 0.5),
                "is_misclassified": is_misclassified,
                "is_ambiguous": is_ambiguous,
                "top_features": [item["feature"] for item in data.get("tuples", [])],
            }
            records.append(record)
            
            # 🔥 핵심: 조건에 맞는 파일만 새로운 폴더로 복사
            shutil.copy(path, OUTPUT_EXTRACT_DIR / path.name)
            
    return records

def write_summary_csv(path: Path, records):
    fieldnames = ["sample_idx", "case_type", "true_label", "pred_label", "bad_probability", "confidence_margin", "top_features", "file_name"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for record in records:
            row = record.copy()
            row["top_features"] = " | ".join(record["top_features"])
            writer.writerow(row)

def main():
    args = parse_args()
    shap_dir = Path(args.shap_dir)
    y_test_path = Path(args.y_test)

    y_true = load_y_test(y_test_path)
    # 파일 로드와 동시에 필터링 및 복사 수행
    interesting_records = load_and_filter_records(shap_dir, y_true, args.low, args.high)

    # 마진 순으로 정렬 (더 애매한 것부터)
    interesting_records.sort(key=lambda r: r["confidence_margin"])

    # CSV 요약본 저장
    write_summary_csv(OUTPUT_EXTRACT_DIR / "extracted_cases_summary.csv", interesting_records)

    print(f"✅ 분석 완료!")
    print(f"📂 추출된 파일 저장소: {OUTPUT_EXTRACT_DIR}")
    print(f"📊 총 추출된 문제적 샘플: {len(interesting_records)}개")
    
    for r in interesting_records:
        print(f"[{r['case_type']}] Index {r['sample_idx']}: Prob={r['bad_probability']:.4f}")

if __name__ == "__main__":
    main()
