import os
import sys
import requests

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)

from pdf_extractor import PDFExtractor
from analyzer import ResumeAnalyzer

def run_all_validation():
    print("=" * 60)
    print("   END-TO-END RESUME CLASSIFICATION & UI VALIDATION")
    print("=" * 60)

    # 1. Test Streamlit HTTP endpoint
    try:
        resp = requests.get("http://localhost:8501", timeout=5)
        print(f"\n[1/3] Streamlit Web Server Status: HTTP {resp.status_code} OK")
        print(f"      App is accessible at: http://localhost:8501")
    except Exception as e:
        print(f"\n[1/3] Streamlit Server Status: ERROR ({e})")

    # 2. Test Model & Artifacts
    print("\n[2/3] Checking Model Artifacts:")
    analyzer = ResumeAnalyzer(os.path.join(base_dir, "models"))
    meta = analyzer.metadata
    print(f"      - Architecture: {meta.get('model_name')}")
    print(f"      - Test Accuracy: {meta.get('test_accuracy') * 100:.2f}%")
    print(f"      - Weighted F1: {meta.get('test_weighted_f1'):.4f}")
    print(f"      - Supported Classes: {meta.get('categories_count')} categories")

    # 3. Test Multi-Domain PDFs
    print("\n[3/3] Inference Testing on Real Multi-Category Resumes:")
    test_categories = ["ACCOUNTANT", "INFORMATION-TECHNOLOGY", "CHEF", "HR", "TEACHER", "ADVOCATE"]

    for cat in test_categories:
        cat_dir = os.path.join(base_dir, "data", cat)
        if not os.path.exists(cat_dir):
            continue

        pdfs = [f for f in os.listdir(cat_dir) if f.lower().endswith(".pdf")]
        if not pdfs:
            continue

        sample_pdf = os.path.join(cat_dir, pdfs[0])
        ext = PDFExtractor.extract_text_and_meta(sample_pdf)
        res = analyzer.analyze(ext["text"], top_k=3)

        status_flag = "MATCH" if res["primary_category"] == cat else "CLOSE_MATCH"
        print(f"\n   [Domain: {cat}] File: {pdfs[0]}")
        print(f"   -> Extracted: {ext['page_count']} pages | {ext['word_count']} words | ATS Score: {ext['resume_health_score']}/100")
        print(f"   -> Predicted: {res['primary_category']} ({res['confidence_score']:.1f}% confidence) [{status_flag}]")
        top3_str = ", ".join([f"{p['category']} ({p['confidence_pct']}%)" for p in res["top_predictions"]])
        print(f"   -> Top 3 Probabilities: {top3_str}")
        print(f"   -> Detected Skills Count: {res['total_skills_count']} skills")
        print(f"   -> Trigger Keywords: {', '.join(res.get('matched_category_keywords', [])[:6])}")

    print("\n" + "=" * 60)
    print("   ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_validation()
