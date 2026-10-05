# ResumeForge AI — Intelligent Resume Classification & Analytics

An end-to-end Machine Learning and NLP system that automatically classifies candidate resumes across **24 industry domains**, evaluates calibrated prediction probabilities, extracts technical competencies, and audits ATS readiness.

---

## 📁 Project Architecture

```text
├── models/                     # Dedicated folder for trained ML artifacts
│   ├── resume_classifier.pkl   # Calibrated Linear SVM classifier
│   ├── tfidf_vectorizer.pkl    # Fitted TF-IDF Vectorizer (ngram 1-2, 5000 features)
│   └── metadata.json           # Model metrics, category keywords, and training config
│
├── src/                        # Modular source code
│   ├── __init__.py             # Package initializer
│   ├── preprocessor.py         # Text cleaning, lemmatization, regex & stopword filter
│   ├── train.py                # Model training, evaluation, and serialization pipeline
│   ├── pdf_extractor.py        # Dual-engine PDF parser (pdfplumber + pypdf) & metadata
│   ├── analyzer.py             # Inference engine, skill taxonomy extractor & role matcher
│   └── app.py                  # Responsive and modern Streamlit web application
│
├── data/                       # Dataset directory with resumes by category
├── Resume.csv                  # Raw dataset (2,484 resumes, 24 classes)
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
streamlit run src/app.py
```
*The web UI will open automatically in your browser at `http://localhost:8501`.*

### 3. (Optional) Re-Train the Model
To re-train the model from `Resume.csv` and regenerate artifacts in `models/`:
```bash
python src/train.py
```

---

## ✨ Application Features

1. **PDF Upload & Instant Text Extraction**:
   - Supports any standard PDF resume up to 200MB.
   - Dual-engine fallback (`pdfplumber` and `pypdf`) for reliable extraction.
   - Includes **Built-in Sample Resumes** dropdown to test immediately without uploading your own file.

2. **Calibrated Multi-Class Prediction**:
   - Classifies across 24 job categories (Accountant, Advocate, Aviation, Banking, Engineering, HR, IT, etc.).
   - Displays prediction confidence percentage and top 5 probability distribution chart.

3. **Classification Rationale & Trigger Keywords**:
   - Shows the exact domain-indicative keywords in the resume that influenced the prediction.

4. **Multi-Domain Skill Extractor**:
   - Scans and detects skills across Programming, Web/Mobile, Data Science & AI/ML, Cloud & DevOps, Finance & Accounting, Healthcare, and Management.

5. **ATS Structure & Health Score**:
   - Checks for presence of core sections: Summary, Work Experience, Education, Skills, Certifications, and Contact info.
   - Computes an ATS Structure Score out of 100 with recommendations.

6. **Export Assessment Report**:
   - 1-click download of the complete analysis in structured **JSON** or **Markdown** format.
