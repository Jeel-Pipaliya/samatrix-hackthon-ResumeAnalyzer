# 📄 ResumeClassifier — Intelligent Resume Classification & Analytics Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![NLP](https://img.shields.io/badge/NLP-NLTK%20%2B%20TF--IDF-green.svg)](https://www.nltk.org/)
[![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)](#)

> An end-to-end Machine Learning and NLP system that automatically classifies candidate resumes across **24 industry domains**, evaluates calibrated prediction probabilities, extracts technical competencies, and audits ATS readiness.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture (Simple Flow)](#-system-architecture-simple-flow)
- [Supported Job Categories (24 Classes)](#-supported-job-categories-24-classes)
- [Model Performance & Benchmarks](#-model-performance--benchmarks)
- [Project Directory Structure](#-project-directory-structure)
- [Quick Start Guide](#-quick-start-guide)
- [Core Source Modules](#-core-source-modules)
- [Tech Stack](#-tech-stack)
- [Verification & Testing](#-verification--testing)

---

## 🔍 Overview

Recruiters and Applicant Tracking Systems (ATS) process large volumes of unstructured candidate resumes daily. Manual screening is slow and prone to inconsistency. 

**ResumeClassifier** automates this workflow by extracting text from diverse file formats (PDF, DOCX, Images, and raw text), preprocessing it with NLP pipelines, classifying it into one of **24 professional domains** using a **Calibrated Linear Support Vector Classifier**, and generating an interactive assessment report complete with **ATS health auditing** and **skill taxonomy breakdown**.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| 🎯 **24-Domain Classification** | Accurately classifies candidate profiles across 24 distinct industry sectors (IT, Finance, Healthcare, Aviation, etc.). |
| 📊 **Calibrated Confidence** | Uses Platt Sigmoid probability calibration (`CalibratedClassifierCV`) to output true posterior confidence percentages. |
| 📈 **Top-5 Probability Horizon** | Renders an interactive probability distribution chart displaying the top 5 matching career domains. |
| 🔍 **Explainable AI (XAI)** | Highlights domain-indicative trigger keywords found in the resume that drove the model's prediction. |
| 📄 **Multi-Engine Document Parsing** | Fault-tolerant extraction waterfall (`pdfplumber` ➔ `pypdfium2` ➔ `pypdf` ➔ `winocr` ➔ `python-docx`) ensuring zero document parse failures. |
| 🛠️ **Multi-Domain Skill Extractor** | Automatically detects and organizes candidate competencies across 8 vertical skill taxonomies. |
| 📋 **ATS Structure & Health Score** | Checks presence of essential resume sections, contact channels, and word count, computing a 0–100 ATS score. |
| 🧪 **Built-in Sample Testing** | Test immediately with pre-loaded candidate resumes directly from a dropdown in the UI without uploading your own file. |
| 💾 **1-Click Export** | Download comprehensive diagnostic reports in structured **JSON** or **Markdown** format. |

---

## 🏗️ System Architecture (Simple Flow)

The application follows a clean, decoupled 3-stage pipeline:

```text
               [ Candidate Resume ]
     (PDF / DOCX / Image / Pasted Raw Text)
                       │
                       ▼
    ┌─────────────────────────────────────┐
    │     Multi-Engine Text Extractor     │
    │  (pdfplumber / pypdf / OCR / DOCX)  │
    └──────────────────┬──────────────────┘
                       │
             Extracted Resume Text
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌──────────────────┐        ┌──────────────────┐
│  NLP & Inference │        │  ATS & Taxonomy  │
│  - Text Clean    │        │  - Contact Check │
│  - Lemmatization │        │  - Section Audit │
│  - TF-IDF Matrix │        │  - Skill Match   │
│  - Calibrated SVM│        │  - ATS Score/100 │
└────────┬─────────┘        └────────┬─────────┘
         │                           │
         └─────────────┬─────────────┘
                       │
                       ▼
        ┌─────────────────────────────┐
        │   Streamlit Web Interface   │
        │ - Primary Domain & Conf. %  │
        │ - Top 5 Probabilities Chart │
        │ - ATS Health Checklist      │
        │ - Skill Pills & Export Rep. │
        └─────────────────────────────┘
```

---

## 🏷️ Supported Job Categories (24 Classes)

The model is trained to recognize 24 diverse industry categories:

| Domains 1–6 | Domains 7–12 | Domains 13–18 | Domains 19–24 |
| :--- | :--- | :--- | :--- |
| 1. `ACCOUNTANT` | 7. `AVIATION` | 13. `CONSULTANT` | 19. `HEALTHCARE` |
| 2. `ADVOCATE` | 8. `BANKING` | 14. `DESIGNER` | 20. `HR` |
| 3. `AGRICULTURE` | 9. `BPO` | 15. `DIGITAL-MEDIA` | 21. `INFORMATION-TECHNOLOGY` |
| 4. `APPAREL` | 10. `BUSINESS-DEV` | 16. `ENGINEERING` | 22. `PUBLIC-RELATIONS` |
| 5. `ARTS` | 11. `CHEF` | 17. `FINANCE` | 23. `SALES` |
| 6. `AUTOMOBILE` | 12. `CONSTRUCTION`| 18. `FITNESS` | 24. `TEACHER` |

---

## 📊 Model Performance & Benchmarks

The classifier was evaluated on a held-out test set of **373 unseen candidate resumes** across all 24 categories:

| Model Architecture | Validation Accuracy | Test Top-1 Accuracy | Test Weighted F1 | Test Macro F1 | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 63.71% | 68.36% | 0.6666 | 0.6412 | ~8 ms |
| **Calibrated Linear SVM (Selected)** | **68.01%** | **73.19%** | **0.7225** | **0.6905** | **~10 ms** |

### Top-K Prediction Accuracy

In multi-class recruitment screening, a candidate often has cross-functional experience. ResumeClassifier achieves exceptional top-K accuracy:

- **Top-1 Accuracy:** `73.19%`
- **Top-2 Accuracy:** `82.41%`
- **Top-3 Accuracy:** `89.54%`
- **Top-5 Accuracy:** `94.10%`

*(94.10% of candidates have their actual domain within the model's top 5 predictions).*

---

## 📁 Project Directory Structure

```text
samatrix-hackthon-ResumeAnalyzer/
│
├── models/                     # Serialized production ML artifacts
│   ├── resume_classifier.pkl   # Trained Calibrated Linear SVM model (2.8 MB)
│   ├── tfidf_vectorizer.pkl    # Fitted TF-IDF Vectorizer (5,000 features, 198 KB)
│   └── metadata.json           # Model metrics, label mappings, and top keywords
│
├── src/                        # Modular application code
│   ├── __init__.py             # Package initializer
│   ├── preprocessor.py         # Text cleaning, lemmatization & stopword removal
│   ├── train.py                # Model training, calibration & serialization pipeline
│   ├── pdf_extractor.py        # Multi-engine document parser & ATS layout analyzer
│   ├── analyzer.py             # Inference engine, skill taxonomy & recommendations
│   ├── app.py                  # Responsive Streamlit web interface
│   └── validate_all.py         # Automated end-to-end integration test suite
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & user guide
├── Summary.md                  # Comprehensive technical report & architectural deep dive
└── Resume_Classification.ipynb # Jupyter notebook for exploratory data analysis
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python **3.10** or higher
- `pip` package manager

### 1. Clone the Repository
```bash
git clone https://github.com/Jeel-Pipaliya/samatrix-hackthon-ResumeAnalyzer.git
cd samatrix-hackthon-ResumeAnalyzer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
streamlit run src/app.py
```
The application will launch in your default web browser at:
👉 **`http://localhost:8501`**

### 4. (Optional) Run the Automated Test Suite
To verify the model, inference pipeline, and end-to-end functionality:
```bash
python src/validate_all.py
```

### 5. (Optional) Retrain the Model
If you provide `Resume.csv` in the root folder, you can retrain the model and regenerate serialized artifacts:
```bash
python src/train.py
```

---

## 🧩 Core Source Modules

- **[`src/app.py`](src/app.py):** Main Streamlit dashboard featuring file uploaders, real-time analytics, confidence meters, ATS health audits, interactive charts, and export buttons.
- **[`src/analyzer.py`](src/analyzer.py):** Core inference engine. Computes calibrated probabilities, extracts domain keywords, performs skill taxonomy matching across 8 categories, and suggests relevant job titles.
- **[`src/pdf_extractor.py`](src/pdf_extractor.py):** Robust document ingestion module. Implements a multi-engine fallback mechanism (`pdfplumber` ➔ `pypdfium2` ➔ `pypdf` ➔ `winocr` ➔ `python-docx`) and analyzes ATS document structure.
- **[`src/preprocessor.py`](src/preprocessor.py):** High-speed NLP text normalization pipeline incorporating regex cleaning, URL/email stripping, C++/C# symbol preservation, and NLTK WordNet lemmatization.
- **[`src/train.py`](src/train.py):** Standalone training pipeline. Handles stratified data splitting, TF-IDF feature extraction, model calibration, cross-validation, and artifact serialization.

---

## ⚙️ Tech Stack

- **Core Language:** Python 3.10+
- **Machine Learning:** Scikit-Learn (`LinearSVC`, `CalibratedClassifierCV`, `TfidfVectorizer`, `LogisticRegression`)
- **Natural Language Processing:** NLTK (`WordNetLemmatizer`, `stopwords`)
- **Document Parsing:** `pdfplumber`, `pypdfium2`, `pypdf`, `python-docx`, `winocr`
- **Visualization:** Altair, Streamlit Native Components
- **Model Serialization:** Joblib
- **Web Framework:** Streamlit 1.30+

---

## 🧪 Verification & Testing

To ensure reliability across environments, execute the test script:

```bash
python src/validate_all.py
```

**Expected Output:**
```text
============================================================
   END-TO-END RESUME CLASSIFICATION & UI VALIDATION
============================================================

[1/3] Streamlit Web Server Status: HTTP 200 OK
      App is accessible at: http://localhost:8501

[2/3] Checking Model Artifacts:
      - Architecture: Calibrated Linear SVM
      - Test Accuracy: 73.19%
      - Weighted F1: 0.7225
      - Supported Classes: 24 categories

[3/3] Inference Testing on Real Multi-Category Resumes:
   ... All predictions validated successfully!

============================================================
   ALL VERIFICATIONS COMPLETED SUCCESSFULLY!
============================================================
```

---

## 📄 License & Attribution

Developed for the **Samatrix Hackathon 2026**.  
Designed with clean code principles, modular software architecture, and production-grade ML engineering.
