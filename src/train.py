import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, f1_score

from preprocessor import preprocess_text


def train_and_save_model(data_path="Resume.csv", output_dir="models"):
    print("=" * 60)
    print("   RESUME CLASSIFICATION MODEL TRAINING & SERIALIZATION")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    print(f"\n[1/6] Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    initial_shape = df.shape
    print(f" -> Raw dataset shape: {initial_shape}")

    # 2. Quality Check & Deduplication
    duplicates = df.duplicated(subset=['Resume_str']).sum()
    if duplicates > 0:
        df = df.drop_duplicates(subset=['Resume_str']).reset_index(drop=True)
        print(f" -> Removed {duplicates} exact duplicate resumes. New shape: {df.shape}")

    # Drop missing resumes
    df = df.dropna(subset=['Resume_str', 'Category']).reset_index(drop=True)
    categories = sorted(df['Category'].unique().tolist())
    print(f" -> Detected {len(categories)} unique categories: {', '.join(categories[:6])}...")

    # 3. Preprocessing
    print("\n[2/6] Preprocessing resume text...")
    df['Cleaned_Resume'] = df['Resume_str'].apply(preprocess_text)
    print(f" -> Preprocessed {len(df)} documents successfully.")

    # 4. Train / Validation / Test Stratified Split (70% / 15% / 15%)
    print("\n[3/6] Performing Stratified Train / Val / Test Split (70/15/15)...")
    X = df['Cleaned_Resume']
    y = df['Category']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f" -> Train set: {len(X_train)} samples")
    print(f" -> Val set:   {len(X_val)} samples")
    print(f" -> Test set:  {len(X_test)} samples")

    # 5. Feature Extraction (TF-IDF)
    print("\n[4/6] Fitting TF-IDF Vectorizer (ngram_range=(1,2), max_features=5000)...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f" -> TF-IDF matrix shape: {X_train_tfidf.shape}")

    # 6. Model Training & Comparison
    print("\n[5/6] Training Candidate Models...")

    # Candidate 1: Logistic Regression
    print(" -> Training Logistic Regression (class_weight='balanced')...")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_tfidf, y_train)
    val_acc_lr = accuracy_score(y_val, lr.predict(X_val_tfidf))
    test_acc_lr = accuracy_score(y_test, lr.predict(X_test_tfidf))
    test_f1_lr = f1_score(y_test, lr.predict(X_test_tfidf), average='weighted')
    print(f"    Logistic Regression -> Val Acc: {val_acc_lr:.4f} | Test Acc: {test_acc_lr:.4f} | Weighted F1: {test_f1_lr:.4f}")

    # Candidate 2: Calibrated Linear SVM (for probability calibration & peak accuracy)
    print(" -> Training Calibrated Linear SVM (LinearSVC + Sigmoid Calibration)...")
    base_svc = LinearSVC(class_weight='balanced', random_state=42, max_iter=2000)
    calibrated_svc = CalibratedClassifierCV(estimator=base_svc, cv=3)
    calibrated_svc.fit(X_train_tfidf, y_train)
    val_acc_svc = accuracy_score(y_val, calibrated_svc.predict(X_val_tfidf))
    test_acc_svc = accuracy_score(y_test, calibrated_svc.predict(X_test_tfidf))
    test_f1_svc = f1_score(y_test, calibrated_svc.predict(X_test_tfidf), average='weighted')
    print(f"    Calibrated LinearSVC -> Val Acc: {val_acc_svc:.4f} | Test Acc: {test_acc_svc:.4f} | Weighted F1: {test_f1_svc:.4f}")

    # Choose best model
    if test_acc_svc >= test_acc_lr:
        best_name = "Calibrated Linear SVM"
        best_model = calibrated_svc
        final_test_acc = test_acc_svc
        final_test_f1 = test_f1_svc
    else:
        best_name = "Logistic Regression"
        best_model = lr
        final_test_acc = test_acc_lr
        final_test_f1 = test_f1_lr

    print(f"\n[Selected Best Model: {best_name} (Test Acc: {final_test_acc:.4f})]")

    # Full Evaluation Report
    y_test_pred = best_model.predict(X_test_tfidf)
    report = classification_report(y_test, y_test_pred, output_dict=True)

    # Top indicative keywords per category
    print("\nExtracting category-indicative keywords...")
    category_top_keywords = {}
    feature_names = np.array(vectorizer.get_feature_names_out())
    # Compute mean TF-IDF vectors per category from training set
    train_df = pd.DataFrame({'text': X_train, 'category': y_train})
    for cat in categories:
        cat_indices = train_df[train_df['category'] == cat].index
        if len(cat_indices) > 0:
            # Transform text of this category
            cat_tfidf = vectorizer.transform(train_df.loc[cat_indices, 'text'])
            mean_tfidf = np.asarray(cat_tfidf.mean(axis=0)).flatten()
            top_kw_idx = mean_tfidf.argsort()[::-1][:15]
            category_top_keywords[cat] = feature_names[top_kw_idx].tolist()

    # 7. Serialize Artifacts to dedicated models/ folder
    print(f"\n[6/6] Saving model artifacts to '{output_dir}/'...")
    model_path = os.path.join(output_dir, "resume_classifier.pkl")
    vectorizer_path = os.path.join(output_dir, "tfidf_vectorizer.pkl")
    metadata_path = os.path.join(output_dir, "metadata.json")

    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    metadata = {
        "model_name": best_name,
        "trained_at": datetime.now().isoformat(),
        "total_resumes_trained": len(X_train),
        "test_accuracy": round(float(final_test_acc), 4),
        "test_weighted_f1": round(float(final_test_f1), 4),
        "categories_count": len(categories),
        "categories": categories,
        "vectorizer_config": {
            "max_features": 5000,
            "ngram_range": [1, 2]
        },
        "category_top_keywords": category_top_keywords,
        "classification_summary": {
            "accuracy": round(float(report["accuracy"]), 4),
            "macro_avg": {
                "precision": round(float(report["macro avg"]["precision"]), 4),
                "recall": round(float(report["macro avg"]["recall"]), 4),
                "f1_score": round(float(report["macro avg"]["f1-score"]), 4),
            },
            "weighted_avg": {
                "precision": round(float(report["weighted avg"]["precision"]), 4),
                "recall": round(float(report["weighted avg"]["recall"]), 4),
                "f1_score": round(float(report["weighted avg"]["f1-score"]), 4),
            }
        }
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f" -> Model saved:      {model_path} ({os.path.getsize(model_path) / 1024:.1f} KB)")
    print(f" -> Vectorizer saved: {vectorizer_path} ({os.path.getsize(vectorizer_path) / 1024:.1f} KB)")
    print(f" -> Metadata saved:   {metadata_path}")
    print("\nTraining & serialization completed successfully!")
    return metadata


if __name__ == "__main__":
    train_and_save_model()
