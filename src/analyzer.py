import os
import json
import re
from typing import Dict, Any, List, Optional
import joblib
import numpy as np

from preprocessor import preprocess_text


# Predefined domain skill dictionary for skill tagging & ATS analysis
SKILL_TAXONOMY = {
    "Programming & Core Tech": [
        "python", "java", "c++", "c#", "javascript", "typescript", "c", "ruby", "php",
        "go", "rust", "kotlin", "swift", "sql", "nosql", "bash", "r", "scala", "html", "css"
    ],
    "Frameworks & Web/Mobile": [
        "react", "angular", "vue", "node.js", "django", "flask", "fastapi", "spring boot",
        "express", "asp.net", "next.js", "flutter", "react native", "android", "ios", "graphql", "rest api"
    ],
    "Data Science & AI/ML": [
        "machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch",
        "keras", "scikit-learn", "pandas", "numpy", "tableau", "power bi", "spark", "hadoop",
        "data analysis", "data engineering", "big data", "word2vec", "transformers"
    ],
    "Cloud & DevOps": [
        "aws", "azure", "google cloud", "gcp", "docker", "kubernetes", "ci/cd", "jenkins",
        "terraform", "git", "github", "linux", "ansible", "microservices", "serverless"
    ],
    "Finance, Accounting & Business": [
        "financial analysis", "auditing", "gaap", "tax preparation", "quickbooks", "sap",
        "erp", "balance sheet", "general ledger", "financial reporting", "budgeting",
        "forecasting", "variance analysis", "compliance", "payroll", "accounts payable", "accounts receivable"
    ],
    "Creative & UI/UX": [
        "figma", "adobe photoshop", "illustrator", "indesign", "ui/ux", "user research",
        "wireframing", "prototyping", "typography", "branding", "premiere pro", "after effects"
    ],
    "Healthcare & Medical": [
        "patient care", "clinical research", "hipaa", "emr", "ehr", "pharmacology",
        "vital signs", "phlebotomy", "triage", "infection control", "cpr", "nursing"
    ],
    "Digital Marketing & Media": [
        "digital marketing", "seo", "sem", "google analytics", "campaign management",
        "email marketing", "social media", "content strategy", "keyword research",
        "conversion optimization", "advertising", "copywriting", "brand strategy",
        "performance marketing", "crm", "lead generation", "ppc", "market research"
    ],
    "Management & Leadership": [
        "agile", "scrum", "project management", "pmp", "cross-functional leadership",
        "stakeholder management", "team leadership", "risk management", "strategic planning", "operations"
    ]
}

# Domain job title suggestions per category
JOB_SUGGESTIONS = {
    "ACCOUNTANT": ["Staff Accountant", "Financial Auditor", "Senior Tax Consultant", "Cost Accountant", "Accounting Manager"],
    "ADVOCATE": ["Corporate Counsel", "Legal Analyst", "Litigation Attorney", "Compliance Officer", "Legal Associate"],
    "AGRICULTURE": ["Agronomist", "Agricultural Consultant", "Farm Operations Specialist", "Crop Production Manager"],
    "APPAREL": ["Fashion Designer", "Garment Production Coordinator", "Merchandise Manager", "Textile Specialist"],
    "ARTS": ["Visual Artist", "Creative Director", "Art Curator", "Studio Specialist", "Gallery Coordinator"],
    "AUTOMOBILE": ["Automotive Engineer", "Vehicle Diagnostics Technician", "Fleet Manager", "Automotive Design Specialist"],
    "AVIATION": ["Aviation Safety Inspector", "Flight Operations Coordinator", "Avionics Specialist", "Airline Operations Manager"],
    "BANKING": ["Commercial Banker", "Credit Analyst", "Investment Associate", "Branch Operations Manager", "Loan Officer"],
    "BPO": ["Customer Operations Specialist", "Quality Assurance Analyst", "Process Trainer", "Team Lead - Customer Experience"],
    "BUSINESS-DEVELOPMENT": ["Business Development Executive", "Account Director", "Strategic Partnerships Manager", "Growth Strategist"],
    "CHEF": ["Executive Chef", "Sous Chef", "Culinary Operations Specialist", "Pastry Specialist", "Menu Developer"],
    "CONSTRUCTION": ["Civil Site Supervisor", "Construction Project Manager", "Estimator & Surveyor", "Safety Director"],
    "CONSULTANT": ["Management Consultant", "Strategy Analyst", "Advisory Specialist", "Business Process Consultant"],
    "DESIGNER": ["UI/UX Designer", "Senior Graphic Designer", "Product Designer", "Brand Identity Specialist"],
    "DIGITAL-MEDIA": ["Digital Marketing Strategist", "Content Marketing Manager", "SEO/SEM Specialist", "Social Media Director"],
    "ENGINEERING": ["Mechanical Engineer", "Systems Design Specialist", "Process Engineer", "Quality Control Engineer"],
    "FINANCE": ["Financial Planning & Analysis (FP&A) Specialist", "Portfolio Analyst", "Treasury Specialist", "Wealth Advisor"],
    "FITNESS": ["Certified Fitness Trainer", "Exercise Physiologist", "Wellness Coordinator", "Gym Operations Manager"],
    "HEALTHCARE": ["Healthcare Administrator", "Clinical Coordinator", "Registered Nurse", "Patient Care Specialist"],
    "HR": ["Human Resources Generalist", "Talent Acquisition Specialist", "HR Business Partner (HRBP)", "People Operations Manager"],
    "INFORMATION-TECHNOLOGY": ["Software Engineer", "Full Stack Developer", "Cloud Architect", "DevOps Engineer", "Solutions Specialist"],
    "PUBLIC-RELATIONS": ["Communications Manager", "Media Relations Specialist", "PR Account Director", "Brand Reputation Officer"],
    "SALES": ["Account Executive", "Sales Director", "Enterprise Sales Specialist", "Client Relationship Manager"],
    "TEACHER": ["Secondary Educator", "Curriculum Developer", "Academic Instructor", "Instructional Designer"]
}


class ResumeAnalyzer:
    """
    Inference and intelligence engine loading the saved model and vectorizer.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.model = None
        self.vectorizer = None
        self.metadata = None
        self._load_artifacts()

    def _load_artifacts(self):
        model_path = os.path.join(self.models_dir, "resume_classifier.pkl")
        vectorizer_path = os.path.join(self.models_dir, "tfidf_vectorizer.pkl")
        meta_path = os.path.join(self.models_dir, "metadata.json")

        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Trained model artifacts not found in '{self.models_dir}'. Run src/train.py first."
            )

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """
        Extracts recognized skills and domain terms from resume text.
        """
        text_lower = text.lower()
        extracted = {}
        for domain, skills in SKILL_TAXONOMY.items():
            matched = []
            for skill in skills:
                # Regex boundary check
                pattern = r'(?:\b|_)' + re.escape(skill) + r'(?:\b|_)'
                if re.search(pattern, text_lower):
                    matched.append(skill.title() if not skill.isupper() else skill)
            if matched:
                extracted[domain] = matched
        return extracted

    def analyze(self, raw_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Full resume classification & analytics pipeline.
        """
        if not raw_text or len(raw_text.strip()) == 0:
            return {
                "error": "No text extracted from resume.",
                "status": "failed"
            }

        # 1. Preprocess
        cleaned_text = preprocess_text(raw_text)

        # 2. Vectorize
        tfidf_features = self.vectorizer.transform([cleaned_text])

        # 3. Model Prediction & Probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(tfidf_features)[0]
            classes = self.model.classes_
            sorted_indices = np.argsort(probs)[::-1]

            top_predictions = []
            for idx in sorted_indices[:top_k]:
                top_predictions.append({
                    "category": str(classes[idx]),
                    "probability": round(float(probs[idx]), 4),
                    "confidence_pct": round(float(probs[idx]) * 100, 2)
                })

            primary_category = top_predictions[0]["category"]
            confidence = top_predictions[0]["confidence_pct"]
        else:
            pred = self.model.predict(tfidf_features)[0]
            primary_category = str(pred)
            confidence = 85.0
            top_predictions = [{"category": primary_category, "probability": 0.85, "confidence_pct": 85.0}]

        # 4. Extract Skills
        detected_skills = self.extract_skills(raw_text)
        total_skills_count = sum(len(v) for v in detected_skills.values())

        # 5. Extract Trigger Keywords for the predicted category
        matched_category_keywords = []
        if self.metadata and "category_top_keywords" in self.metadata:
            cat_keywords = self.metadata["category_top_keywords"].get(primary_category, [])
            cleaned_words = set(cleaned_text.split())
            for kw in cat_keywords:
                if kw in cleaned_words:
                    matched_category_keywords.append(kw)

        # 6. Suggestions
        suggested_roles = JOB_SUGGESTIONS.get(
            primary_category,
            ["Specialist", "Senior Consultant", "Team Lead", "Project Manager"]
        )

        return {
            "status": "success",
            "primary_category": primary_category,
            "confidence_score": confidence,
            "top_predictions": top_predictions,
            "detected_skills": detected_skills,
            "total_skills_count": total_skills_count,
            "matched_category_keywords": matched_category_keywords,
            "suggested_roles": suggested_roles,
            "cleaned_text_preview": cleaned_text[:500] + ("..." if len(cleaned_text) > 500 else "")
        }
