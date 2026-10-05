import os
import sys
import json
import io
import pandas as pd
import altair as alt
import streamlit as st

# Ensure src is on sys.path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from pdf_extractor import PDFExtractor
from analyzer import ResumeAnalyzer, SKILL_TAXONOMY, JOB_SUGGESTIONS

# Configuration
MODELS_DIR = os.path.join(base_dir, "models")
DATA_DIR = os.path.join(base_dir, "data")

# Category Icons Map for high visual appeal
CATEGORY_ICONS = {
    "ACCOUNTANT": "💰",
    "ADVOCATE": "⚖️",
    "AGRICULTURE": "🌾",
    "APPAREL": "👗",
    "ARTS": "🎨",
    "AUTOMOBILE": "🚗",
    "AVIATION": "✈️",
    "BANKING": "🏦",
    "BPO": "📞",
    "BUSINESS-DEVELOPMENT": "📈",
    "CHEF": "👨‍🍳",
    "CONSTRUCTION": "🏗️",
    "CONSULTANT": "💼",
    "DESIGNER": "✨",
    "DIGITAL-MEDIA": "📱",
    "ENGINEERING": "⚙️",
    "FINANCE": "📊",
    "FITNESS": "🏋️",
    "HEALTHCARE": "🩺",
    "HR": "👥",
    "INFORMATION-TECHNOLOGY": "💻",
    "PUBLIC-RELATIONS": "📢",
    "SALES": "🤝",
    "TEACHER": "📚"
}


# Streamlit Page Config
st.set_page_config(
    page_title="ResumeClassifier | Intelligent Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Gradient Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4338ca 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
        color: #f8fafc;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #c7d2fe;
        max-width: 820px;
        line-height: 1.5;
        margin-bottom: 1.2rem;
    }
    
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }
    
    .hero-badge {
        background: rgba(255, 255, 255, 0.14);
        backdrop-filter: blur(8px);
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #e0e7ff;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Glassmorphic Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.4rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        border-color: #cbd5e1;
    }
    
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
    
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    
    .metric-footer {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.4rem;
    }
    
    /* Prediction Hero Card */
    .prediction-box {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        color: white;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.4);
        margin-bottom: 1.5rem;
    }
    
    .prediction-header {
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #d1fae5;
    }
    
    .prediction-role {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.2rem 0;
        letter-spacing: -0.02em;
    }
    
    /* Skill Pill Badges */
    .skill-pill {
        display: inline-block;
        background: #eef2ff;
        color: #4338ca;
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
        border: 1px solid #c7d2fe;
    }
    
    .keyword-pill {
        display: inline-block;
        background: #f0fdf4;
        color: #15803d;
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
        border: 1px solid #bbf7d0;
    }
    
    .role-pill {
        display: inline-block;
        background: #f8fafc;
        color: #334155;
        padding: 0.4rem 0.9rem;
        border-radius: 10px;
        font-size: 0.88rem;
        font-weight: 600;
        margin: 0.3rem 0.3rem 0.3rem 0;
        border: 1px solid #e2e8f0;
    }
    
    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_analyzer():
    """Load and cache the trained ML model and vectorizer."""
    try:
        analyzer = ResumeAnalyzer(MODELS_DIR)
        return analyzer
    except Exception as e:
        st.error(f"⚠️ Failed to load model from '{MODELS_DIR}': {e}")
        return None


def get_available_samples():
    """Discover available sample PDFs grouped by category in the data/ directory."""
    samples = {}
    if os.path.exists(DATA_DIR):
        for item in sorted(os.listdir(DATA_DIR)):
            sub_path = os.path.join(DATA_DIR, item)
            if os.path.isdir(sub_path):
                pdfs = [f for f in os.listdir(sub_path) if f.lower().endswith('.pdf')]
                if pdfs:
                    samples[item] = [os.path.join(sub_path, f) for f in pdfs[:5]]
    return samples


def main():
    # 1. Initialize Backend Analyzer
    analyzer = load_analyzer()
    if not analyzer:
        st.warning("Please run `python src/train.py` to train and serialize the model into `models/` folder first.")
        return

    metadata = analyzer.metadata or {}
    model_name = metadata.get("model_name", "Calibrated Linear SVM")
    test_accuracy = metadata.get("test_accuracy", 0.7319) * 100
    total_categories = metadata.get("categories_count", 24)

    # 2. Sidebar Navigation & Controls
    with st.sidebar:
        st.markdown("### 📄 **ResumeClassifier**")
        st.caption("Next-Generation Multi-Class Resume Classification & ATS Intelligence Engine")
        
        # Model Status Box
        st.markdown(f"""
        <div style="background:#f1f5f9; padding:1rem; border-radius:12px; border:1px solid #e2e8f0; margin-bottom:1.2rem;">
            <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#64748b;">Active Classifier</div>
            <div style="font-size:1.05rem; font-weight:700; color:#0f172a; margin:0.2rem 0;">{model_name}</div>
            <div style="display:flex; justify-content:space-between; font-size:0.82rem; color:#475569; margin-top:0.4rem;">
                <span>🎯 Test Accuracy:</span>
                <strong style="color:#059669;">{test_accuracy:.1f}%</strong>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.82rem; color:#475569; margin-top:0.2rem;">
                <span>🏷️ Categories:</span>
                <strong>{total_categories} Industries</strong>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.82rem; color:#475569; margin-top:0.2rem;">
                <span>⚡ Representation:</span>
                <strong>TF-IDF (1-2 N-grams)</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📥 Select Input Source")
        input_source = st.radio(
            "Choose resume input mode:",
            ["📁 Upload Resume File (PDF, Word, Image)", "✍️ Paste Resume Text Directly", "📚 Built-in Sample Resumes"],
            index=0,
            label_visibility="collapsed"
        )

        selected_sample_path = None
        sample_dict = get_available_samples()

        if input_source == "📚 Built-in Sample Resumes":
            st.markdown("##### 🔍 Choose a Sample")
            chosen_cat = st.selectbox("Industry Category:", list(sample_dict.keys()), index=0)
            sample_files = sample_dict.get(chosen_cat, [])
            if sample_files:
                sample_labels = [os.path.basename(p) for p in sample_files]
                chosen_file_name = st.selectbox("Sample Resume PDF:", sample_labels)
                selected_sample_path = os.path.join(DATA_DIR, chosen_cat, chosen_file_name)
            else:
                st.info("No sample PDFs found for this category.")

        st.markdown("---")
        with st.expander("ℹ️ Supported 24 Categories", expanded=False):
            categories = metadata.get("categories", sorted(list(CATEGORY_ICONS.keys())))
            for cat in categories:
                icon = CATEGORY_ICONS.get(cat, "📁")
                st.write(f"{icon} **{cat.replace('-', ' ')}**")

        st.markdown("---")
        st.caption("Built with PyPDF, Scikit-Learn & Streamlit | Hackathon 2026")

    # 3. Main Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">📄 Resume Classification & Intelligence System</div>
        <div class="hero-subtitle">
            Upload any candidate resume in PDF, Word, Image, or plain text to instantly categorize it across 24 industry domains, 
            evaluate probability distributions, detect extracted technical competencies, and audit ATS readiness.
        </div>
        <div class="badge-container">
            <span class="hero-badge">⚡ Multi-Engine PDF & OCR Extraction</span>
            <span class="hero-badge">🎯 Calibrated Probability Breakdown</span>
            <span class="hero-badge">🧠 Multi-Domain Skill Extractor</span>
            <span class="hero-badge">📊 ATS Structure Score</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4. Upload or Load Resume
    file_bytes = None
    file_name = None

    if input_source == "📁 Upload Resume File (PDF, Word, Image)":
        uploaded_file = st.file_uploader(
            "Drop your resume here (PDF, DOCX, PNG, JPG, TXT)",
            type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
            help="Upload a candidate resume in PDF, Word, or image format (up to 200MB)"
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_name = uploaded_file.name

    elif input_source == "✍️ Paste Resume Text Directly":
        st.markdown("##### ✍️ Paste Candidate Resume Text")
        pasted_text = st.text_area(
            "Paste full resume content (summary, experience, skills, education):",
            height=260,
            placeholder="Paste text here... e.g. Digital Marketing Specialist with 5 years experience in SEO, SEM, Google Analytics..."
        )
        if pasted_text and len(pasted_text.strip()) >= 20:
            file_bytes = pasted_text.strip().encode("utf-8")
            file_name = "pasted_resume.txt"
        elif pasted_text:
            st.warning("Please paste at least 20 characters of resume content.")

    else:
        if selected_sample_path and os.path.exists(selected_sample_path):
            with open(selected_sample_path, "rb") as f:
                file_bytes = f.read()
            file_name = os.path.basename(selected_sample_path)
            st.info(f"Loaded built-in sample: **{file_name}** from category `{chosen_cat}`")

    # 5. Process and Display Results
    if file_bytes is not None:
        with st.spinner("Extracting text via multi-engine parser, analyzing structure, and computing predictions..."):
            try:
                extraction = PDFExtractor.extract_text_and_meta(file_bytes, file_name=file_name or "resume.pdf")
                raw_text = extraction.get("text", "")
                
                if not raw_text or len(raw_text.strip()) < 20:
                    st.error("⚠️ We could not extract readable text from this document. If it is a flattened scanned image, try pasting the text in '✍️ Paste Resume Text Directly' mode.")
                    return

                analysis = analyzer.analyze(raw_text, top_k=5)
            except Exception as ex:
                st.error(f"Analysis error: {ex}")
                return

        # Top Result Bar
        primary = analysis["primary_category"]
        conf = analysis["confidence_score"]
        icon = CATEGORY_ICONS.get(primary, "🎯")

        # Extraction Engine Banner
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; background:#f8fafc; padding:0.6rem 1.1rem; border-radius:12px; margin-bottom:1.1rem; border:1px solid #e2e8f0; font-size:0.85rem; color:#475569;">
            <span>🔍 Parser Engine: <strong style="color:#1e293b;">{extraction['extraction_method']}</strong></span>
            <span>📄 Document: <strong style="color:#1e293b;">{file_name}</strong></span>
            <span>📝 Parsed: <strong style="color:#059669;">{extraction['word_count']} words</strong></span>
        </div>
        """, unsafe_allow_html=True)

        # Hero Prediction Box
        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-header">Primary Classified Domain</div>
            <div class="prediction-role">{icon} {primary.replace('-', ' ')}</div>
            <div style="font-size: 1rem; color: #ecfdf5; font-weight: 500;">
                Model Prediction Confidence: <strong>{conf:.1f}%</strong> | Status: <strong>Optimal Match</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 4 Key Metrics Grid
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Prediction Confidence</div>
                <div class="metric-value" style="color:#059669;">{conf:.1f}%</div>
                <div class="metric-footer">Calibrated Probability</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Document Volume</div>
                <div class="metric-value">{extraction['word_count']}</div>
                <div class="metric-footer">{extraction['page_count']} Pages • ~{extraction['reading_time_minutes']} min read</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Detected Skills</div>
                <div class="metric-value" style="color:#4338ca;">{analysis['total_skills_count']}</div>
                <div class="metric-footer">Across {len(analysis['detected_skills'])} Domains</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            health_color = "#059669" if extraction['resume_health_score'] >= 80 else "#d97706"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ATS Structure Score</div>
                <div class="metric-value" style="color:{health_color};">{extraction['resume_health_score']}/100</div>
                <div class="metric-footer">Key Sections Completeness</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        # Tabs for Deep Analytics
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Probability & Predictions",
            "🛠️ Detected Skills & Keywords",
            "📋 Resume Structure & Health",
            "🔍 Extracted Text & NLP",
            "📥 Export Assessment Report"
        ])

        # TAB 1: Predictions & Distribution
        with tab1:
            col_chart, col_details = st.columns([3, 2])
            
            with col_chart:
                st.subheader("Top Category Probabilities")
                top_preds = analysis["top_predictions"]
                df_preds = pd.DataFrame(top_preds)
                df_preds['Category_Name'] = df_preds['category'].apply(lambda x: f"{CATEGORY_ICONS.get(x, '📌')} {x}")

                # Modern Altair Bar Chart
                chart = alt.Chart(df_preds).mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8).encode(
                    x=alt.X('confidence_pct:Q', title='Confidence Score (%)', scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y('Category_Name:N', sort='-x', title=None),
                    color=alt.Color(
                        'confidence_pct:Q',
                        scale=alt.Scale(scheme='tealblues'),
                        legend=None
                    ),
                    tooltip=[
                        alt.Tooltip('category:N', title='Category'),
                        alt.Tooltip('confidence_pct:Q', title='Confidence %', format='.2f')
                    ]
                ).properties(height=260)

                # Add text labels on bars
                text = chart.mark_text(
                    align='left',
                    baseline='middle',
                    dx=5,
                    color='#1e293b',
                    fontWeight=600
                ).encode(
                    text=alt.Text('confidence_pct:Q', format='.1f')
                )

                st.altair_chart(chart + text, use_container_width=True)

            with col_details:
                st.subheader("Classification Rationale")
                st.markdown(f"""
                The classifier mapped this profile to **`{primary}`** based on vocabulary distribution, n-gram patterns, and specific domain terms.
                """)
                
                # Matched Trigger Keywords
                kws = analysis.get("matched_category_keywords", [])
                if kws:
                    st.markdown("**Category-Indicative Terms in Resume:**")
                    kw_html = " ".join([f"<span class='keyword-pill'>#{kw}</span>" for kw in kws[:12]])
                    st.markdown(kw_html, unsafe_allow_html=True)
                else:
                    st.info("General domain vocabulary matched model weights.")

                st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
                st.markdown("**Recommended Target Roles:**")
                roles_html = " ".join([f"<span class='role-pill'>{role}</span>" for role in analysis['suggested_roles']])
                st.markdown(roles_html, unsafe_allow_html=True)

        # TAB 2: Skills & Competencies
        with tab2:
            st.subheader("Extracted Skills by Domain")
            skills_dict = analysis.get("detected_skills", {})

            if not skills_dict:
                st.info("No specific technology keywords detected from standard taxonomy. The resume might use non-standard acronyms.")
            else:
                cols = st.columns(2)
                idx = 0
                for domain, skill_list in skills_dict.items():
                    target_col = cols[idx % 2]
                    with target_col:
                        with st.container():
                            st.markdown(f"""
                            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:14px; padding:1.1rem; margin-bottom:1rem;">
                                <div style="font-weight:700; font-size:0.95rem; color:#1e293b; margin-bottom:0.5rem;">
                                    📌 {domain} <span style="font-size:0.8rem; font-weight:600; color:#64748b;">({len(skill_list)})</span>
                                </div>
                                <div>
                                    {' '.join([f"<span class='skill-pill'>{s}</span>" for s in skill_list])}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    idx += 1

            # Skill Recommendations
            st.markdown("---")
            st.subheader("💡 ATS Optimization Suggestions")
            st.markdown(f"""
            - **Target Keyword Density:** Ensure your resume emphasizes high-impact keywords specific to **{primary.replace('-', ' ')}**.
            - **Quantified Metrics:** Use numeric milestones (e.g. *"Increased efficiency by 25%"*, *"Managed budget of $500K"*).
            - **Standard Headings:** Keep standard section names (`Experience`, `Education`, `Skills`) so ATS systems parse content without skipping.
            """)

        # TAB 3: Structure & Health
        with tab3:
            st.subheader("Resume Sections & ATS Compliance")
            
            c_sections, c_contacts = st.columns([3, 2])
            with c_sections:
                st.markdown("##### Detected Core Sections")
                sections = extraction.get("detected_sections", {})
                for section_name, detected in sections.items():
                    if detected:
                        st.markdown(f"✅ **{section_name}** — *Detected*")
                    else:
                        st.markdown(f"⚠️ **{section_name}** — *Not explicitly found (Recommended to add)*")

            with c_contacts:
                st.markdown("##### Extracted Contact Coordinates")
                contacts = extraction.get("contacts", {})
                
                emails = contacts.get("emails", [])
                if emails:
                    st.write(f"📧 **Email:** {', '.join(emails)}")
                else:
                    st.warning("No email address detected.")

                phones = contacts.get("phones", [])
                if phones:
                    st.write(f"📞 **Phone:** {', '.join(phones)}")
                else:
                    st.info("No standardized phone format detected.")

                links = contacts.get("links", [])
                if links:
                    st.write(f"🔗 **Profiles / Links:**")
                    for link in links[:3]:
                        st.caption(f"• `{link}`")

            st.markdown("---")
            st.markdown("##### Document Summary")
            st.write(f"- **Engine Utilized:** `{extraction['extraction_method']}`")
            st.write(f"- **Total Character Count:** {extraction['char_count']:,} characters")
            st.write(f"- **Average Words per Page:** {int(extraction['word_count'] / max(extraction['page_count'], 1))} words")

        # TAB 4: Extracted Text & NLP
        with tab4:
            st.subheader("Extracted & Cleaned Text Inspection")
            view_mode = st.radio("Display Format:", ["Cleaned & Lemmatized Tokens (Model Input)", "Raw Extracted PDF Text"], horizontal=True)

            if view_mode.startswith("Cleaned"):
                st.caption("This normalized text was processed through regex cleaning, stopword removal, and WordNet lemmatization before TF-IDF vectorization:")
                st.text_area("Normalized Text:", analysis.get("cleaned_text_preview", ""), height=280)
            else:
                st.caption("Full textual representation extracted from PDF:")
                st.text_area("Raw Text:", raw_text, height=350)

        # TAB 5: Export Report
        with tab5:
            st.subheader("Export Assessment Summary")
            st.caption("Download the comprehensive classification and resume analysis report for candidate tracking or internal archiving.")

            report_data = {
                "document_name": file_name,
                "primary_category": primary,
                "confidence_score_pct": conf,
                "top_predictions": top_preds,
                "total_words": extraction["word_count"],
                "total_pages": extraction["page_count"],
                "ats_health_score": extraction["resume_health_score"],
                "detected_skills": analysis["detected_skills"],
                "contacts": extraction["contacts"],
                "suggested_roles": analysis["suggested_roles"],
                "matched_indicative_keywords": analysis.get("matched_category_keywords", [])
            }

            json_str = json.dumps(report_data, indent=2)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="📥 Download JSON Report",
                    data=json_str,
                    file_name=f"resume_report_{primary.lower()}.json",
                    mime="application/json",
                    use_container_width=True
                )
            with col_dl2:
                # Markdown report summary
                md_summary = f"""# Resume Analysis Report
- **File:** {file_name}
- **Predicted Category:** {primary} ({conf:.1f}% Confidence)
- **ATS Health Score:** {extraction['resume_health_score']}/100
- **Word Count:** {extraction['word_count']} words across {extraction['page_count']} pages
- **Key Skills Detected:** {analysis['total_skills_count']}
- **Suggested Roles:** {', '.join(analysis['suggested_roles'])}
"""
                st.download_button(
                    label="📄 Download Summary (Markdown)",
                    data=md_summary,
                    file_name=f"resume_summary_{primary.lower()}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

    else:
        # Placeholder screen when no file is uploaded
        st.markdown("""
        <div style="background:#f8fafc; border: 2px dashed #cbd5e1; border-radius: 16px; padding: 3rem 2rem; text-align: center; margin-top: 1rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">📄</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: #1e293b;">No Resume Uploaded Yet</div>
            <div style="font-size: 0.95rem; color: #64748b; max-width: 520px; margin: 0.5rem auto 1.5rem auto;">
                Upload a candidate resume in PDF format using the file uploader above, or switch to 
                <strong>'Built-in Sample Resumes'</strong> in the sidebar to test immediately.
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
