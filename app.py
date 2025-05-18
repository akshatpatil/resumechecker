import streamlit as st
import PyPDF2
import pytesseract
from PIL import Image
import io
import spacy
import plotly.graph_objects as go
import plotly.express as px
import re
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import random

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Updated color scheme for darker, aesthetic muted lilac and charcoal black
COLORS = {
    "primary": "#7C6A8A",   # Dark Muted Lilac
    "secondary": "#232129", # Charcoal Black
    "accent": "#A18FC6",    # Muted Lilac Accent
    "background": "#232129",# Charcoal Black (changed from dark lilac)
    "text": "#F3F0F7",      # Light text for dark background
    "success": "#4B5E4D",   # Dark Muted Green
    "warning": "#B08B4F",   # Muted Gold
    "danger": "#7B3A3A",    # Dark Muted Red
}

# Set page config
st.set_page_config(
    page_title="Resume Checker",
    page_icon="📝",
    layout="wide"
)

# Custom CSS for styling
st.markdown(f"""
<style>
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: {COLORS["background"]};
    }}
    h1, h2, h3 {{
        color: {COLORS["text"]};
    }}
    body, .stApp {{
        background-color: {COLORS["background"]} !important;
        color: {COLORS["text"]} !important;
    }}
    .stButton>button {{
        background-color: {COLORS["accent"]};
        color: white;
    }}
    .stButton>button:hover {{
        background-color: {COLORS["primary"]};
        color: {COLORS["secondary"]};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: {COLORS["primary"]};
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: {COLORS["text"]};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS["accent"]};
        color: white;
    }}
    div[data-testid="stSidebarNav"] {{
        background-color: {COLORS["primary"]};
        padding-top: 2rem;
    }}
    div[data-testid="stSidebar"] {{
        background-color: {COLORS["primary"]};
    }}
    .bias-high {{
        color: {COLORS["danger"]};
        font-weight: bold;
    }}
    .bias-medium {{
        color: {COLORS["warning"]};
        font-weight: bold;
    }}
    .bias-low {{
        color: {COLORS["success"]};
        font-weight: bold;
    }}
    .suggestion-card {{
        background-color: #2E2935; /* Slightly lighter charcoal for cards */
        color: {COLORS["text"]};
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 10px;
    }}
    .section-card {{
        background-color: #2E2935; /* Slightly lighter charcoal for cards */
        color: {COLORS["text"]};
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 20px;
    }}
</style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_image(image_file):
    """Extract text from uploaded image file"""
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

def extract_resume_text(uploaded_file):
    """Extract text from resume file based on file type"""
    if uploaded_file is None:
        return None
    
    file_type = uploaded_file.type
    if 'pdf' in file_type:
        return extract_text_from_pdf(uploaded_file)
    elif 'image' in file_type:
        return extract_text_from_image(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload PDF, JPG, or PNG.")
        return None

def analyze_resume(resume_text, job_description):
    """Analyze resume against job description"""
    if not resume_text or not job_description:
        return None

    # Process text with spaCy
    resume_doc = nlp(resume_text)
    job_doc = nlp(job_description)

    # Extract key skills from job description
    job_skills = extract_skills(job_doc)
    resume_skills = extract_skills(resume_doc)

    # Calculate metrics
    metrics = calculate_metrics(resume_doc, job_doc, resume_skills, job_skills)

    # Generate improvement suggestions
    suggestions = generate_suggestions(metrics, resume_doc, job_doc)

    # Resume structure
    resume_structure = analyze_resume_structure(resume_text)

    return {
        "metrics": metrics,
        "suggestions": suggestions,
        "resume_structure": resume_structure,
        "overall_rating": calculate_overall_rating(metrics)
    }

def extract_skills(doc):
    """Extract potential skills from text"""
    # This is a simple implementation - could be enhanced with a skills database
    skills = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2:
            skills.append(token.text.lower())
    return list(set(skills))

def calculate_metrics(resume_doc, job_doc, resume_skills, job_skills):
    """Calculate various metrics for resume analysis"""
    # Keyword matching
    keyword_match = calculate_keyword_match(resume_skills, job_skills)
    
    # ATS compatibility metrics
    ats_metrics = analyze_ats_compatibility(resume_doc.text)
    
    # Content metrics
    content_metrics = analyze_content(resume_doc)
    
    # Combine all metrics
    metrics = {
        "keyword_match": keyword_match,
        "ats_compatibility": ats_metrics,
        "content_metrics": content_metrics
    }
    
    return metrics

def calculate_keyword_match(resume_skills, job_skills):
    """Calculate keyword match percentage between resume and job description"""
    if not job_skills:
        return 0
    
    matched_skills = [skill for skill in resume_skills if skill in job_skills]
    match_percentage = (len(matched_skills) / len(job_skills)) * 100
    
    return {
        "percentage": min(match_percentage, 100),
        "matched_skills": matched_skills,
        "missing_skills": [skill for skill in job_skills if skill not in resume_skills]
    }

def analyze_ats_compatibility(resume_text):
    """Analyze ATS compatibility of the resume"""
    # Check for common ATS issues
    has_tables = "table" in resume_text.lower()
    has_images = len(re.findall(r'image|img|figure|pic', resume_text.lower())) > 0
    has_headers_footers = len(re.findall(r'header|footer', resume_text.lower())) > 0
    
    # Check formatting
    formatting_score = 85  # Base score, would be more sophisticated in a full implementation
    if has_tables:
        formatting_score -= 15
    if has_images:
        formatting_score -= 10
    if has_headers_footers:
        formatting_score -= 10
    
    return {
        "formatting_score": max(formatting_score, 0),
        "issues": {
            "has_tables": has_tables,
            "has_images": has_images,
            "has_headers_footers": has_headers_footers
        }
    }

def analyze_content(doc):
    """Analyze content quality of the resume"""
    # Count action verbs (simplified implementation)
    action_verbs = ["achieved", "improved", "led", "managed", "developed", 
                    "created", "implemented", "increased", "reduced", "resolved"]
    action_verb_count = sum(1 for token in doc if token.text.lower() in action_verbs)
    
    # Calculate sentence length and complexity
    sentences = list(doc.sents)
    avg_sentence_length = sum(len(sent) for sent in sentences) / max(len(sentences), 1)
    
    # Section completeness (simplified)
    key_sections = ["experience", "education", "skills"]
    sections_present = sum(1 for section in key_sections if section in doc.text.lower())
    section_completeness = (sections_present / len(key_sections)) * 100
    
    return {
        "action_verb_count": action_verb_count,
        "avg_sentence_length": avg_sentence_length,
        "section_completeness": section_completeness
    }

def analyze_resume_structure(resume_text):
    """Analyze resume structure for hierarchy chart"""
    # Simplified implementation - in a real app, you'd use more sophisticated parsing
    structure = {
        "name": "Resume",
        "children": []
    }
    
    # Look for common resume sections
    sections = {
        "Contact Information": len(re.findall(r'email|phone|address', resume_text.lower())),
        "Summary": len(re.findall(r'summary|objective|profile', resume_text.lower())),
        "Experience": len(re.findall(r'experience|work|employment', resume_text.lower())),
        "Education": len(re.findall(r'education|degree|university|college', resume_text.lower())),
        "Skills": len(re.findall(r'skills|abilities|competencies', resume_text.lower())),
        "Projects": len(re.findall(r'project', resume_text.lower())),
        "Certifications": len(re.findall(r'certification|certificate', resume_text.lower())),
    }
    
    # Add sections with mentions to the structure
    for section, count in sections.items():
        if count > 0:
            structure["children"].append({
                "name": section,
                "value": count * 10  # Scale for better visualization
            })
    
    return structure

def calculate_overall_rating(metrics):
    """Calculate overall ATS compatibility rating"""
    # Weight different factors
    keyword_weight = 0.4
    ats_weight = 0.4
    content_weight = 0.2
    
    # Calculate weighted scores
    keyword_score = metrics["keyword_match"]["percentage"] * keyword_weight
    ats_score = metrics["ats_compatibility"]["formatting_score"] * ats_weight
    content_score = metrics["content_metrics"]["section_completeness"] * content_weight
    
    # Overall rating
    overall = keyword_score + ats_score + content_score
    
    # Rating category
    if overall >= 85:
        category = "Excellent"
    elif overall >= 70:
        category = "Good"
    elif overall >= 50:
        category = "Needs Improvement"
    else:
        category = "Poor"
    
    return {
        "score": overall,
        "category": category
    }

def generate_suggestions(metrics, resume_doc, job_doc):
    """Generate improvement suggestions based on analysis"""
    suggestions = []
    # Keyword suggestions
    if metrics["keyword_match"]["percentage"] < 70:
        missing = metrics["keyword_match"]["missing_skills"]
        if missing:
            suggestions.append({
                "type": "keywords",
                "title": "Add Missing Keywords",
                "description": f"Include these keywords from the job description: {', '.join(missing[:5])}",
                "impact": "high"
            })
    # ATS suggestions
    ats_issues = metrics["ats_compatibility"]["issues"]
    if ats_issues["has_tables"]:
        suggestions.append({
            "type": "ats",
            "title": "Replace Tables with Bullet Points",
            "description": "ATS systems often struggle to parse tables. Convert tabular data to bullet points for better compatibility.",
            "impact": "high"
        })
    if ats_issues["has_images"]:
        suggestions.append({
            "type": "ats",
            "title": "Minimize Use of Images",
            "description": "ATS systems cannot read images. Ensure all important information is in text format.",
            "impact": "medium"
        })
    # Content suggestions
    if metrics["content_metrics"]["action_verb_count"] < 5:
        suggestions.append({
            "type": "content",
            "title": "Use More Action Verbs",
            "description": "Start bullet points with impactful action verbs like 'achieved', 'implemented', 'developed', 'increased', 'reduced'.",
            "impact": "medium"
        })
    if metrics["content_metrics"]["section_completeness"] < 100:
        suggestions.append({
            "type": "structure",
            "title": "Include All Key Sections",
            "description": "Ensure your resume has all essential sections: Experience, Education, and Skills.",
            "impact": "high"
        })
    return suggestions

def create_radar_chart(metrics, overall_rating):
    """Create radar chart for resume metrics with dark background and neat style"""
    categories = [
        'Keyword Match', 'ATS Compatibility', 
        'Section Completeness', 'Content Quality', 'Overall'
    ]
    values = [
        metrics["keyword_match"]["percentage"],
        metrics["ats_compatibility"]["formatting_score"],
        metrics["content_metrics"]["section_completeness"],
        min(metrics["content_metrics"]["action_verb_count"] * 10, 100),  # Scale action verbs
        overall_rating["score"]
    ]
    dark_bg = "#232129"
    grid_color = "#444054"
    accent_rgba = "rgba(161,143,198,0.5)"
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Resume Metrics',
        line_color=COLORS["accent"],
        fillcolor=accent_rgba,
        marker=dict(size=8, color=COLORS["accent"])
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=dark_bg,
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor=grid_color,
                linecolor=COLORS["accent"],
                tickfont=dict(color=COLORS["accent"])
            ),
            angularaxis=dict(
                gridcolor=grid_color,
                linecolor=COLORS["accent"],
                tickfont=dict(color=COLORS["accent"])
            ),
        ),
        paper_bgcolor=dark_bg,
        plot_bgcolor=dark_bg,
        font=dict(color=COLORS["accent"]),
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
        title={
            "text": "Resume Strengths & Weaknesses",
            "font": {"color": COLORS["accent"], "size": 22}
        }
    )
    return fig

def create_hierarchy_chart(structure):
    """Create hierarchy chart for resume structure with charcoal black background and visible title"""
    # Sort children by value descending for neatness
    children_sorted = sorted(structure["children"], key=lambda x: x["value"], reverse=True)
    labels = [structure["name"]]
    parents = [""]
    values = [100]
    section_palette = [
        "#A18FC6", "#7C6A8A", "#B08B4F", "#4B5E4D", "#7B3A3A", "#F3F0F7", "#444054"
    ]
    node_colors = [COLORS["accent"]]
    for idx, child in enumerate(children_sorted):
        labels.append(child["name"])
        parents.append(structure["name"])
        values.append(child["value"])
        node_colors.append(section_palette[idx % len(section_palette)])
    dark_bg = COLORS["background"]  # "#232129"
    node_line_color = "#3A2E4D"
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(
            colors=node_colors,
            line=dict(color=node_line_color, width=4)
        ),
        insidetextfont=dict(color=dark_bg, size=18, family="Arial"),
        outsidetextfont=dict(color=COLORS["accent"], size=15, family="Arial"),
        maxdepth=2
    ))
    fig.update_layout(
        margin=dict(t=10, l=10, r=10, b=10),
        paper_bgcolor=dark_bg,
        plot_bgcolor=dark_bg,
        font=dict(color=COLORS["text"], family="Arial"),
        # Remove the title from the chart
    )
    return fig

# --- Main app interface (single page, all metrics and bias) ---

def main():
    st.title("📝 Resume ATS Compatibility Checker")
    st.markdown('<p style="font-size: 20px;">Optimize your resume for ATS systems</p>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📄 Upload Files")
    resume_file = st.file_uploader("Upload your resume", type=["pdf", "png", "jpg", "jpeg"])
    job_description = st.text_area("Paste the job description", height=200)
    analyze_button = st.button("Analyze Resume & Job Description", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if analyze_button and (resume_file or job_description):
        with st.spinner("Analyzing your resume and job description..."):
            resume_text = extract_resume_text(resume_file) if resume_file else ""
            if resume_text or job_description:
                analysis_results = analyze_resume(resume_text, job_description)
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.header("🎯 Resume & Job Description Analysis")

                # Metrics Section (all on one page)
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Keyword Match", f"{analysis_results['metrics']['keyword_match']['percentage']:.1f}%", delta=None)
                col2.metric("ATS Formatting", f"{analysis_results['metrics']['ats_compatibility']['formatting_score']:.1f}%", delta=None)
                col3.metric("Section Completeness", f"{analysis_results['metrics']['content_metrics']['section_completeness']:.1f}%", delta=None)
                col4.metric("Action Verbs", f"{analysis_results['metrics']['content_metrics']['action_verb_count']} verbs", delta=None)

                # Radar chart (darker, aesthetic)
                radar_chart = create_radar_chart(analysis_results["metrics"], analysis_results["overall_rating"])
                st.plotly_chart(radar_chart, use_container_width=True)

                # --- Removed Resume Structure Analysis Section ---

                # Keyword matching details
                st.subheader("Keyword Matching")
                st.write("**Matched:**", ", ".join(analysis_results['metrics']['keyword_match']['matched_skills']))
                st.write("**Missing:**", ", ".join(analysis_results['metrics']['keyword_match']['missing_skills']))

                # Suggestions
                st.header("💡 Personalized Suggestions")
                suggestions = analysis_results["suggestions"]
                if suggestions:
                    for suggestion in suggestions:
                        impact_color = COLORS["primary"]
                        if suggestion["impact"] == "high":
                            impact_color = COLORS["danger"]
                        elif suggestion["impact"] == "medium":
                            impact_color = COLORS["warning"]
                        st.markdown(f"""
                        <div class="suggestion-card">
                            <h4>{suggestion["title"]}</h4>
                            <p>{suggestion["description"]}</p>
                            <p style="color: {impact_color}; font-weight: bold;">Impact: {suggestion["impact"].title()}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("Your resume looks great! No specific suggestions at this time.")

                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Could not extract text from the resume. Please check the file format or paste a job description.")

    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 20px; color: gray; font-size: 0.8em;">
        Resume ATS Compatibility Checker
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()