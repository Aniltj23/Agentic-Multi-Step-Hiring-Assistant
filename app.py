import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    raise ValueError("GROQ_API_KEY not found! Make sure .env exists or variable is set.")

import streamlit as st
from screening_engine import run_candidate_screening_from_text, run_candidate_screening_from_pdf
import pdfplumber
import io

st.set_page_config(
    page_title="AI Candidate Screening System",
    layout="wide"
)

st.title("🤖 Agentic AI Hiring Assistant")
st.markdown("Autonomous candidate evaluation using multi-step reasoning")

# -----------------------------
# INPUT SECTION
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Candidate Resume")

    uploaded_file = st.file_uploader(
        "Upload Resume (PDF)",
        type="pdf"
    )

    resume_text = ""

    if uploaded_file is not None:
        try:
            # Use pdfplumber for cleaner extraction
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        resume_text += text + "\n"

            st.success("✅ PDF Resume Uploaded Successfully")

        except Exception as e:
            st.error(f"Error reading PDF file: {e}")

    # Manual fallback option
    manual_resume = st.text_area(
        "Or paste resume text manually",
        height=150
    )

    # If manual text exists, override PDF text
    if manual_resume.strip():
        resume_text = manual_resume.strip()

    # Optional: Summarize resume text (simple example)
    if resume_text:
        summarize_resume = st.checkbox("🔹 Summarize Resume Text", value=False)
        if summarize_resume:
            # Simple summarization by taking first 10 lines (can replace with AI summarization later)
            resume_lines = resume_text.splitlines()
            resume_text = "\n".join(resume_lines[:10])
            st.info("Resume text summarized for faster processing.")

with col2:
    st.subheader("📝 Job Description")
    job_desc = st.text_area(
        "Paste job description here",
        height=280
    )

# -----------------------------
# RUN BUTTON
# -----------------------------
if st.button("🚀 Run AI Screening", use_container_width=True):

    if not resume_text or not job_desc:
        st.warning("Please provide Resume and Job Description.")
    else:
        with st.spinner("Running autonomous hiring agents..."):
            if uploaded_file is not None:
                results = run_candidate_screening_from_pdf(uploaded_file, job_desc)
            else:
                results = run_candidate_screening_from_text(resume_text, job_desc)

        st.success("Screening Complete!")

        colA, colB = st.columns(2)

        with colA:
            st.subheader("📊 Evaluation Summary")
            st.write("**Experience Level:**", results.get("experience_level"))
            st.write("**Skill Match:**", results.get("skill_match"))

            relevance = results.get("relevance_score", 0)
            st.write("**Relevance Score:**")
            st.progress(int(relevance))

        with colB:
            st.subheader("🎯 Final Decision")

            decision = results.get("final_decision")

            if decision == "schedule_interview":
                st.success("✅ Schedule Interview")
            elif decision == "notify_recruiter":
                st.info("📩 Notify Recruiter")
            else:
                st.error("❌ Reject Application")

            confidence = results.get("confidence_score", 0)
            st.metric("Confidence Score", f"{confidence}%")

        st.divider()

        st.subheader("🧠 AI Analysis Summary")
        st.write(results.get("analysis_summary"))