# screening_engine.py
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    raise ValueError("GROQ_API_KEY not found! Make sure .env exists or variable is set.")

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq.chat_models import ChatGroq
import json
import re
import pdfplumber

# ===============================
# LLM INITIALIZATION
# ===============================
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# ===============================
# STATE DEFINITION
# ===============================
class State(TypedDict, total=False):
    application: str
    job_description: str
    experience_level: str
    skill_match: str
    relevance_score: float
    analysis_summary: str
    agent_decision: str
    final_decision: str
    confidence_score: float
    reflection_attempts: int

# ===============================
# TOOL 1 — EXPERIENCE CATEGORY
# ===============================
def categorize_experience(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Categorize candidate as 'Entry-level', 'Mid-level', or 'Senior-level'. "
        "Return ONLY one.\n\nResume:\n{application}"
    )
    result = (prompt | llm).invoke(state).content.strip()
    return {"experience_level": result}

# ===============================
# TOOL 2 — SKILL MATCH
# ===============================
def assess_skillset(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Compare resume with job description. "
        "Return ONLY: 'Match' or 'No Match'.\n\n"
        "Resume:\n{application}\n\n"
        "Job Description:\n{job_description}"
    )
    result = (prompt | llm).invoke(state).content.strip()
    return {"skill_match": result}

# ===============================
# TOOL 3 — DEEP ANALYSIS
# ===============================
def safe_json_extract(text: str):
    try:
        text = re.sub(r"```json|```", "", text).strip()
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return None
    except:
        return None

def deep_profile_analysis(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Analyze candidate deeply.\n"
        "Return JSON:\n"
        "{{\n"
        '"relevance_score": number_0_to_100,\n'
        '"analysis_summary": "short paragraph"\n'
        "}}\n\n"
        "Resume:\n{application}\n\n"
        "Job Description:\n{job_description}"
    )
    result = (prompt | llm).invoke(state).content
    parsed = safe_json_extract(result)
    if not parsed:
        return {"relevance_score": 50.0, "analysis_summary": "Parsing failed."}
    return {
        "relevance_score": float(parsed.get("relevance_score", 50)),
        "analysis_summary": parsed.get("analysis_summary", "")
    }

# ===============================
# AGENT DECISION
# ===============================
def hiring_decision_agent(state: State) -> State:
    experience_level = state.get("experience_level", "Unknown")
    skill_match = state.get("skill_match", "Unknown")
    relevance_score = state.get("relevance_score", 50)
    analysis_summary = state.get("analysis_summary", "Not available.")

    prompt = ChatPromptTemplate.from_template(
        "You are an autonomous hiring agent.\n\n"
        "Candidate Data:\n"
        "- Experience Level: {experience_level}\n"
        "- Skill Match: {skill_match}\n"
        "- Relevance Score: {relevance_score}\n"
        "- Analysis: {analysis_summary}\n\n"

        "Decision Guidelines:\n"
        "1. Consider ALL factors together — do NOT base decision on a single factor.\n"
        "2. If Skill Match is 'No Match', you MUST reject the candidate regardless of any other factor.\n"
        "3. High relevance score (>75) with good experience → schedule interview.\n"
        "4. Moderate relevance (50–75) → notify recruiter for manual review.\n"
        "5. Low relevance (<50) → reject.\n\n"

        "Decide ONE action:\n"
        "- schedule_interview\n"
        "- reject_application\n"
        "- notify_recruiter\n\n"
        "Return ONLY the action name."
    )

    decision = (prompt | llm).invoke({
        "experience_level": experience_level,
        "skill_match": skill_match,
        "relevance_score": relevance_score,
        "analysis_summary": analysis_summary,
    }).content.strip()

    return {"agent_decision": decision}

# ===============================
# REFLECTION NODE
# ===============================
def reflection_agent(state: State) -> State:
    attempts = state.get("reflection_attempts", 0) + 1
    agent_decision = state.get("agent_decision", "reject_application")
    experience_level = state.get("experience_level", "Unknown")
    skill_match = state.get("skill_match", "Unknown")
    relevance_score = state.get("relevance_score", 50)
    analysis_summary = state.get("analysis_summary", "Not available.")

    prompt = ChatPromptTemplate.from_template(
        "You are reviewing your previous hiring decision.\n\n"
        "Candidate Data:\n"
        "- Experience Level: {experience_level}\n"
        "- Skill Match: {skill_match}\n"
        "- Relevance Score: {relevance_score}\n"
        "- Analysis Summary: {analysis_summary}\n\n"
        "Your Previous Decision: {agent_decision}\n\n"
        "Carefully evaluate whether the previous decision logically aligns "
        "with the candidate data.\n\n"
        "If the previous decision is correct, keep it.\n"
        "If inconsistent, change it.\n\n"
        "Return ONLY one:\n"
        "- schedule_interview\n"
        "- reject_application\n"
        "- notify_recruiter"
    )

    final = (prompt | llm).invoke({
        "agent_decision": agent_decision,
        "experience_level": experience_level,
        "skill_match": skill_match,
        "relevance_score": relevance_score,
        "analysis_summary": analysis_summary
    }).content.strip()

    return {"final_decision": final, "reflection_attempts": attempts}

# ===============================
# CONFIDENCE NODE
# ===============================
def confidence_agent(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Provide confidence score (0-100) for this final decision: {final_decision}.\n"
        "Return ONLY number."
    )

    score = (prompt | llm).invoke(state).content.strip()
    try:
        score = float(score)
    except:
        score = 75.0
    return {"confidence_score": score}

# ===============================
# ROUTING FUNCTIONS
# ===============================
def route_after_skill_check(state: State):
    if state["skill_match"] == "No Match":
        return "hiring_decision_agent"
    return "deep_profile_analysis"

def route_after_company_check(state: State):
    return "hiring_decision_agent"

def route_after_agent_decision(state: State):
    return "reflection_agent"

def route_after_confidence(state: State):
    if state.get("reflection_attempts", 0) >= 3:
        return END
    if state.get("confidence_score", 0) < 60:
        return "reflection_agent"
    return END

# ===============================
# GRAPH BUILD
# ===============================
workflow = StateGraph(State)

workflow.add_node("categorize_experience", categorize_experience)
workflow.add_node("assess_skillset", assess_skillset)
workflow.add_node("deep_profile_analysis", deep_profile_analysis)
workflow.add_node("hiring_decision_agent", hiring_decision_agent)
workflow.add_node("reflection_agent", reflection_agent)
workflow.add_node("confidence_agent", confidence_agent)

workflow.add_edge(START, "categorize_experience")
workflow.add_edge("categorize_experience", "assess_skillset")

workflow.add_conditional_edges(
    "assess_skillset",
    route_after_skill_check,
    {"deep_profile_analysis": "deep_profile_analysis",
     "hiring_decision_agent": "hiring_decision_agent"}
)

workflow.add_edge("deep_profile_analysis", "hiring_decision_agent")

workflow.add_conditional_edges(
    "hiring_decision_agent",
    route_after_agent_decision,
    {"reflection_agent": "reflection_agent"}
)

workflow.add_edge("reflection_agent", "confidence_agent")

workflow.add_conditional_edges(
    "confidence_agent",
    route_after_confidence,
    {"reflection_agent": "reflection_agent", END: END}
)

app = workflow.compile()

# ===============================
# RUN FUNCTIONS FOR TEXT AND PDF
# ===============================

# Function for plain text resumes
def run_candidate_screening_from_text(text: str, job_description: str):
    # Summarize resume text
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize this resume in a concise way, keeping key skills, experience, and companies:\n\n{text}"
    )
    summary = (summary_prompt | llm).invoke({"text": text}).content.strip()

    # Run the main screening pipeline
    return app.invoke({
        "application": summary,
        "full_application": text,
        "job_description": job_description
    })


# Function for uploaded PDF resumes
def run_candidate_screening_from_pdf(pdf_file, job_description: str):
    # pdf_file can be a path or Streamlit UploadedFile
    with pdfplumber.open(pdf_file) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    return run_candidate_screening_from_text(text, job_description)

# ===============================
# EXAMPLE USAGE
# ===============================
if __name__ == "__main__":
    pdf_resume = "resume.pdf"
    job_desc = "Looking for senior Python ML engineer with production experience."

    results = run_candidate_screening_from_pdf(pdf_resume, job_desc)

    print("\n===== AGENTIC AI OUTPUT =====")
    for k, v in results.items():
        print(f"{k}: {v}")