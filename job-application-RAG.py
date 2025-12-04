import streamlit as st
import requests
import json
import pandas as pd
from typing import List
from PyPDF2 import PdfReader


st.set_page_config(page_title="Job Matching RAG")


@st.cache_data
def load_jobs():
    
    return pd.read_csv("jobs.csv")

jobs_df = load_jobs()

jobs_df['description'] = jobs_df['description'].fillna("")


def call_ollama(prompt: str, model: str = "llama3.2") -> str:
    """Call Ollama API to generate response"""
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        return f"Error calling Ollama: {str(e)}"


def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    chunks = []
    current = []
    length = 0
    
    for w in words:
        current.append(w)
        length += len(w) + 1
        if length >= chunk_size:
            chunks.append(" ".join(current))
            current = []
            length = 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def extract_resume_text(uploaded_pdf):
    reader = PdfReader(uploaded_pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def simple_job_match(resume_text: str, jobs_df: pd.DataFrame, top_k: int = 5):
    resume_words = set(resume_text.lower().split())
    scores = []

    for _, row in jobs_df.iterrows():
        job_text = row['description']
        if not isinstance(job_text, str):
            job_text = str(job_text)

        job_words = set(job_text.lower().split())
        overlap = len(resume_words.intersection(job_words))
        scores.append((overlap, row))

    scores.sort(reverse=True, key=lambda x: x[0])
    return [row for _, row in scores[:top_k]]

st.title("Resume Job Application RAG")
st.write("Upload your resume to find relevant job postings")

uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
model = st.selectbox("Choose Ollama Model:", ["llama3.2", "llama3.1", "mistral", "phi3"])

if uploaded_resume:
    with st.spinner("Extracting resume text..."):
        resume_text = extract_resume_text(uploaded_resume)
        chunks = chunk_text(resume_text)

    st.subheader("Extracted Resume Text")
    st.write(resume_text[:1500] + "...")

    if st.button("Find Matching Jobs"):
        with st.spinner("Matching jobs..."):
            top_jobs = simple_job_match(resume_text, jobs_df)

        st.subheader("Top Matching Jobs")

        context = ""
        for job in top_jobs:
            st.markdown(f"""
            ### **{job['title']}**  
            **{job['company']}**  
            {job['description'][:300]}...
            """)
            context += (
                f"Job Title: {job['title']}\n"
                f"Company: {job['company']}\n"
                f"Description: {job['description']}\n\n"
            )

        
        rag_prompt = f"""
        You are a job-matching assistant.

        Given the user's resume:

        {resume_text}

        And the following job postings:

        {context}

        Explain which jobs best fit the candidate and why. Provide a friendly summary.
        """

with st.expander("ℹ️ How it works"):
    st.markdown("""
### Job Matching RAG Architecture
1. Upload a **resume PDF**
2. Extract + chunk the resume text
3. Load job postings from CSV
4. Simple keyword scoring selects top matches  
5. Ollama generates a final recommendation response
""")
