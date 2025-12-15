import streamlit as st
import re
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# ---------- Helper functions ----------
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    return docx2txt.process(file)

def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    words = [word for word in text.split() if word not in stopwords.words('english')]
    return ' '.join(words)

def calculate_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(similarity * 100, 2)

def suggest_improvements(resume_text, jd_text):
    resume_words = set(resume_text.split())
    jd_words = set(jd_text.split())
    missing_keywords = jd_words - resume_words
    suggestions = [word for word in missing_keywords if len(word) > 3]
    return list(suggestions)[:15]

# ---------- Streamlit UI ----------
st.title("🧠 Resume–JD Match Analyzer")
st.write("Upload your **Resume** and provide the **Job Description** (upload or paste), to get your match percentage!")

# Resume upload
resume_file = st.file_uploader("📄 Upload your Resume (PDF or DOCX)", type=["pdf", "docx"])

# JD input options
st.subheader("💼 Job Description Input")
jd_input_option = st.radio("Choose how you'd like to provide the Job Description:",
                           ["Paste text manually", "Upload file"])

jd_text = ""

if jd_input_option == "Paste text manually":
    jd_text = st.text_area("Paste the Job Description here:", height=200)
else:
    jd_file = st.file_uploader("Upload Job Description (PDF or DOCX)", type=["pdf", "docx"])
    if jd_file:
        if jd_file.name.endswith(".pdf"):
            jd_text = extract_text_from_pdf(jd_file)
        else:
            jd_text = extract_text_from_docx(jd_file)

# Process button
if resume_file and jd_text.strip():
    if resume_file.name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(resume_file)
    else:
        resume_text = extract_text_from_docx(resume_file)

    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    match_score = calculate_similarity(resume_clean, jd_clean)

    st.subheader("✅ Match Score:")
    st.metric(label="Resume–JD Match", value=f"{match_score}%")

    suggestions = suggest_improvements(resume_clean, jd_clean)
    st.subheader("🛠️ Suggested Keywords to Add:")
    if suggestions:
        st.write(", ".join(suggestions))
    else:
        st.write("Your resume already matches well with the job description! 🎉")

    st.download_button(
        label="⬇️ Download Suggestions",
        data="\n".join(suggestions),
        file_name="resume_improvement_suggestions.txt",
        mime="text/plain"
    )
