# Resume Screening System using NLP (TF-IDF + Cosine Similarity)
# Professional University Project Submission
# Author: Anto Tom | Matriculation No: 111122757
# Instructor: Nadra Tabassum

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class ResumeScreeningSystem:
    def __init__(self, threshold=0.2):
        """
        Initialize the Resume Screening System.
        threshold: Similarity score above which resume is considered a 'Good Match'
        """
        self.threshold = threshold
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()

    def preprocess_text(self, text):
        """
        Text Preprocessing:
        - Convert to lowercase
        - Remove punctuation and special characters
        - Remove extra spaces
        - Remove English stopwords
        - Tokenize into words
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)

    def screen_resume(self, resume_text, job_description):
        """
        Complete screening pipeline:
        1. Preprocess both texts
        2. TF-IDF vectorization
        3. Cosine similarity computation
        4. Threshold-based decision
        """
        # Preprocess texts
        processed_resume = self.preprocess_text(resume_text)
        processed_job = self.preprocess_text(job_description)
        
        # TF-IDF Vectorization (fit on both documents)
        documents = [processed_resume, processed_job]
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        # Extract vectors
        resume_vector = tfidf_matrix[0:1]  # Shape (1, n_features)
        job_vector = tfidf_matrix[1:2]     # Shape (1, n_features)
        
        # Compute cosine similarity
        similarity_score = cosine_similarity(resume_vector, job_vector)[0][0]
        
        # Make decision
        if similarity_score >= self.threshold:
            decision = "Good Match"
        else:
            decision = "Not a Match"
        
        return {
            'similarity_score': round(similarity_score, 4),
            'decision': decision,
            'processed_resume': processed_resume,
            'processed_job': processed_job
        }

# Sample data for demonstration (Real-world ready)
SAMPLE_RESUMES = {
    "Data Scientist": """
    Experienced Data Scientist with 5+ years in machine learning, Python, SQL, and data analysis. 
    Proficient in TensorFlow, PyTorch, NLP, and statistical modeling. 
    Delivered production ML models for customer segmentation and churn prediction.
    """,
    "Software Engineer": """
    Full-stack Software Engineer skilled in Java, Spring Boot, React, AWS cloud services. 
    Built scalable microservices and RESTful APIs. 
    Experience in Agile development and CI/CD pipelines.
    """,
    "HR Manager": """
    HR professional with expertise in recruitment, employee relations, and talent acquisition. 
    Implemented performance management systems and conducted training programs.
    """
}

SAMPLE_JOBS = {
    "Data Scientist JD": """
    Looking for Data Scientist with expertise in machine learning, Python programming, 
    NLP, deep learning frameworks like TensorFlow, data preprocessing, model deployment.
    Experience with large datasets and statistical analysis required.
    """,
    "Software Engineer JD": """
    Software Engineer position requiring Java, Spring framework, REST APIs, 
    cloud computing (AWS), database design, and agile methodologies.
    """,
    "HR Manager JD": """
    HR Manager role involving talent acquisition, employee engagement, 
    performance reviews, compliance, and organizational development.
    """
}

# Streamlit Web Interface for Professional Demo
def run_streamlit_app():
    st.set_page_config(page_title="Resume Screening System", layout="wide")
    st.title("🧠 Resume Screening System using NLP (TF-IDF + Cosine Similarity)")
    st.markdown("---")
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Configuration")
    threshold = st.sidebar.slider("Match Threshold", 0.0, 1.0, 0.2, 0.01)
    
    system = ResumeScreeningSystem(threshold)
    
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Resume Input")
        resume_input = st.text_area("Paste candidate resume here:", 
                                  value=SAMPLE_RESUMES["Data Scientist"], 
                                  height=200)
    
    with col2:
        st.subheader("💼 Job Description")
        job_input = st.text_area("Paste job description here:", 
                               value=SAMPLE_JOBS["Data Scientist JD"], 
                               height=200)
    
    if st.button("🚀 Screen Resume", type="primary"):
        if resume_input.strip() and job_input.strip():
            result = system.screen_resume(resume_input, job_input)
            
            st.markdown("---")
            st.subheader("✅ Screening Results")
            
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.metric("Similarity Score", f"{result['similarity_score']}", delta=None)
                st.success(f"**Decision**: {result['decision']}")
            
            with col_b:
                st.subheader("🔍 Processed Texts")
                st.info(f"**Resume**: {result['processed_resume'][:300]}...")
                st.info(f"**Job**: {result['processed_job'][:300]}...")
            
            # Demo with samples
            st.markdown("---")
            st.subheader("🎯 Sample Tests")
            for role in ["Data Scientist", "Software Engineer"]:
                resume_sample = SAMPLE_RESUMES[role]
                job_sample = SAMPLE_JOBS[f"{role} JD"]
                sample_result = system.screen_resume(resume_sample, job_sample)
                st.caption(f"{role}: Score={sample_result['similarity_score']:.3f} → {sample_result['decision']}")
        else:
            st.warning("Please provide both resume and job description.")

if __name__ == "__main__":
    # For Jupyter/Colab: Run individual screening
    print("=== Resume Screening System Demo ===")
    system = ResumeScreeningSystem(threshold=0.2)
    
    # Test cases
    test_cases = [
        (SAMPLE_RESUMES["Data Scientist"], SAMPLE_JOBS["Data Scientist JD"]),
        (SAMPLE_RESUMES["Software Engineer"], SAMPLE_JOBS["Data Scientist JD"]),
        (SAMPLE_RESUMES["HR Manager"], SAMPLE_JOBS["Data Scientist JD"])
    ]
    
    for i, (resume, job) in enumerate(test_cases, 1):
        result = system.screen_resume(resume, job)
        print(f"\nTest {i}:")
        print(f"Similarity: {result['similarity_score']:.4f}")
        print(f"Decision: {result['decision']}")
    
    print("\n🎉 System ready! Run 'streamlit run this_file.py' for web demo.")
    print("📦 Requirements: pip install streamlit scikit-learn nltk numpy")
