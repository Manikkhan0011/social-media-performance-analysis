import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# NLTK Downloads
nltk.download('stopwords')
from nltk.corpus import stopwords

# 1. Sample Resume Dataset
data = {
    'Candidate_ID': ['INT001', 'INT002', 'INT003', 'INT004', 'INT005'],
    'Candidate_Name': ['Ali Khan', 'Sara Ahmed', 'Hamza Sheikh', 'Fatima Zain', 'Usman Raza'],
    'Resume_Text': [
        "Data Analyst intern experienced in Python, SQL, Pandas, Matplotlib, and Seaborn for EDA. Strong data cleaning skills.",
        "Web Developer skilled in HTML, CSS, JavaScript, React, Shopify, and Node.js with frontend UI experience.",
        "Data Science enthusiast proficient in Python, Machine Learning, Scikit-Learn, NLTK, SQL, and Deep Learning.",
        "Digital Marketer skilled in SEO, Content Writing, Social Media Marketing, Canva, and Google Ads.",
        "Python Developer with expertise in Flask, Django, SQL, REST APIs, Git, GitHub, and Automation."
    ]
}

df = pd.DataFrame(data)

# Target Job Description (Data Analyst / Data Scientist Role)
job_description = """
Looking for a Data Science / Data Analyst Intern with strong skills in Python, SQL, 
Pandas, Machine Learning, Data Visualization, NLTK, and EDA.
"""

# 2. Text Cleaning Function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Special characters remove karna
    text = text.lower()  # Lowercase conversion
    words = text.split()
    words = [w for w in words if w not in stop_words]  # Stopwords removal
    return ' '.join(words)

df['Cleaned_Resume'] = df['Resume_Text'].apply(clean_text)
cleaned_jd = clean_text(job_description)

# 3. TF-IDF & Cosine Similarity (Matching Logic)
corpus = [cleaned_jd] + df['Cleaned_Resume'].tolist()

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# Calculate similarity between Job Description (index 0) and all resumes
similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# Assign match percentages
df['Match_Score_%'] = np.round(similarity_scores * 100, 2)
df['Status'] = df['Match_Score_%'].apply(lambda x: 'Shortlisted' if x >= 40 else 'Rejected')

# Sort by Best Match
df = df.sort_values(by='Match_Score_%', ascending=False).reset_index(drop=True)

print("--- SCREENING RESULTS ---")
print(df[['Candidate_ID', 'Candidate_Name', 'Match_Score_%', 'Status']])

# 4. Generate Visualization Chart (Seaborn Warning Fixed)
plt.figure(figsize=(9, 5))
palette = ['#2ecc71' if s == 'Shortlisted' else '#e74c3c' for s in df['Status']]

ax = sns.barplot(
    x='Match_Score_%', 
    y='Candidate_Name', 
    data=df, 
    hue='Candidate_Name', 
    palette=palette, 
    legend=False
)

plt.title('Resume Matching Score against Job Description', fontsize=14, fontweight='bold')
plt.xlabel('Match Score (%)', fontsize=12)
plt.ylabel('Candidate Name', fontsize=12)
plt.xlim(0, 100)

for p in ax.patches:
    width = p.get_width()
    ax.annotate(f'{width:.1f}%', (width + 1, p.get_y() + p.get_height() / 2),
                ha='left', va='center', fontsize=10, color='black', fontweight='bold')

plt.tight_layout()
plt.savefig('resume_matching_results.png', dpi=300)
plt.show()