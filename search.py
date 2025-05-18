import streamlit as st
import pandas as pd
import numpy as np
import re
import faiss
import requests
from bs4 import BeautifulSoup
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(
    page_title="Job Resource Recommender",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    body, .stApp {
        background-color: #1a1a1a !important; /* Darker shade of #2e2e2e */
        color: #D6CFE1 !important;
    }
    .main-header {
        font-size: 2.5rem;
        color: #D6CFE1;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #D6CFE1;
    }
    .resource-card {
        background-color: #232129; /* Very dark background for cards */
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 10px;
        border-left: 5px solid #D6CFE1;
    }
    .resource-title {
        font-weight: bold;
        font-size: 1.2rem;
        color: #D6CFE1;
    }
    .resource-desc {
        font-size: 1rem;
        color: #D6CFE1;
    }
    .resource-meta {
        font-size: 0.8rem;
        color: #D6CFE1;
    }
    .source-tag {
        background-color: #232129;
        border-radius: 15px;
        padding: 5px 10px;
        margin-right: 5px;
        font-size: 0.8rem;
        color: #D6CFE1;
        border: 1px solid #D6CFE1;
    }
    .stButton>button {
        background-color: #D6CFE1;
        color: #232129;
    }
    .stButton>button:hover {
        background-color: #232129;
        color: #D6CFE1;
        border: 1px solid #D6CFE1;
    }
    .stTextArea>div>div>textarea {
        background-color: #232129;
        color: #D6CFE1;
        border: 1px solid #D6CFE1;
    }
    .stSidebar, .css-1d391kg, .css-1lcbmhc {
        background-color: #1a1a1a !important;
        color: #D6CFE1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Application title
st.markdown("<h1 class='main-header'>Job Skill Resource Finder</h1>", unsafe_allow_html=True)
st.markdown("Find free open-source learning resources tailored to your job requirements")

# Create a database of free educational resources with embedded vectors
@st.cache_data
def load_resources():
    # This would typically scrape or load from an API, but we'll mock it for the demo
    resources = [
        {
            "title": "JavaScript Algorithms and Data Structures",
            "description": "Learn JavaScript fundamentals, ES6, regular expressions, debugging, data structures, OOP, functional programming, and algorithmic thinking",
            "skills": "javascript, data structures, algorithms, programming, es6, functional programming, oop",
            "duration": "300 hours",
            "source": "freeCodeCamp",
            "url": "https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures/"
        },
        {
            "title": "Responsive Web Design Certification",
            "description": "Learn HTML, CSS, visual design, accessibility, and responsive web design principles",
            "skills": "html, css, responsive design, web development, frontend, accessibility",
            "duration": "300 hours",
            "source": "freeCodeCamp",
            "url": "https://www.freecodecamp.org/learn/responsive-web-design/"
        },
        {
            "title": "Scientific Computing with Python",
            "description": "Python fundamentals, Python for data science, and completing scientific computing projects",
            "skills": "python, data science, scientific computing, programming",
            "duration": "300 hours",
            "source": "freeCodeCamp",
            "url": "https://www.freecodecamp.org/learn/scientific-computing-with-python/"
        },
        {
            "title": "Data Analysis with Python",
            "description": "Learn data analysis techniques using NumPy, Pandas, Matplotlib, and Seaborn",
            "skills": "python, data analysis, pandas, numpy, matplotlib, visualization",
            "duration": "300 hours",
            "source": "freeCodeCamp",
            "url": "https://www.freecodecamp.org/learn/data-analysis-with-python/"
        },
        {
            "title": "Information Security",
            "description": "Learn information security with HelmetJS, Python for penetration testing, and security concepts",
            "skills": "security, information security, python, penetration testing, cybersecurity",
            "duration": "300 hours",
            "source": "freeCodeCamp",
            "url": "https://www.freecodecamp.org/learn/information-security/"
        },
        {
            "title": "Machine Learning with Python",
            "description": "Learn TensorFlow and various machine learning algorithms and techniques",
            "skills": "machine learning, python, tensorflow, deep learning, neural networks, ai",
            "duration": "300 hours",
            "source": "freeCodeCamp",
            "url": "https://www.freecodecamp.org/learn/machine-learning-with-python/"
        },
        {
            "title": "Learn Python - Full Course for Beginners",
            "description": "A complete Python tutorial covering all the basics for beginners",
            "skills": "python, programming, beginners, fundamentals",
            "duration": "4 hours",
            "source": "freeCodeCamp YouTube",
            "url": "https://www.youtube.com/watch?v=rfscVS0vtbw"
        },
        {
            "title": "Learn SQL - Full Database Course for Beginners",
            "description": "A comprehensive introduction to SQL and database concepts",
            "skills": "sql, database, data engineering, postgresql, mysql",
            "duration": "4 hours",
            "source": "freeCodeCamp YouTube",
            "url": "https://www.youtube.com/watch?v=HXV3zeQKqGY"
        },
        {
            "title": "React Tutorial for Beginners",
            "description": "Learn the React JavaScript library from the ground up",
            "skills": "react, javascript, frontend, web development",
            "duration": "10 hours",
            "source": "freeCodeCamp YouTube",
            "url": "https://www.youtube.com/watch?v=bMknfKXIFA8"
        },
        {
            "title": "Git and GitHub for Beginners - Crash Course",
            "description": "Learn the basics of Git version control and GitHub",
            "skills": "git, github, version control, collaboration",
            "duration": "1 hour",
            "source": "freeCodeCamp YouTube",
            "url": "https://www.youtube.com/watch?v=RGOj5yH7evk"
        },
        {
            "title": "Docker Tutorial for Beginners",
            "description": "Full Docker course teaching containerization from scratch",
            "skills": "docker, devops, containerization, deployment",
            "duration": "3 hours",
            "source": "freeCodeCamp YouTube",
            "url": "https://www.youtube.com/watch?v=fqMOX6JJhGo"
        },
        {
            "title": "The Rust Programming Language Tutorial",
            "description": "Learn systems programming with Rust",
            "skills": "rust, systems programming, low-level, performance",
            "duration": "3 hours",
            "source": "freeCodeCamp YouTube",
            "url": "https://www.youtube.com/watch?v=MsocPEZBd-M"
        },
        {
            "title": "Object Oriented Programming in Python",
            "description": "Master OOP concepts using Python",
            "skills": "python, oop, object oriented programming, classes, inheritance",
            "duration": "1.5 hours",
            "source": "Programiz",
            "url": "https://www.programiz.com/python-programming/object-oriented-programming"
        },
        {
            "title": "Learn Node.js",
            "description": "Comprehensive guide to server-side JavaScript with Node.js",
            "skills": "nodejs, javascript, backend, server, express, api",
            "duration": "Various",
            "source": "MDN Web Docs",
            "url": "https://developer.mozilla.org/en-US/docs/Learn/Server-side/Node_server_without_framework"
        },
        {
            "title": "React Hooks Tutorial",
            "description": "Modern React development using functional components and hooks",
            "skills": "react, javascript, hooks, frontend, web development",
            "duration": "Various",
            "source": "React Docs",
            "url": "https://reactjs.org/docs/hooks-intro.html"
        },
        {
            "title": "Data Visualization with D3.js",
            "description": "Learn to create interactive data visualizations for the web",
            "skills": "d3js, data visualization, javascript, svg, web development",
            "duration": "Various",
            "source": "Observable",
            "url": "https://observablehq.com/@d3/learn-d3"
        },
        {
            "title": "Learn AWS Serverless",
            "description": "Build serverless applications on AWS",
            "skills": "aws, serverless, lambda, cloud computing, backend",
            "duration": "Various",
            "source": "AWS Workshops",
            "url": "https://aws.amazon.com/getting-started/hands-on/build-serverless-web-app-lambda-apigateway-s3-dynamodb-cognito/"
        },
        {
            "title": "Django Web Framework",
            "description": "Build web applications quickly with Django",
            "skills": "python, django, web development, backend, mvc, databases",
            "duration": "Various",
            "source": "Django Project",
            "url": "https://docs.djangoproject.com/en/stable/intro/tutorial01/"
        },
        {
            "title": "Flutter Mobile App Development",
            "description": "Build cross-platform mobile apps with Flutter",
            "skills": "flutter, dart, mobile development, ui design, cross-platform",
            "duration": "Various",
            "source": "Flutter Dev",
            "url": "https://flutter.dev/docs/get-started/codelab"
        },
        {
            "title": "CSS Grid and Flexbox",
            "description": "Master modern CSS layout techniques",
            "skills": "css, web development, frontend, responsive design, layout",
            "duration": "Various",
            "source": "CSS-Tricks",
            "url": "https://css-tricks.com/snippets/css/complete-guide-grid/"
        }
    ]
    
    return pd.DataFrame(resources)

# Initialize FAISS index and TF-IDF vectorizer
@st.cache_resource
def setup_faiss(df):
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Create corpus by combining skills and descriptions
    corpus = df['skills'] + " " + df['description']
    
    # Fit and transform corpus to TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(corpus)
    
    # Convert sparse matrix to dense for FAISS
    dense_vectors = tfidf_matrix.toarray().astype('float32')
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(dense_vectors)
    
    # Create FAISS index
    dimension = dense_vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
    index.add(dense_vectors)
    
    return index, tfidf, dense_vectors

# Function to extract skills from job description
def extract_skills(job_description):
    # This would typically use NLP to extract skills
    # For demo purposes, we'll use a simple tokenization approach
    common_skills = [
        "python", "javascript", "java", "c++", "ruby", "php", "html", "css",
        "react", "angular", "vue", "node", "express", "django", "flask",
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "sql", "mysql", "postgresql", "mongodb", "nosql", "database",
        "machine learning", "ai", "data science", "nlp", "computer vision",
        "git", "devops", "ci/cd", "jenkins", "github actions", "gitlab",
        "agile", "scrum", "kanban", "project management", "jira",
        "rest api", "graphql", "microservices", "serverless",
        "linux", "unix", "bash", "shell scripting",
        "frontend", "backend", "fullstack", "web development",
        "mobile development", "ios", "android", "flutter", "react native",
        "testing", "qa", "selenium", "cypress", "junit", "pytest",
        "security", "cybersecurity", "penetration testing",
        "data analysis", "data visualization", "tableau", "power bi",
        "blockchain", "ethereum", "smart contracts",
        "ux", "ui", "design", "figma", "adobe xd", "sketch"
    ]
    
    job_description = job_description.lower()
    found_skills = []
    
    for skill in common_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', job_description):
            found_skills.append(skill)
    
    return found_skills

# Function to search for relevant resources
def search_resources(job_description, index, tfidf, dense_vectors, df, k=5):
    # Extract key skills
    extracted_skills = extract_skills(job_description)
    skills_text = " ".join(extracted_skills)
    
    # Combine skills with job description for better matching
    search_text = job_description.lower() + " " + skills_text
    
    # Transform search text using the same TF-IDF vectorizer
    search_vector = tfidf.transform([search_text]).toarray().astype('float32')
    
    # Normalize the search vector
    faiss.normalize_L2(search_vector)
    
    # Search for similar vectors in the FAISS index
    distances, indices = index.search(search_vector, k)
    
    # Get the matching resources
    results = df.iloc[indices[0]].copy()
    results['score'] = distances[0] * 100  # Convert similarity score to percentage
    
    return results, extracted_skills

# Main app
def main():
    # Load resources
    df = load_resources()
    
    # Setup FAISS index
    index, tfidf, dense_vectors = setup_faiss(df)
    
    # Sidebar for input
    st.sidebar.markdown("## Input Job Details")
    
    # Job description input
    job_description = st.sidebar.text_area(
        "Paste Job Description or Skills Required",
        """
We are looking for a skilled Python Developer to join our data science team. 
The ideal candidate should have expertise in Python programming, data analysis, and machine learning. 
Experience with pandas, numpy, scikit-learn, and TensorFlow is required.
Knowledge of SQL and database concepts is a must.
        """,
        height=300
    )
    
    # Additional filters
    st.sidebar.markdown("## Additional Filters")
    sources = list(df['source'].unique())
    selected_sources = st.sidebar.multiselect(
        "Filter by Source",
        sources,
        default=sources
    )
    
    # Search button
    search_clicked = st.sidebar.button("Find Resources")
    
    # About section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        """
        This app uses FAISS (Facebook AI Similarity Search) to find relevant learning 
        resources based on job descriptions. It extracts key skills from your job 
        description and matches them with free and open-source educational content.
        """
    )
    
    # Main content
    if search_clicked and job_description:
        # Search for relevant resources
        results, extracted_skills = search_resources(job_description, index, tfidf, dense_vectors, df, k=10)
        
        # Filter by selected sources
        if selected_sources:
            results = results[results['source'].isin(selected_sources)]
        
        # Display extracted skills
        st.markdown("### Extracted Skills")
        skill_html = ' '.join([f'<span class="source-tag">{skill}</span>' for skill in extracted_skills])
        st.markdown(f"<div>{skill_html}</div>", unsafe_allow_html=True)
        
        # Display results
        st.markdown("### Recommended Learning Resources")
        
        if len(results) > 0:
            for _, resource in results.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="resource-card">
                        <div class="resource-title">{resource['title']}</div>
                        <div class="resource-desc">{resource['description']}</div>
                        <div class="resource-meta">
                            <span><b>Source:</b> {resource['source']}</span> | 
                            <span><b>Duration:</b> {resource['duration']}</span> | 
                            <span><b>Match Score:</b> {resource['score']:.1f}%</span>
                        </div>
                        <div style="margin-top: 10px;">
                            <a href="{resource['url']}" target="_blank">Open Resource</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No matching resources found. Try adjusting your search filters.")
        
        # Feature to suggest missing resources
        st.markdown("---")
        st.markdown("### Can't Find What You Need?")
        col1, col2 = st.columns([3, 1])
        with col1:
            missing_resource = st.text_input("Suggest a resource to add:", placeholder="Name of tutorial or course...")
        with col2:
            if st.button("Submit Suggestion"):
                st.success("Thank you for your suggestion! We'll consider adding it to our database.")
    
    else:
        # Display welcome message and instructions
        st.markdown("""
        <h3 style="text-align: center;">Welcome to the Job Skill Resource Finder</h3>
        
        This tool helps you find free, open-source learning resources that match the skills required in your job description.
        
        **How to use:**
        1. Paste a job description or list of required skills in the sidebar
        2. Optionally filter by resource source
        3. Click "Find Resources" to get personalized recommendations
        
        **Features:**
        - Automatically extracts key skills from job descriptions
        - Uses FAISS for efficient similarity search
        - Focuses on free and open-source content
        - Provides match scores to help you prioritize learning
        
        **Get started by entering a job description in the sidebar!**
        """, unsafe_allow_html=True)
        
        # Sample job roles for quick selection
        st.markdown("### Try these sample job roles:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Data Scientist"):
                job_desc = """
                Data Scientist position requiring expertise in Python, machine learning algorithms, 
                statistical analysis, deep learning, and data visualization. Experience with TensorFlow, 
                PyTorch, and data manipulation using Pandas and NumPy is essential. SQL skills required.
                """
                st.session_state.job_description = job_desc
                st.experimental_rerun()
        with col2:
            if st.button("Full-Stack Developer"):
                job_desc = """
                Full-Stack Developer needed with strong JavaScript skills. Experience with React, 
                Node.js, Express, and MongoDB required. Knowledge of HTML/CSS, RESTful APIs, and 
                version control systems like Git is essential. DevOps experience is a plus.
                """
                st.session_state.job_description = job_desc
                st.experimental_rerun()
        with col3:
            if st.button("DevOps Engineer"):
                job_desc = """
                DevOps Engineer with expertise in CI/CD pipelines, Docker, Kubernetes, and cloud platforms 
                (AWS/Azure). Experience with infrastructure as code using Terraform or CloudFormation required. 
                Scripting skills in Python or Bash essential. Knowledge of monitoring and logging solutions a plus.
                """
                st.session_state.job_description = job_desc
                st.experimental_rerun()
        
        # Display some featured resources
        st.markdown("### Featured Free Learning Resources")
        featured_df = df.sample(n=3).reset_index(drop=True)
        
        for _, resource in featured_df.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="resource-card">
                    <div class="resource-title">{resource['title']}</div>
                    <div class="resource-desc">{resource['description']}</div>
                    <div class="resource-meta">
                        <span><b>Source:</b> {resource['source']}</span> | 
                        <span><b>Duration:</b> {resource['duration']}</span>
                    </div>
                    <div style="margin-top: 10px;">
                        <a href="{resource['url']}" target="_blank">Open Resource</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()