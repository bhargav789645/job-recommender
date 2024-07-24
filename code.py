import pandas as pd
import numpy as np
import pickle
!npm install streamlit
streamlit run app.py
import streamlit as st

# Load the data
df1 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/dice_com-job_us_sample.csv')

# Preprocess the data
df1.dropna(subset=['jobdescription'], inplace=True)

# Vectorize the job descriptions
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df1['jobdescription'])

# Compute the sigmoid kernel
from sklearn.metrics.pairwise import sigmoid_kernel
cosine_sim = sigmoid_kernel(tfidf_matrix, tfidf_matrix)

# Create a reverse map of indices and job titles
indices = pd.Series(df1.index, index=df1['jobtitle']).drop_duplicates()

# Function to get recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 similar jobs
    job_indices = [i[0] for i in sim_scores]
    return df1['jobtitle'].iloc[job_indices]

# Save necessary data
new1 = df1[['jobdescription', 'jobtitle']]
new1.to_csv('new1.csv')
pickle.dump(new1, open('job_list.pkl', 'wb'))
pickle.dump(cosine_sim, open('cosine_sim.pkl', 'wb'))

# Streamlit app
st.title('Job Recommendation System')

# Load data for the app
jobs_df = pd.read_csv('new1.csv')

# Select a job title
selected_job_title = st.selectbox('Select a Job', jobs_df['jobtitle'].tolist())

if st.button('Show Recommendation'):
    recommended_jobs = get_recommendations(selected_job_title)
    st.subheader('Recommended Jobs:')
    for job in recommended_jobs:
        st.write(job)

# To run Streamlit app
!npm install localtunnel
!npx localtunnel --port 8501
