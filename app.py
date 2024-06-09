import streamlit as st
import pandas as pd
import pickle
from fuzzywuzzy import process
from gensim.models import Word2Vec

st.set_page_config(layout="wide")

# Load the pickled DataFrame and similarity matrix
df_sample = pickle.load(open('df.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Train a Word2Vec model on job titles
title_corpus = [title.split() for title in df_sample['Title'].values]
word2vec_model = Word2Vec(sentences=title_corpus, vector_size=100, window=5, min_count=1, sg=1)

# Define recommendation function with fuzzy matching
def recommend(title, num_recommendations=20):
    # Find the best match for the input title in the dataset
    best_match = process.extractOne(title, df_sample['Title'].values)

    # Check if the match score is above a threshold
    if best_match and best_match[1] > 60:  # Adjust the threshold as needed
        indx = df_sample[df_sample['Title'] == best_match[0]].index[0]
        indx = df_sample.index.get_loc(indx)
        distances = sorted(list(enumerate(similarity[indx])), key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]

        # Get the recommended job details
        jobs = [{
            'Title': df_sample.iloc[i[0]].Title,
            'Company': df_sample.iloc[i[0]].Company,
            'Job Description': df_sample.iloc[i[0]]['Job.Description'],
            'Location': df_sample.iloc[i[0]]['Location'],
            'Salary': df_sample.iloc[i[0]]['Salary']
        } for i in distances]
        return jobs
    else:
        return None

# Assume you have a function to collect user preferences and store them in a DataFrame called user_preferences
def collect_user_preferences():
    # Code to collect user preferences
    user_preferences = pd.DataFrame()  # Example: Placeholder for user preferences
    return user_preferences

# Assume you have a function to incorporate user preferences into recommendation
def incorporate_user_preferences(recommendations, user_preferences):
    # Code to incorporate user preferences into recommendations
    # Example: Filter recommendations based on user preferences
    personalized_recommendations = recommendations  # Placeholder for personalized recommendations
    return personalized_recommendations

# Center elements using HTML and CSS
st.markdown("""
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .stTextInput, .stButton {
        margin: auto;
    }
    .stColumn {
        border: 1px solid #ddd;
        padding: 10px;
        margin: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit App
st.markdown('<h1 class="center">Job Recommendation System</h1>', unsafe_allow_html=True)

# Center the input and button
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.write('')
with col2:
    job_title = st.text_input('Enter job title:', key='job_title', label_visibility='visible')
with col3:
    st.write('')

center_button = st.empty()
center_button.markdown('<div class="center"></div>', unsafe_allow_html=True)
if center_button.button('Recommend'):
    if job_title:
        recommendations = recommend(job_title)
        if recommendations:
            # Collect user preferences
            user_preferences = collect_user_preferences()
            # Incorporate user preferences into recommendations
            personalized_recommendations = incorporate_user_preferences(recommendations, user_preferences)

            # Streamlit App: Add filters for location, salary range, required skills, experience level, etc.
            location_filter = st.selectbox('Location:', options=['All'] + df_sample['Location'].unique().tolist())
            salary_range_filter = st.slider('Salary Range:', min_value=0, max_value=200000, value=(0, 200000), step=1000)
            # Add more filters as needed

            # Filter recommendations based on user-selected criteria
            filtered_recommendations = personalized_recommendations
            if location_filter != 'All':
                filtered_recommendations = [rec for rec in filtered_recommendations if rec['Location'] == location_filter]
            if salary_range_filter != (0, 200000):
                filtered_recommendations = [rec for rec in filtered_recommendations if rec['Salary'] >= salary_range_filter[0] and rec['Salary'] <= salary_range_filter[1]]
            # Apply more filters as needed

            st.write('Recommendations:')
            col1, col2, col3 = st.columns(3)  # Create three columns
            for idx, job in enumerate(filtered_recommendations, 1):
                formatted_title = " ".join([word.capitalize() for word in job['Title'].split()])
                formatted_company = " ".join([word.capitalize() for word in job['Company'].split()])
                main_line = f"<b>{formatted_title}</b> at <b>{formatted_company}</b>"
                job_desc = job['Job Description']

                if idx % 3 == 1:
                    with col1:
                        st.markdown(f'<div class="stColumn">{main_line}<br><br><strong>Job Description:</strong><br>{job_desc}</div>', unsafe_allow_html=True)
                elif idx % 3 == 2:
                    with col2:
                        st.markdown(f'<div class="stColumn">{main_line}<br><br><strong>Job Description:</strong><br>{job_desc}</div>', unsafe_allow_html=True)
                else:
                    with col3:
                        st.markdown(f'<div class="stColumn">{main_line}<br><br><strong>Job Description:</strong><br>{job_desc}</div>', unsafe_allow_html=True)
        else:
            st.write('No recommendations found. Please check the job title and try again.')
    else:
        st.write('Please enter a job title.')
