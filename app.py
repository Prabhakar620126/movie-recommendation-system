import streamlit as st
import pickle
import pandas as pd
import requests
import os 

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# ============================
# PAGE CONFIG
# ============================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ============================
# OMDb API KEY
# ============================

API_KEY = os.getenv("OMDB_API_KEY", "bb9124a7")

# ============================
# LOAD DATA
# ============================

movies = pickle.load(open("movie1.pkl", "rb"))
movies = pd.DataFrame(movies)

vectors = load_npz("vectors.npz")

# ============================
# FETCH POSTER
# ============================

def fetch_poster(movie_name):

    url = f"https://www.omdbapi.com/?t={movie_name}&apikey={API_KEY}"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if data.get("Response") == "True":
            return data.get("Poster")

        return "https://via.placeholder.com/300x450.png?text=No+Poster"

    except:
        return "https://via.placeholder.com/300x450.png?text=No+Poster"

# ============================
# RECOMMEND FUNCTION
# ============================

def recommend(movie_name):

    movie_index = movies[movies["title"] == movie_name].index[0]

    similarity_scores = cosine_similarity(
        vectors[movie_index],
        vectors
    ).flatten()

    movie_list = sorted(
        list(enumerate(similarity_scores)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movie_list:

        movie_title = movies.iloc[i[0]].title

        recommended_movies.append(movie_title)

        recommended_posters.append(
            fetch_poster(movie_title)
        )

    return recommended_movies, recommended_posters

# ============================
# SIDEBAR
# ============================

with st.sidebar:

    st.title("🎬 Movie Recommendation")

    st.markdown("---")

    st.subheader("Algorithm")

    st.write("✔ Content-Based Filtering")

    st.write("✔ CountVectorizer")

    st.write("✔ Cosine Similarity")

# ============================
# TITLE
# ============================

st.title("🎬 Movie Recommendation System")

movie_names = movies["title"].values

selected_movie = st.selectbox(
    "Select a Movie",
    movie_names
)

# ============================
# BUTTON
# ============================

if st.button("Recommend"):

    names, posters = recommend(selected_movie)

    cols = st.columns(5)

    for i in range(5):

        with cols[i]:

            st.image(posters[i])

            st.caption(names[i])