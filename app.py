
import streamlit as st
import pandas as pd
import redis
import json

st.set_page_config(
    page_title="Movie Recommendation Engine",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Personalized Movie Recommendation Engine")
st.write("Discover movies using AI-powered recommendations.")

# Load movie data
@st.cache_data
def load_movies():
    return pd.read_csv("data/processed/clean_movies.csv")

movies = load_movies()

# Connect to Redis
def get_redis():
    try:
        client = redis.Redis(
            host="localhost",
            port=6379,
            decode_responses=True
        )
        client.ping()
        return client
    except redis.RedisError:
        return None

st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Choose a section",
    ["Home", "Popular Movies", "Genre Recommendations", "Model Evaluation"]
)

if page == "Home":

    st.subheader("Welcome!")

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset", "MovieLens 25M")
    col2.metric("Recommendation Model", "Two-Tower")
    col3.metric("Caching", "Redis")

    st.write(
        "This project uses machine learning to generate "
        "personalized movie recommendations."
    )

elif page == "Popular Movies":

    st.subheader("Top 10 Popular Movies")

    try:
        popular = pd.read_csv(
            "reports/cold_start_popular.csv"
        )

        st.dataframe(
            popular.head(10),
            use_container_width=True
        )

    except FileNotFoundError:
        st.error("Popular movie report not found.")

elif page == "Genre Recommendations":

    st.subheader("Find Movies by Genre")

    genre = st.selectbox(
        "Select a genre",
        ["Action", "Comedy", "Drama", "Romance",
         "Thriller", "Animation", "Adventure", "Horror"]
    )

    try:
        popular = pd.read_csv(
            "reports/cold_start_popular.csv"
        )

        recommendations = popular[
            popular["genres"].str.contains(
                genre,
                case=False,
                na=False
            )
        ]

        st.dataframe(
            recommendations.head(10),
            use_container_width=True
        )

    except FileNotFoundError:
        st.error("Popular movie report not found.")

elif page == "Model Evaluation":

    st.subheader("Model Performance")

    try:
        evaluation = pd.read_csv(
            "reports/evaluation_results.csv"
        )

        st.dataframe(
            evaluation,
            use_container_width=True
        )

        for _, row in evaluation.iterrows():
            st.metric(
                row["metric"],
                round(row["value"], 5)
            )

    except FileNotFoundError:
        st.error("Evaluation report not found.")

st.sidebar.success("Recommendation Engine Ready!")
