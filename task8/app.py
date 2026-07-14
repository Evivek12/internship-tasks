import streamlit as st
import pickle

# Load files
movies = pickle.load(open("movie_list.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

# Recommendation function
def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for i in movie_list:
        recommendations.append(movies.iloc[i[0]].title)

    return recommendations

# Streamlit UI
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬"
)

st.title("🎬 AI Movie Recommendation System")

selected_movie = st.selectbox(
    "Select a Movie",
    movies["title"].values
)

if st.button("Recommend"):

    recommended_movies = recommend(selected_movie)

    st.success("Recommended Movies")

    for movie in recommended_movies:
        st.write("⭐", movie)