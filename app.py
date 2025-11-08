import streamlit as st
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download("punkt_tab")
import pickle
import requests
import os

def fetch_poster(movie_id):
       API_KEY = os.getenv("MOVIES_API_TOKEN")
       url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
       response = requests.get(url)
       data = response.json()
       poster_path = data.get("poster_path")
       return "https://www.themoviedb.org/t/p/w1280" + poster_path

st.header("Movie Recommender System")
df= pickle.load(open("Artifacts/Models/data.pkl","rb"))
movies_list = df["title"]

model = pickle.load(open("Artifacts/Models/sim.pkl","rb"))

movies = st.selectbox('Select the Movie',movies_list.values)

def recommend(movie_title):
    recommend_movie = []
    recommend_poster = []

    movie_index = df[df['title'].str.lower() == movie_title.lower()].index[0]
    distances = model[movie_index]
    similar_movies = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]

    for i in similar_movies:
        movie_id = df.iloc[i[0]].movie_id  # ✅ fetch ID for each similar movie
        recommend_movie.append(df.iloc[i[0]].title)
        recommend_poster.append(fetch_poster(movie_id))

    return recommend_movie, recommend_poster


if st.button("Recommend Movie"):
     movie,poster = recommend(movies)
     cols= st.columns(5)
     for idx,col in enumerate(cols):
             st.subheader(movie[idx])
             st.image(poster[idx],width=100)

     