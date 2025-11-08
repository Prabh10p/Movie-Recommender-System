import streamlit as st
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download("punkt_tab")
import pickle


st.header("Movie Recommender System")
df= pickle.load(open("Artifacts/Models/data.pkl","rb"))
movies_list = df["title"]

model = pickle.load(open("Artifacts/Models/sim.pkl","rb"))

movies = st.selectbox('Select the Movie',movies_list.values)

if st.button("Recommend Similar Movies"):
    def recommend(movie_title):
    # find index of the movie
     movie_index = df[df['title'].str.lower() == movie_title.lower()].index[0]
    
     distances = model[movie_index]
    
     similar_movies = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)
    
     for i in similar_movies[1:6]:  # skip the first one (itself)
        st.write(df.iloc[i[0]].title)

    recommend(movies)
     