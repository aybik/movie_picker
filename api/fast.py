import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from moviepicker.ml_logic import model
import pickle
import os
import re
import requests
import string


app = FastAPI()
app.state.model = pickle.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)),"models/knn_model.pkl"), "rb"))
app.state.data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),"final_set_a.csv"))
app.state.full_data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),"full_df.csv"))
app.state.matrix = pickle.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)),"models/matrix.pkl"), "rb"))


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(
        input_name: str,
        n_recommendations: int
    ):

    result = model.get_similar_movies_knn(app.state.model, app.state.matrix, app.state.data, input_name, "key_b", n_recommendations)
    return [movie["input_name"] for movie in result]



@app.get("/find")
def find_movies(input_name, dataset_choice):
    if dataset_choice == "full":
        df = app.state.full_data
    elif dataset_choice == "final":
        df = app.state.data
    movie_name = input_name.lower()
    df['name'] = df.name.apply(lambda x: x.lower())
    same_movies_df = df[df.name == movie_name]
    return list(same_movies_df.key_b)



@app.get("/get_url")
def get_url(movie):

    df = app.state.full_data

    if movie in df.key_b.values:
        film_id = df[df["key_b"] == movie]["film_id"].iloc[0]
        return f"https://letterboxd.com/film/{film_id}/"

    def clean_movie_url(movie):
        # Remove parentheses
        movie = re.sub(r"[()]", "", movie)
        # Replace all punctuation with a space
        movie = re.sub(f"[{re.escape(string.punctuation)}]", " ", movie)
        # Replace spaces with hyphens and convert to lowercase
        movie_modified = re.sub(r'\s+', '-', movie.lower().strip())

        return f"https://letterboxd.com/film/{movie_modified}/"

    movies_list = find_movies(movie[:-7], "full")
    sorted_movies_list = sorted(movies_list)
    if movie == sorted_movies_list[0]:
        true_url = clean_movie_url(movie[:-7])
    else:
        true_url = clean_movie_url(movie)

    return true_url

@app.get("/get_image")
def get_image(movie):

    df = app.state.full_data
    if movie in df.key_b.values:
        poster = df.loc[df.key_b == movie, "poster"].iloc[0]
        return poster
    else:
        return None
@app.get("/get_description")
def get_description(movie):
    df = app.state.full_data
    result = df[df["key_b"] == movie]["description"]
    return result.iloc[0]


@app.get("/")
def root():
    return {'greeting': 'Hello'}
