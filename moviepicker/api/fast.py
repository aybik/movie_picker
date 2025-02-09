import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from moviepicker.ml_logic import advanced_model
import pickle
import os
import re
import requests
import string
from tensorflow.keras.models import load_model
# from typing import List

app = FastAPI()
parent_dir = os.path.dirname(os.path.dirname(__file__))

# Load pickle files #save
app.state.model = pickle.load(open(os.path.join(parent_dir,"models/knn_20.pkl"), "rb"))
app.state.latent_embeddings = pickle.load(open(os.path.join(parent_dir,"models/latent_embeddings.pkl"), "rb"))
app.state.vectorizer = pickle.load(open(os.path.join(parent_dir,"models/vectorizer.pkl"), "rb"))
app.state.encoder_trained = load_model(os.path.join(parent_dir,"models/encoder_trained.keras"))

# Load dataframes
app.state.data = pd.read_pickle(os.path.join(parent_dir,"data_encode.pkl"))
app.state.full_data = pd.read_pickle(os.path.join(parent_dir,"streamlit.pkl"))

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict_name")
def predict(
        input_name: str,
        n_recommendations: int
    ):
    result = advanced_model.get_movie_recommendations(input_name, app.state.data, app.state.model, app.state.latent_embeddings, n_recommendations)
    return result

@app.get("/predict_filter")
def predict(
        user_description: str,
        user_language: str,
        # user_genres: List[str],
        user_genres: str,
        n_recommendations: int
    ):
    result = advanced_model.recommend_movies_by_details_new(user_description, app.state.data, app.state.encoder_trained, app.state.model, app.state.vectorizer, user_language, user_genres, n_recommendations)
    return result

@app.get("/find")
# def find_movies(input_name, dataset_choice):
#     if dataset_choice == "full":
#         df = app.state.full_data
#     elif dataset_choice == "final":
#         df = app.state.data
#     movie_name = input_name.lower()
#     df['name'] = df.name.apply(lambda x: x.lower())
#     same_movies_df = df[df.name == movie_name]
#     return list(same_movies_df.key)
def find_movies(input_name, dataset_choice):
    movie_name_words = set(input_name.lower().split())
    def word_match(name):
        name_words = set(name.lower().split())  # Convert movie name in df to a set of words
        return not movie_name_words.isdisjoint(name_words)  # Check if there is an exact word match
    if dataset_choice == "full":
        df = app.state.full_data
        same_movies_df = df[df['name'].apply(word_match)]
    elif dataset_choice == "final":
        df = app.state.data
        same_movies_df = df[df['name_streamlit'].apply(word_match)]
    return list(same_movies_df.key)

@app.get("/get_url")
def get_url(movie):
    film_id = app.state.full_data[app.state.full_data["key"] == movie]["film_id"].iloc[0]
    return app.state.full_data.loc[app.state.full_data['film_id']==film_id, 'streamlit_url'].iloc[0]

@app.get("/get_image")
def get_image(movie):

    df = app.state.full_data
    if movie in df.key.values:
        poster = df.loc[df.key == movie, "poster"].iloc[0]
        return poster
    else:
        return None

@app.get("/get_description")
def get_description(movie):
    df = app.state.full_data
    result = df[df["key"] == movie]["description"]
    return result.iloc[0]

@app.get("/")
def root():
    return {'greeting': 'Hello'}
