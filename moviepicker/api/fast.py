import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from moviepicker.ml_logic import archived_preprocessor, model
import pickle
import os

app = FastAPI()
app.state.model = pickle.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)),"models/knn_model.pkl"), "rb"))
app.state.data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),"final_set_a.csv")) ###EDIT
app.state.matrix = pickle.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)),"models/matrix.pkl"), "rb"))


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?input_name=Inception%20%282010%29&name_column=key_b
# http://127.0.0.1:8000/predict?input_name=Inception&name_column=name

# http://127.0.0.1:8000/predict?input=Inception%20%282010%29&type=key_b
# http://127.0.0.1:8000/predict?input=Inception&type=name

@app.get("/predict")
def predict(
        input: str,
        type: str,
    ):      # 1
    """
    Make a single course prediction.
    Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
    Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
    """
    # YOUR CODE HERE
    result = model.get_similar_movies_knn(app.state.model, app.state.matrix, app.state.data, input, type)
    return result

@app.get("/")
def root():
    # YOUR CODE HERE
    return {'greeting': 'Hello'}
