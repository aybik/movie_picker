import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from movie_picker.moviepicker import preprocessor, model
import pickle

app = FastAPI()
app.state.model = pickle.load(open(os.path.join(os.path.dirname(__file__),"models/knn_model.pkl"), "rb"))
app.state.data = pd.read_csv(os.path.join(os.path.dirname(__file__),"final_set_a.csv")) ###EDIT
app.state.matrix = pickle.load(open(os.path.join(os.path.dirname(__file__),"models/matrix.pkl"), "rb"))


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2014-07-06+19:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2
@app.get("/predict")
def predict(
        input_name: str,
        name_column: str,
    ):      # 1
    """
    Make a single course prediction.
    Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
    Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
    """
    # YOUR CODE HERE
    result = model.get_similar_movies_knn(app.state.model, app.state.matrix, app.state.data, input_name, name_column)
    return {'movies': result}

@app.get("/")
def root():
    # YOUR CODE HERE
    return {'greeting': 'Hello'}
