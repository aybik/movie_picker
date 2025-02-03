import pandas as pd
from surprise import Dataset, Reader
from surprise import KNNBasic, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise.dataset import DatasetAutoFolds

def load_data(df):
    df = df.rename(columns={'key_a': 'movie_id'}, inplace=True)
    reader = Reader(rating_scale=(0,5)) # rating scale range (from 0 to 5)
    data = Dataset.load_from_df(df[['user_name', 'movie_id', 'rating']], reader) # load a dataset from a pandas dataframe

    return data
