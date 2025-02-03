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

def train_test_split(data):
    trainset, testset = train_test_split(data, test_size=0.25)

    return trainset, testset

def model_fit(trainset):
    svd_algo = SVD()
    svd_algo = svd_algo.fit(trainset)

    return svd_algo

def model_test(testset, svd_algo):
    predictions = svd_algo.test(testset)
    test_rmse = accuracy.rmse(predictions)

    return predictions, test_rmse
