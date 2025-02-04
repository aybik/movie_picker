import os
import pickle

import pandas as pd

from surprise import Dataset, Reader
from surprise import KNNBasic, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise.dataset import DatasetAutoFolds

def save_pickle(my_obj, filepath):
    if os.path.isfile(filepath):
        print(f"File {filepath} already exists. Doing nothing")
        return None
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(my_obj, f)


def load_pickle(filepath):
    if not os.path.isfile(filepath):
        print(f"File {filepath} does not exist.")
    else:
        with open(filepath, 'rb') as f:
            return pickle.load(f)

def load_data(df):
    reader = Reader(rating_scale=(0,5)) # rating scale range (from 0 to 5)
    data = Dataset.load_from_df(df[['user_name', 'film_id', 'rating']], reader) # load a dataset from a pandas dataframe

    save_pickle(data, '../../artifacts/dataset_b_surprise_dataset.pkl')

    return data

def train_test_split(data):
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

    save_pickle(trainset, '../../artifacts/trainset_svd_algo_rseed42.pkl')
    save_pickle(testset, '../../artifacts/testset_svd_algo_rseed42.pkl')

    return trainset, testset

def model_fit(trainset):
    svd_algo = SVD(random_state=42)
    svd_algo = svd_algo.fit(trainset)

    save_pickle(svd_algo, '../../artifacts/fitted_svd_algo_rseed42.pkl')

    return svd_algo

def model_test(testset, svd_algo):
    predictions = svd_algo.test(testset)
    test_rmse = accuracy.rmse(predictions)

    save_pickle(predictions, '../../artifacts/predictions_svd_algo_rseed42.pkl')

    return predictions, test_rmse
