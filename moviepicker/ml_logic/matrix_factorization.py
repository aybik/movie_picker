import os
import pickle
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

def save_pickle(obj, filepath):
    """
    Save an object to a pickle file if the file does not already exist.

    Parameters:
        obj (any): The object to be saved.
        filepath (str): The file path where the object should be saved.
    """
    if os.path.isfile(filepath):
        print(f"File {filepath} already exists. Doing nothing.")
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)


def load_pickle(filepath):
    """
    Load an object from a pickle file if it exists.

    Parameters:
        filepath (str): The file path from which the object should be loaded.

    Returns:
        any: The loaded object, or None if the file does not exist.
    """
    if not os.path.isfile(filepath):
        print(f"File {filepath} does not exist.")
        return None

    with open(filepath, 'rb') as f:
        return pickle.load(f)

def load_data(df):
    """
    Load a dataset from a pandas DataFrame and save it as a pickle file.

    Parameters:
        df (pd.DataFrame): A DataFrame containing user ratings with columns ['user_name', 'film_id', 'rating'].

    Returns:
        Dataset: A Surprise dataset object.
    """
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[['user_name', 'film_id', 'rating']], reader)
    save_pickle(data, '../../artifacts/dataset_b_surprise.pkl')

    return data

def split_train_test(data, test_size=0.25, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
        data (Dataset): A Surprise dataset object.
        test_size (float): The proportion of data to include in the test set (default: 0.25).
        random_state (int): Random seed for reproducibility (default: 42).

    Returns:
        tuple: A tuple containing the training set and test set.
    """
    trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)
    save_pickle(trainset, '../../artifacts/trainset.pkl')
    save_pickle(testset, '../../artifacts/testset.pkl')

    return trainset, testset

def train_model(trainset, random_state=42):
    """
    Train an SVD model on the training set.

    Parameters:
        trainset (Trainset): The training dataset.
        random_state (int): Random seed for model reproducibility (default: 42).

    Returns:
        SVD: The trained SVD model.
    """
    svd_model = SVD(random_state=random_state)
    svd_model.fit(trainset)
    save_pickle(svd_model, '../../artifacts/fitted_svd.pkl')

    return svd_model

def evaluate_model(testset, model):
    """
    Evaluate the trained model on the test set using RMSE.

    Parameters:
        testset (list): The test dataset.
        model (SVD): The trained SVD model.

    Returns:
        tuple: A tuple containing predictions and the RMSE score.
    """
    predictions = model.test(testset)
    rmse_score = accuracy.rmse(predictions)
    save_pickle(predictions, '../../artifacts/predictions.pkl')

    return predictions, rmse_score

def main_pipeline():
    """
    Wrapper function to execute the full pipeline.

    Parameters:
        df (pd.DataFrame): Input dataset containing user ratings.
    """
    dataset_path = '../../artifacts/cleaned_dataset_b.pkl'
    data_path = '../../artifacts/dataset_surprise.pkl'
    trainset_path = '../../artifacts/trainset.pkl'
    testset_path = '../../artifacts/testset.pkl'
    model_path = '../../artifacts/fitted_svd.pkl'
    predictions_path = '../../artifacts/predictions.pkl'

    dataset_df = load_pickle(dataset_path)

    if predictions is None:
        print(f"Dataframe couldn't be loaded! Check if data path is correct.")

    data = load_pickle(data_path) or load_data(dataset_df)
    trainset = load_pickle(trainset_path)
    testset = load_pickle(testset_path)

    if trainset is None or testset is None:
        trainset, testset = split_train_test(data)

    model = load_pickle(model_path) or train_model(trainset)
    predictions = load_pickle(predictions_path)

    if predictions is None:
        predictions, rmse_score = evaluate_model(testset, model)
        print(f"Model RMSE: {rmse_score}")

    return model, predictions
