import os
import pickle
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

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
        df (pd.DataFrame): A DataFrame containing user ratings with columns ['user_name', 'movie_id', 'rating'].

    Returns:
        Dataset: A Surprise dataset object.
    """
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[['user_name', 'movie_id', 'rating']], reader)
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
    save_pickle(trainset, '../../artifacts/trainset_svd_model.pkl')
    save_pickle(testset, '../../artifacts/testset_svd_model.pkl')

    return trainset, testset

def train_svd_model(trainset, random_state=42):
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
    save_pickle(svd_model, '../../artifacts/fitted_svd_model.pkl')

    return svd_model

def evaluate_svd_model(testset, model):
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
    save_pickle(predictions, '../../artifacts/predictions_svd_model.pkl')

    return predictions, rmse_score

def train_and_test_svd_model():
    """
    Execute the full recommendation pipeline, including data loading, model training, and evaluation.

    Returns:
        tuple: A tuple containing the trained svd model and its predictions.
    """
    dataset_path = '../../artifacts/cleaned_dataset_b.pkl'
    data_path = '../../artifacts/dataset_b_surprise.pkl'
    trainset_path = '../../artifacts/trainset_svd_model.pkl'
    testset_path = '../../artifacts/testset_svd_model.pkl'
    svd_model_path = '../../artifacts/fitted_svd_model.pkl'
    predictions_path = '../../artifacts/predictions_svd_model.pkl'

    dataset_df = load_pickle(dataset_path)
    if dataset_df is None:
        print("Dataset couldn't be loaded! Check if data path is correct.")
        return None, None

    data = load_pickle(data_path) or load_data(dataset_df)
    trainset = load_pickle(trainset_path)
    testset = load_pickle(testset_path)

    if trainset is None or testset is None:
        trainset, testset = split_train_test(data)

    svd_model = load_pickle(svd_model_path) or train_svd_model(trainset)
    predictions = load_pickle(predictions_path)

    if predictions is None:
        predictions, rmse_score = evaluate_svd_model(testset, svd_model)
        print(f"Model RMSE: {rmse_score}")

    return svd_model, predictions

def get_movie_embeddings(svd_model, trainset):
    """
    Create a mapping from internal movie indices to their original IDs and extract movie embeddings.

    Parameters:
        svd_model (SVD): The trained SVD model.
        trainset (Trainset): The training dataset.

    Returns:
        pd.DataFrame: A DataFrame containing movie embeddings with original IDs as index.
    """
    mapping_dict = dict()
    for i in range(len(svd_model.qi)):
        mapping_dict[i] = trainset.to_raw_iid(i)

    embedding_movies = dict()
    for key in mapping_dict.keys():
        embedding_movies[mapping_dict[key]] = svd_model.qi[key]

    movie_embedding_mf = pd.DataFrame(embedding_movies).T

    return movie_embedding_mf

def train_knn_for_movies(svd_model, movie_embedding_mf):
    """
    Train a k-NN model on movie embeddings extracted from the trained SVD model.

    Returns:
        NearestNeighbors: The trained k-NN model.
    """
    knn = NearestNeighbors(metric='cosine', algorithm='auto')
    if svd_model:
        knn.fit(movie_embedding_mf)
        save_pickle(knn, '../../artifacts/fitted_knn_svd_model.pkl')

        return knn
    return None

def get_similar_movies_knn_mf(knn_model, movie_embedding_mf, movie_name, mapping_dict, n_neighbors=10):
    """
    Find similar movies based on cosine similarity using k-NN.

    Parameters:
        model (NearestNeighbors): The trained k-NN model.
        movie_embedding_mf (pd.DataFrame): Movie embeddings matrix.
        movie_name (str): The name of the target movie.
        mapping_dict (dict): Dictionary mapping internal movie indices to original IDs.
        n_neighbors (int): Number of similar movies to retrieve (default: 10).

    Returns:
        list: A list of similar movies.
    """
    embedding_vector = movie_embedding_mf.loc[movie_name].tolist()
    distances, indices = knn_model.kneighbors([embedding_vector], n_neighbors=n_neighbors+1)

    return [mapping_dict[x] for x in indices[0]]
