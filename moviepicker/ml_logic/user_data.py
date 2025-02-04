import os
import pickle
import pandas as pd

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

def load_and_clean_data(ratings_path, films_path):
    """
    Load and clean ratings and films datasets, then merge them.

    Parameters:
        ratings_path (str): File path to the ratings CSV file.
        films_path (str): File path to the films CSV file.

    Returns:
        pd.DataFrame: A cleaned and merged dataset.
    """
    output_pickle_path = '../../artifacts/cleaned_dataset_b.pkl'

    try:
        ratings = pd.read_csv(ratings_path).dropna()
    except Exception as e:
        print(f"Error loading ratings data: {e}")
        return None

    try:
        films = pd.read_csv(films_path, on_bad_lines='skip')
    except Exception as e:
        print(f"Error loading films data: {e}")
        return None

    # Clean films dataset
    films.drop(columns=['poster_url'], errors='ignore', inplace=True)
    films['year'] = pd.to_numeric(films['year'], errors='coerce').astype('Int64')
    films['key'] = films['film_name'] + films['year'].apply(
        lambda x: '' if pd.isna(x) else f" ({int(x)})"
    )
    films.drop(columns=['year'], inplace=True)
    films.dropna(subset=['film_id', 'film_name', 'key'], inplace=True)

    # Merge datasets
    cleaned_data = ratings.merge(films, how='left', on='film_id')
    cleaned_data.dropna(subset=['film_name'], inplace=True)
    cleaned_data.drop(columns=['film_id'], inplace=True)
    cleaned_data.rename(columns={'key': 'movie_id'}, inplace=True)

    save_pickle(cleaned_data, output_pickle_path)

    return cleaned_data
