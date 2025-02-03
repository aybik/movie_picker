import os
import pandas as pd
import pickle

def save_pickle(my_obj, filepath):
    """
    Save an object to a pickle file.

    Args:
        my_obj: The object to be saved.
        filepath (str): The path where the pickle file will be saved.
    """
    if os.path.isfile(filepath):
        print(f"File {filepath} already exists. Doing nothing")
        return None
    else:
        with open(filepath, 'wb') as file:
            pickle.dump(my_obj, file)

def load_pickle(filepath):
    """
    Load an object from a pickle file.

    Args:
        filepath (str): The path to the pickle file.

    Returns:
        The loaded object.
    """
    if not os.path.isfile(filepath):
        print(f"File {filepath} does not exist.")
        return None
    with open(filepath, 'rb') as file:
        return pickle.load(file)

def load_dataset(file_path: str, skip_bad_lines: bool = False) -> pd.DataFrame:
    """
    Load a CSV dataset from the given file path.

    Args:
        file_path (str): The path to the CSV file.
        skip_bad_lines (bool): Whether to skip faulty lines in the file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """

    return pd.read_csv(file_path, on_bad_lines='skip' if skip_bad_lines else 'error')

def clean_films_data(films: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the films dataset.

    Args:
        films (pd.DataFrame): The raw films dataset.

    Returns:
        pd.DataFrame: The cleaned films dataset.
    """
    films = films.drop(columns='poster_url', errors='ignore')
    films['year'] = pd.to_numeric(films['year'], errors='coerce').astype('Int64')
    films['key'] = films['film_name'] + films['year'].apply(
        lambda x: '' if pd.isna(x) else f" ({int(x)})"
    )
    films = films.drop(columns='year')

    return films.dropna(subset=['film_id', 'film_name', 'key'])

def merge_datasets(ratings: pd.DataFrame, films: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the ratings dataset with the cleaned films dataset.

    Args:
        ratings (pd.DataFrame): The ratings dataset.
        films (pd.DataFrame): The cleaned films dataset.

    Returns:
        pd.DataFrame: The merged validation dataset.
    """
    ratings = ratings.dropna()
    validation_dataset = ratings.merge(films, how='left', on='film_id')
    validation_dataset = validation_dataset.dropna(subset=['film_name']).drop(columns='film_id')

    return validation_dataset

def filter_common_keys(validation_dataset: pd.DataFrame, data_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Filter merged dataset by retaining only common keys between validation_dataset 'key' and data_dataset 'key'.

    Args:
        merged_data (pd.DataFrame): The merged dataset.
        clean_data (pd.DataFrame): The cleaned reference dataset.

    Returns:
        pd.DataFrame: The filtered dataset.
    """
    unique_key_values_data_dataset = set(data_dataset['key'].unique())
    unique_key_values_validation_dataset = set(validation_dataset['key'].unique())
    common_values = unique_key_values_data_dataset.intersection(unique_key_values_validation_dataset)
    is_common = {x: x in common_values for x in unique_key_values_validation_dataset}

    return validation_dataset[validation_dataset['key'].map(is_common)]

def get_filtered_data():
    """
    Main function to load, clean, merge, and filter datasets.
    """

    # Load datasets
    films = load_dataset('../../raw_data/set_b/films.csv', skip_bad_lines=True)
    ratings = load_dataset('../../raw_data/set_b/ratings.csv')
    data_dataset = load_pickle('../../artifacts/data_encode.pkl')

    if films is None or ratings is None or data_dataset is None:
        print("One or more datasets could not be loaded. Exiting.")
        return None

    # Process datasets
    films_cleaned = clean_films_data(films)
    validation_dataset = merge_datasets(ratings, films_cleaned)
    filtered_data = filter_common_keys(validation_dataset, data_dataset)

    return filtered_data

if __name__ == "__main__":
    filtered_data = main()

    save_pickle(filtered_data, '../../artifacts/filtered_validation_data.pkl')
