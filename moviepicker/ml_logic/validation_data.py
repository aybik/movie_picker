import os
import pandas as pd

def get_parent_directory() -> str:
    """
    Get the parent directory of the current working directory.

    Returns:
        str: Parent directory path.
    """
    current_dir = os.getcwd()

    return os.path.dirname(current_dir)

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
    films['key_a'] = films['film_name'] + films['year'].apply(
        lambda x: f" ({int(x)})" if not pd.isna(x) else ''
    )
    films = films.drop(columns='year')

    return films.dropna(subset=['film_id', 'film_name', 'key_a'])

def merge_datasets(ratings: pd.DataFrame, films: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the ratings dataset with the cleaned films dataset.

    Args:
        ratings (pd.DataFrame): The ratings dataset.
        films (pd.DataFrame): The cleaned films dataset.

    Returns:
        pd.DataFrame: The merged dataset.
    """
    ratings = ratings.dropna()
    merged = ratings.merge(films, how='left', on='film_id')
    merged = merged.dropna(subset=['film_name']).drop(columns='film_id')

    return merged
