import os
import pickle
import pandas as pd
from tqdm import tqdm
import sqlite3
import model as m

# Pickle functions
def save_pickle(my_obj, filepath):
    """
    Save an object to a pickle file.
    """
    if os.path.isfile(filepath):
        print(f"File {filepath} already exists. Doing nothing")
        return None
    with open(filepath, 'wb') as file:
        pickle.dump(my_obj, file)

def load_pickle(filepath):
    """
    Load an object from a pickle file.
    """
    if not os.path.isfile(filepath):
        print(f"File {filepath} does not exist.")
        return None
    with open(filepath, 'rb') as file:
        return pickle.load(file)

# Movie Evaluation Functions
def get_top_movies(filtered_validation_data, cutoff=25000):
    """
    Returns the top 25,000 movies with the most ratings.
    """
    top_movies = (filtered_validation_data['key']
                  .value_counts()
                  .head(cutoff)
                  .index.tolist())
    return top_movies

def get_avg_rating_per_recommendation_query(query_movie: str, similar_movies_list: list) -> str:
    """
    Generates an SQL query to calculate the average rating per recommended movie
    watched by users who have rated the query movie at least 3.5.
    """
    placeholders = ','.join("?" for x in range(len(similar_movies_list)))
    query = f"""
    WITH users_who_watched AS (
        SELECT user_name
        FROM filtered_validation_data
        WHERE key = ?
        AND rating >= 3.5
    )
    SELECT fvd.key, AVG(fvd.rating) as average_rating
    FROM filtered_validation_data fvd
    JOIN users_who_watched uw ON fvd.user_name = uw.user_name
    WHERE fvd.key in ({placeholders})
    GROUP by fvd.key
    """
    return query

def get_overall_avg_recommendation_rating_query(num_similar_movies: int) -> str:
    """
    Generates an SQL query to calculate the overall average rating for similar movies.
    """
    placeholders = ','.join(['?'] * num_similar_movies)
    query = f"""
    SELECT AVG(fvd.rating) as average_rating
    FROM filtered_validation_data fvd
    JOIN filtered_validation_data uw ON fvd.user_name = uw.user_name
    WHERE uw.key = ? AND uw.rating >= 3.5 AND fvd.key IN ({placeholders})
    """
    return query

def batch_process_eval_metrics(db_path: str, similarity_dict: dict, batch_size: int = 3000) -> dict:
    """
    Processes movie evaluations in batches to optimize memory usage.
    """
    results = {}
    with sqlite3.connect(db_path) as conn:
        movies = list(similarity_dict.keys())

        for i in tqdm(range(0, len(movies), batch_size)):
            batch = movies[i:i + batch_size]
            for movie in tqdm(batch):
                results[movie] = get_eval_metric(conn, movie, similarity_dict[movie])
    return results

def get_eval_metric(conn, query_movie, similar_movies, use_avg_per_recommendation=False):
    """
    Fetches evaluation metric for a single movie.
    """
    if not similar_movies:
        return None

    if use_avg_per_recommendation:
        query = get_avg_rating_per_recommendation_query(query_movie, similar_movies)
    else:
        query = get_overall_avg_recommendation_rating_query(len(similar_movies))

    params = [query_movie] + similar_movies
    with conn:
        return conn.execute(query, params).fetchone()

def create_database_indexes(db_path: str):
    """
    Optimizes the database by creating necessary indexes.
    """
    with sqlite3.connect(db_path) as conn:
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA cache_size=1000000;")
            conn.execute("PRAGMA temp_store=MEMORY;")

            c = conn.cursor()
            c.execute("CREATE INDEX IF NOT EXISTS idx_key ON filtered_validation_data (key);")
            c.execute("CREATE INDEX IF NOT EXISTS idx_user_name ON filtered_validation_data (user_name);")
            conn.commit()
            print("Indexes created successfully.")
        except sqlite3.OperationalError as e:
            print("SQLite error:", e)

# Helper function to compute evaluation metrics
def compute_eval_metrics(filtered_validation_data, similar_movies):
    """
    Computes evaluation metrics using DataFrame operations.
    """
    filtered_data = filtered_validation_data[filtered_validation_data['rating'] >= 3.5]
    merged_data = filtered_data.merge(filtered_data, on='user_name', suffixes=('_query', '_rec'))
    merged_data = merged_data[merged_data['key_rec'].isin(similar_movies.keys())]
    result = merged_data.groupby('key_rec')['rating_rec'].mean().to_dict()

    return result


# If the code is run directly
if __name__ == "__main__":
    get_evaluation_score()
