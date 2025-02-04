import sqlite3
import pickle
from tqdm import tqdm

def get_avg_rating_per_recommendation_query(query_movie: str, similar_movies: list) -> str:
    """
    Generates an SQL query to calculate the average rating per recommended movie
    watched by users who have rated the query movie at least 3.5.
    """
    placeholders = ','.join("?" for x in range(len(similar_movies)))

    query = f"""
    WITH users_who_watched AS (
        SELECT user_name
        FROM clean_dataset_b
        WHERE key_a = ?
        AND rating >= 3.5
    )
    SELECT r.key_a, AVG(r.rating) as average_rating
    FROM clean_dataset_b r
    JOIN users_who_watched uw ON r.user_name = uw.user_name
    WHERE r.key_a in ({placeholders})
    GROUP by r.key_a
    """

    return query

def get_overall_avg_recommendation_rating_query(num_similar_movies: int) -> str:
    """
    Generates an SQL query to calculate the overall average rating for similar movies.
    """
    placeholders = ','.join(['?'] * num_similar_movies)
    query = f"""
    SELECT AVG(r.rating) as average_rating
    FROM clean_dataset_b r
    JOIN clean_dataset_b uw ON r.user_name = uw.user_name
    WHERE uw.key_a = ? AND uw.rating >= 3.5 AND r.key_a IN ({placeholders})
    """

    return query

def get_unique_eval_movies(db_path: str) -> list:
    """
    Fetches a list of unique movies from the dataset.
    """
    query = """
    SELECT DISTINCT key_a
    FROM clean_dataset_b
    """

    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute(query)
        return [row[0] for row in c.fetchall()]

def get_eval_metric(conn, query_movie: str, similar_movies: list):
    """
    Fetches evaluation metric for a single movie.
    """
    if not similar_movies:
        return None

    query = get_overall_avg_recommendation_rating_query(len(similar_movies))
    params = [query_movie] + similar_movies

    with conn:
        c = conn.cursor()
        c.execute(query, params)
        return c.fetchone()
