import sqlite3

def get_avg_rating_per_recommendation_query(query_movie: str, similar_movies: list):
    number_of_similar_movies = ','.join("?" for x in range(len(similar_movies)))

    query = f"""
    WITH users_who_watched AS (
        SELECT user_name
        FROM clean_dataset_b
        WHERE key_a = "{query_movie}"
        AND rating >= 3.5
    )
    SELECT r.key_a, AVG(r.rating) as average_rating
    FROM clean_dataset_b r
    JOIN users_who_watched uw ON r.user_name = uw.user_name
    WHERE r.key_a in ({number_of_similar_movies})
    GROUP by r.key_a
    """

    return query

def get_overall_avg_recommendation_rating_query(query_movie: str, similar_movies: list):
    number_of_similar_movies = ','.join("?" for x in range(len(similar_movies)))

    query = f"""
    WITH users_who_watched AS (
        SELECT user_name
        FROM clean_dataset_b
        WHERE key_a = "{query_movie}"
        AND rating >= 3.5
    )
    SELECT AVG(r.rating) as average_rating
    FROM clean_dataset_b r
    JOIN users_who_watched uw ON r.user_name = uw.user_name
    WHERE r.key_a in ({number_of_similar_movies})
    """

    return query
