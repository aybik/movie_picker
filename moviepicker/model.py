from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def vectorize_descriptions(df, text_column):
    """
    Vectorize movie descriptions using TF-IDF.

    Args:
        df: The DataFrame containing movie descriptions.
        text_column: The column in the DataFrame that contains descriptions.

    Returns:
        tfidf_matrix: The TF-IDF matrix.
        vectorizer: The fitted TfidfVectorizer object (useful if needed later).
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df[text_column])
    return tfidf_matrix

def knn_fit(tfidf_matrix):
    # Fit KNN on the TF-IDF matrix
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(tfidf_matrix)
    return knn

def verify_input(df, input_name, name_column):
    if input_name not in df[name_column].values:
        raise ValueError(f"Movie '{input_name}' not found in the DataFrame.")

def get_similar_movies_knn(knn, tfidf_matrix, df, input_name, name_column, n_neighbors=5):
    """
    Find similar movies using KNN based on a TF-IDF matrix.

    Args:
        tfidf_matrix: The TF-IDF matrix.
        df: The DataFrame containing movie names and descriptions.
        input_name: The name of the movie to find similarities for.
        name_column: The column in the DataFrame that contains movie names.
        n_neighbors: The number of similar movies to retrieve (default is 5).

    Returns:
        A list of dictionaries with movie names and similarity scores.
    """
    # Get the index of the input movie
    # if input_name not in df[name_column].values:
    #     raise ValueError(f"Movie '{input_name}' not found in the DataFrame.")
    verify_input(df, input_name, name_column)

    idx = df[df[name_column] == input_name].index[0] #can be moved to verify_input

    # Find nearest neighbors
    distances, indices = knn.kneighbors(tfidf_matrix[idx], n_neighbors=n_neighbors + 1)

    # Exclude the input movie itself
    similar_movies = []
    for i in range(1, len(indices.flatten())):
        similar_movies.append({
            'input_name': df.iloc[indices.flatten()[i]][name_column],
            'similarity_score': 1 - distances.flatten()[i]  # Convert distance to similarity
        })
    return similar_movies
