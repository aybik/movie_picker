from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

####################################
# Data Preparation & Vectorization #
####################################

def vectorize_descriptions(df, text_column, tfidf_dim=2500):
    """
    Vectorizes movie descriptions using TF-IDF with a fixed vocabulary size.

    Parameters:
        df (pd.DataFrame): The dataset containing movie descriptions.
        text_column (str): The column in the DataFrame that contains descriptions.
        tfidf_dim (int): The maximum number of features for TF-IDF.

    Returns:
        tuple: (tfidf_array, vectorizer)
            tfidf_array (np.array): TF-IDF vector representation of descriptions.
            vectorizer (TfidfVectorizer): The trained TF-IDF vectorizer.
    """
    vectorizer = TfidfVectorizer(max_features=tfidf_dim)
    tfidf_matrix = vectorizer.fit_transform(df[text_column])
    return tfidf_matrix.toarray(), vectorizer  # Return both array & vectorizer


def prepare_model_inputs(df, tfidf_dim=2500):
    """
    Prepares input features for the autoencoder model.

    Parameters:
        df (pd.DataFrame): The dataset containing movie descriptions, language, genres.
        tfidf_dim (int): The maximum number of features for TF-IDF vectorization.

    Returns:
        tuple: (tfidf_array, vectorizer, num_languages, language_data_np, genres_data_np, num_genres)
    """
    # 1. TF-IDF Vectorization
    tfidf_array, vectorizer = vectorize_descriptions(df, 'description', tfidf_dim)

    # 2. Language Encoding (assumes that there is a column "language_encoded")
    num_languages = df['language_encoded'].nunique()
    language_data_np = df['language_encoded'].values.reshape(-1, 1).astype(np.int32)

    # 3. Genre Extraction
    genre_columns = ['action', 'adventure', 'animation', 'comedy', 'crime', 'documentary',
                     'drama', 'family', 'fantasy', 'history', 'horror', 'music', 'mystery',
                     'romance', 'science_fiction', 'thriller', 'tv_movie', 'war', 'western']
    genres_data_np = df[genre_columns].values.astype(np.int32)
    num_genres = len(genre_columns)

    return tfidf_array, vectorizer, num_languages, language_data_np, genres_data_np, num_genres

#############################
# Model Building Functions  #
#############################

def build_encoder(tfidf_dim, num_languages, num_genres):
    """
    Builds an encoder model that fuses:
      - A TF-IDF vector input (continuous, shape: [tfidf_dim])
      - A language input (integer, shape: [1])
      - A one-hot encoded genres input (shape: [num_genres])

    Parameters:
      tfidf_dim (int): Dimensionality of the TF-IDF vector (e.g., 2500).
      num_languages (int): Total number of language categories.
      num_genres (int): Number of genres (e.g., 19).

    Returns:
      encoder_model (tf.keras.Model): A model that outputs a fused latent embedding.
    """
    # TF-IDF Branch
    tfidf_input = tf.keras.layers.Input(shape=(tfidf_dim,), name="tfidf_input")
    tfidf_dense = tf.keras.layers.Dense(128, activation='relu', name="tfidf_dense")(tfidf_input)

    # Language Branch
    language_input = tf.keras.layers.Input(shape=(1,), name="language_input")
    language_embedding = tf.keras.layers.Embedding(
        input_dim=num_languages,
        output_dim=8,
        name="language_embedding"
    )(language_input)
    language_vector = tf.keras.layers.Flatten(name="language_flatten")(language_embedding)

    # Genres Branch (one-hot encoded)
    genre_input = tf.keras.layers.Input(shape=(num_genres,), name="genre_input")
    genre_dense = tf.keras.layers.Dense(32, activation='relu', name="genre_dense")(genre_input)

    # Merge all branches
    merged = tf.keras.layers.concatenate([tfidf_dense, language_vector, genre_dense], name="merged_features")
    x = tf.keras.layers.Dense(64, activation='relu', name="dense_1")(merged)
    final_embedding = tf.keras.layers.Dense(32, activation='relu', name="final_embedding")(x)

    encoder_model = tf.keras.models.Model(
        inputs=[tfidf_input, language_input, genre_input],
        outputs=final_embedding
    )
    return encoder_model


def build_autoencoder(num_languages, num_genres, tfidf_dim=2500, initial_lr=0.001):
    """
    Builds an autoencoder with a custom learning rate.

    Parameters:
      tfidf_dim (int): Dimensionality of the TF-IDF vector.
      num_languages (int): Total number of unique languages.
      num_genres (int): Number of genres.
      initial_lr (float): Initial learning rate for the optimizer.

    Returns:
      autoencoder_model (tf.keras.Model): The compiled autoencoder model.
      encoder_model (tf.keras.Model): The encoder model.
    """
    # Define inputs
    tfidf_input = tf.keras.layers.Input(shape=(tfidf_dim,), name="tfidf_input")
    language_input = tf.keras.layers.Input(shape=(1,), name="language_input")
    genre_input = tf.keras.layers.Input(shape=(num_genres,), name="genre_input")

    # Build encoder model
    encoder_model = build_encoder(tfidf_dim, num_languages, num_genres)
    latent = encoder_model([tfidf_input, language_input, genre_input])

    # Decoder for TF-IDF reconstruction
    decoder_tfidf = tf.keras.layers.Dense(64, activation='relu', name="decoder_tfidf_dense")(latent)
    tfidf_output = tf.keras.layers.Dense(tfidf_dim, activation='relu', name="tfidf_output")(decoder_tfidf)

    # Decoder for Language reconstruction
    decoder_language = tf.keras.layers.Dense(16, activation='relu', name="decoder_language_dense")(latent)
    language_output = tf.keras.layers.Dense(num_languages, activation='softmax', name="language_output")(decoder_language)

    # Decoder for Genres reconstruction
    decoder_genre = tf.keras.layers.Dense(16, activation='relu', name="decoder_genre_dense")(latent)
    genre_output = tf.keras.layers.Dense(num_genres, activation='sigmoid', name="genre_output")(decoder_genre)

    # Build the autoencoder model
    autoencoder_model = tf.keras.models.Model(
        inputs=[tfidf_input, language_input, genre_input],
        outputs=[tfidf_output, language_output, genre_output],
        name="autoencoder"
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    autoencoder_model.compile(
        optimizer=optimizer,
        loss={
            'tfidf_output': 'mse',
            'language_output': 'sparse_categorical_crossentropy',
            'genre_output': 'binary_crossentropy'
        }
    )

    return autoencoder_model, encoder_model

#############################
# Training & Embedding Extraction #
#############################

def train_autoencoder(autoencoder_model, tfidf_array, language_data_np, genres_data_np, batch_size, epochs):
    """
    Trains the autoencoder model using the given input data.

    Parameters:
        autoencoder_model (tf.keras.Model): The compiled autoencoder model.
        tfidf_array (np.array): TF-IDF input data.
        language_data_np (np.array): Encoded language data.
        genres_data_np (np.array): One-hot encoded genres data.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.

    Returns:
        history (tf.keras.callbacks.History): Training history object containing loss values.
    """
    # Define callbacks
    model_checkpoint = ModelCheckpoint("model_best.keras", monitor='val_loss', verbose=1, save_best_only=True)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-6, verbose=1)
    early_stopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

    history = autoencoder_model.fit(
        x=[tfidf_array, language_data_np, genres_data_np],
        y=[tfidf_array, language_data_np, genres_data_np],
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[model_checkpoint, lr_reducer, early_stopper]
    )
    return history


def load_trained_encoder(autoencoder_path, tfidf_dim, num_languages, num_genres):
    """
    Loads a trained autoencoder model, rebuilds the encoder, and transfers the encoder's weights.

    Parameters:
        autoencoder_path (str): Path to the saved autoencoder model file (.keras format).
        tfidf_dim (int): Dimensionality of the TF-IDF vector.
        num_languages (int): Number of unique languages.
        num_genres (int): Number of genres.

    Returns:
        encoder_trained (tf.keras.Model): The trained encoder model with weights loaded.
    """
    # Load the trained autoencoder model
    trained_autoencoder = tf.keras.models.load_model(autoencoder_path)
    print("✅ Autoencoder model loaded successfully!")

    # Rebuild the encoder model structure
    encoder_trained = build_encoder(tfidf_dim, num_languages, num_genres)
    print("✅ Encoder model structure rebuilt!")

    # Transfer weights from the trained autoencoder to the encoder
    encoder_trained.set_weights(trained_autoencoder.get_weights()[:len(encoder_trained.weights)])
    print("✅ Encoder weights loaded successfully!")

    return encoder_trained


def extract_latent_embeddings(encoder_trained, tfidf_array, language_data_np, genres_data_np):
    """
    Extracts latent embeddings from the encoder model.

    Parameters:
        encoder_traşned (tf.keras.Model): The trained encoder model.
        tfidf_array (np.array): TF-IDF input data.
        language_data_np (np.array): Encoded language data.
        genres_data_np (np.array): One-hot encoded genres data.

    Returns:
        latent_embeddings (np.array): The extracted latent representations.
    """
    latent_embeddings = encoder_trained.predict([tfidf_array, language_data_np, genres_data_np])
    return latent_embeddings

#############################
# KNN & Recommendation Functions #
#############################

def knn_fit(latent_embeddings, n_neighbors=10, metric='cosine'):
    """
    Fits a KNN model for similarity search using the latent embeddings.

    Parameters:
        latent_embeddings (np.array): The extracted latent embeddings from the encoder.
        n_neighbors (int): Number of nearest neighbors to find.
        metric (str): Distance metric for KNN.

    Returns:
        knn_model (NearestNeighbors): The trained KNN model.
    """
    # +1 neighbor to allow exclusion of the queried movie later
    knn_model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    knn_model.fit(latent_embeddings)
    return knn_model

def get_movie_recommendations(user_input, df, knn_model, latent_embeddings, n_recommendations=5, alpha=0.05):
    """
    Finds similar movies based on the KNN model and latent embeddings, incorporating rating effects.

    Parameters:
        user_input (str): The name of the movie to find recommendations for.
        df (pd.DataFrame): DataFrame containing movie names and combined ratings.
        knn_model (NearestNeighbors): The trained KNN model.
        latent_embeddings (np.array): The extracted latent embeddings.
        n_recommendations (int): Number of movie recommendations to return.
        alpha (float): Scaling factor for rating impact on similarity. Default is 0.05.

    Returns:
        list: A list of tuples containing recommended movies and their adjusted similarity scores.
    """
    # Ensure DataFrame index is reset to align with latent embeddings
    df = df.reset_index(drop=True)

    # Case-insensitive matching for user input
    matched_rows = df[df["key"].str.lower() == user_input.lower()]

    if matched_rows.empty:
        return {"error": f"Movie '{user_input}' not found in dataset."}

    sample_index = matched_rows.index[0]

    try:
        distances, indices = knn_model.kneighbors(latent_embeddings[sample_index].reshape(1, -1))
    except IndexError:
        return {"error": "Invalid movie index. Check if latent embeddings align with DataFrame."}

    indices = indices.flatten()
    distances = distances.flatten()

    # Compute similarity scores (higher similarity = more relevant)
    similarity_scores = [1 / (1 + dist) for dist in distances]

    # Extract the top 20 most similar movies (excluding the input movie)
    similar_movies = [df.iloc[idx]["key"] for idx in indices if idx != sample_index][:20]
    similar_similarities = [sim for idx, sim in zip(indices, similarity_scores) if idx != sample_index][:20]

    # Compute rating effects and adjust similarity
    adjusted_similarities = []
    for movie, sim in zip(similar_movies, similar_similarities):
        rating = df.loc[df["key"] == movie, "combined_rating"]
        if not rating.empty:
            adjusted_sim = sim + (float(rating.values[0]) * alpha)  # Increase similarity with rating
        else:
            adjusted_sim = sim  # No rating adjustment if rating is missing
        adjusted_similarities.append(adjusted_sim)

    # Sort by adjusted similarity scores (higher is better)
    sorted_recommendations = sorted(zip(similar_movies, adjusted_similarities), key=lambda x: x[1], reverse=True)[:n_recommendations]
# 🔹 Return only the "key" column values for recommended movies
    recommended_keys = [df.loc[df["key"] == movie, "key"].values[0] for movie, _ in sorted_recommendations]

    return recommended_keys if recommended_keys else {"message": "No recommendations found."}

    #return sorted_recommendations if sorted_recommendations else {"message": "No recommendations found."}


def recommend_movies_by_details(user_description, user_language, user_genres, df, encoder_model, knn_model, vectorizer, tfidf_dim=2500, n_recommendations=5, alpha=0.05):
    """
    Finds similar movies based on a user-provided description, language, and genres, incorporating rating effects.

    Parameters:
        user_description (str): The user's movie description input.
        user_language (str): The user's selected language.
        user_genres (list): A list of genres selected by the user.
        df (pd.DataFrame): The dataset containing movies.
        encoder_model (tf.keras.Model): The trained encoder model.
        knn_model (NearestNeighbors): The trained KNN model.
        vectorizer (TfidfVectorizer): The TF-IDF vectorizer used during training.
        tfidf_dim (int): The number of TF-IDF features.
        n_recommendations (int): Number of movie recommendations to return.
        alpha (float): Scaling factor for rating impact on similarity.

    Returns:
        list: A list of recommended movie names.
    """

    # Define genre columns (should match those used during training)
    genre_columns = [col for col in df.columns if col in [
        'action', 'adventure', 'animation', 'comedy', 'crime', 'documentary',
        'drama', 'family', 'fantasy', 'history', 'horror', 'music', 'mystery',
        'romance', 'science_fiction', 'thriller', 'tv_movie', 'war', 'western'
    ]]

    # 1️⃣ Convert the user description to a TF-IDF vector
    user_tfidf = vectorizer.transform([user_description]).toarray()

    # 2️⃣ Encode the user language (defaults to English if not found)
    if user_language in df["language"].values:
        user_language_encoded = np.array([[df.loc[df["language"] == user_language, "language_encoded"].values[0]]], dtype=np.int32)
    else:
        # Default to "English"
        default_language = "English"
        if default_language in df["language"].values:
            user_language_encoded = np.array([[df.loc[df["language"] == default_language, "language_encoded"].values[0]]], dtype=np.int32)
        else:
            user_language_encoded = np.array([[0]], dtype=np.int32)  # Fallback to 0 if "English" is not found

    # 3️⃣ Extract the one-hot encoded genre vector directly
    user_genre_vector = np.zeros((1, len(genre_columns)), dtype=np.int32)
    for genre in user_genres:
        if genre in genre_columns:
            user_genre_vector[0, genre_columns.index(genre)] = 1

    # 4️⃣ Generate latent embedding using the trained encoder model
    user_embedding = encoder_model.predict([user_tfidf, user_language_encoded, user_genre_vector])

    # 5️⃣ Use KNN to find similar movies
    distances, indices = knn_model.kneighbors(user_embedding)
    indices = indices.flatten()
    distances = distances.flatten()

    # 6️⃣ Compute similarity scores (higher similarity = better match)
    similarity_scores = [1 / (1 + dist) for dist in distances]

    # 7️⃣ Extract movie names and their similarity scores
    df = df.reset_index(drop=True)
    similar_movies = [df.iloc[idx]["name"] for idx in indices][:n_recommendations]
    similar_similarities = [sim for sim in similarity_scores][:n_recommendations]

    # 8️⃣ Adjust similarity using ratings
    adjusted_similarities = []
    for movie, sim in zip(similar_movies, similar_similarities):
        rating = df.loc[df["name"] == movie, "combined_rating"]
        if not rating.empty:
            adjusted_sim = sim + (float(rating.values[0]) * alpha)  # Increase similarity with rating
        else:
            adjusted_sim = sim  # No rating adjustment if rating is missing
        adjusted_similarities.append(adjusted_sim)

    # 9️⃣ Sort by adjusted similarity scores (higher is better)
    sorted_recommendations = sorted(zip(similar_movies, adjusted_similarities), key=lambda x: x[1], reverse=True)[:n_recommendations]

    # 🔹 Return only movie names (without distances)
    recommended_movies = [movie for movie, _ in sorted_recommendations]

    return recommended_movies if recommended_movies else {"message": "No recommendations found."}


#############################
# Example Usage (Commented) #
#############################

# # Prepare your data
# tfidf_array, vectorizer, num_languages, language_data_np, genres_data_np, num_genres = prepare_model_inputs(df)
#
# # Build autoencoder and encoder
# autoencoder_model, encoder_model = build_autoencoder(num_languages, num_genres, tfidf_dim=2500, initial_lr=0.001)
#
# # Train autoencoder (the encoder_model will be updated during training)
# history = train_autoencoder(autoencoder_model, tfidf_array, language_data_np, genres_data_np, batch_size=16, epochs=50)
#
# # Extract latent embeddings using the trained encoder
# latent_embeddings = extract_latent_embeddings(encoder_model, tfidf_array, language_data_np, genres_data_np)
#
# # Fit a KNN model on the latent embeddings
# knn_model = knn_fit(latent_embeddings, n_neighbors=10, metric='cosine')
#
# # Get movie recommendations based on a movie name
# recommendations = get_movie_recommendations("Parasite", df, knn_model, latent_embeddings, n_recommendations=5)
# print("Recommendations based on movie name:", recommendations)
#
# # Get recommendations based on user details
# recommendations_details = recommend_movies_by_details("people falling in love", "English", ["drama"], df, encoder_model, knn_model, vectorizer)
# print("Recommendations based on user details:", recommendations_details)

import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')

def get_extended_synonyms(word):
    """
    Retrieves synonyms from WordNet for a given word, keeping only single words.

    Parameters:
        word (str): The word to find synonyms for.

    Returns:
        set: A set of single-word synonyms.
    """
    synonyms = set()

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            clean_word = lemma.name().replace("_", " ")  # Replace underscores with spaces
            if " " not in clean_word:  # Keep only single words
                synonyms.add(clean_word)

    return synonyms


def expand_description(user_description):
    """
    Expands the user-provided description by adding synonyms for each word.

    Parameters:
        user_description (str): The input movie description.

    Returns:
        str: The enhanced description with additional synonyms.
    """
    final_words = set(user_description.split())  # Start with the original words
    for w in user_description.split():
        extended_synonyms = get_extended_synonyms(w)
        final_words.update(extended_synonyms)  # Add synonyms

    return " ".join(final_words)  # Return as a single string




def recommend_movies_by_details_final(user_description, df, encoder_model, knn_model, vectorizer,
                                user_language="English", user_genres=None,
                                tfidf_dim=2500, n_recommendations=5, alpha=0.05):
    """
    Finds similar movies based on a user-provided description, language, and genres, incorporating rating effects.

    Parameters:
        user_description (str): The user's movie description input.
        df (pd.DataFrame): The dataset containing movies.
        encoder_model (tf.keras.Model): The trained encoder model.
        knn_model (NearestNeighbors): The trained KNN model.
        vectorizer (TfidfVectorizer): The TF-IDF vectorizer used during training.
        user_language (str): The user's selected language. Defaults to "English".
        user_genres (list): A list of genres selected by the user. Defaults to ["thriller"] if not provided.
        tfidf_dim (int): The number of TF-IDF features.
        n_recommendations (int): Number of movie recommendations to return.
        alpha (float): Scaling factor for rating impact on similarity.

    Returns:
        list: A list of recommended movie keys.
    """
    # Set default genres if none are provided
    if user_genres is None:
        user_genres = ["thriller"]

    # Define genre columns (should match those used during training)
    genre_columns = [col for col in df.columns if col in [
        'action', 'adventure', 'animation', 'comedy', 'crime', 'documentary',
        'drama', 'family', 'fantasy', 'history', 'horror', 'music', 'mystery',
        'romance', 'science_fiction', 'thriller', 'tv_movie', 'war', 'western'
    ]]

    # 1️⃣ Convert the user description to a TF-IDF vector
    enhanced_description = expand_description(user_description)
    user_tfidf = vectorizer.transform([enhanced_description]).toarray()

    # 2️⃣ Encode the user language.
    # If the provided language is not in the dataset, default to "English"
    if user_language in df["language"].values:
        lang_val = df.loc[df["language"] == user_language, "language_encoded"].values[0]
    else:
        default_language = "English"
        if default_language in df["language"].values:
            lang_val = df.loc[df["language"] == default_language, "language_encoded"].values[0]
        else:
            lang_val = 0  # Fallback value if English is not found
    user_language_encoded = np.array([[lang_val]], dtype=np.int32)

    # 3️⃣ Extract the one-hot encoded genre vector directly
    user_genre_vector = np.zeros((1, len(genre_columns)), dtype=np.int32)
    for genre in user_genres:
        if genre in genre_columns:
            user_genre_vector[0, genre_columns.index(genre)] = 1

    # 4️⃣ Generate latent embedding using the trained encoder model
    user_embedding = encoder_model.predict([user_tfidf, user_language_encoded, user_genre_vector])

    # 5️⃣ Use KNN to find similar movies
    distances, indices = knn_model.kneighbors(user_embedding)
    indices = indices.flatten()
    distances = distances.flatten()

    # 6️⃣ Compute similarity scores (higher similarity = better match)
    similarity_scores = [1 / (1 + dist) for dist in distances]

    # 7️⃣ Extract movie names and their similarity scores
    df = df.reset_index(drop=True)
    similar_movies = [df.iloc[idx]["name"] for idx in indices][:n_recommendations]
    similar_similarities = [sim for sim in similarity_scores][:n_recommendations]

    # 8️⃣ Adjust similarity using ratings
    adjusted_similarities = []
    for movie, sim in zip(similar_movies, similar_similarities):
        rating = df.loc[df["name"] == movie, "combined_rating"]
        if not rating.empty:
            adjusted_sim = sim + (float(rating.values[0]) * alpha)  # Increase similarity with rating
        else:
            adjusted_sim = sim  # No rating adjustment if rating is missing
        adjusted_similarities.append(adjusted_sim)

    # 9️⃣ Sort by adjusted similarity scores (higher is better)
    sorted_recommendations = sorted(zip(similar_movies, adjusted_similarities),
                                    key=lambda x: x[1], reverse=True)[:n_recommendations]

    # 🔹 Return only the "key" column values for recommended movies
    recommended_keys = [df.loc[df["name"] == movie, "key"].values[0]
                        for movie, _ in sorted_recommendations]

    return recommended_keys if recommended_keys else {"message": "No recommendations found."}

def recommend_movies_by_details_new(user_description, df, encoder_model, knn_model, vectorizer,
                                user_language="English", user_genres="action",
                                tfidf_dim=2500, n_recommendations=5, alpha=0.05):
    """
    Finds similar movies based on a user-provided description, language, and genres, incorporating rating effects.

    Parameters:
        user_description (str): The user's movie description input.
        df (pd.DataFrame): The dataset containing movies.
        encoder_model (tf.keras.Model): The trained encoder model.
        knn_model (NearestNeighbors): The trained KNN model.
        vectorizer (TfidfVectorizer): The TF-IDF vectorizer used during training.
        user_language (str): The user's selected language. Defaults to "English".
        user_genres (list): A list of genres selected by the user. Defaults to ["thriller"] if not provided.
        tfidf_dim (int): The number of TF-IDF features.
        n_recommendations (int): Number of movie recommendations to return.
        alpha (float): Scaling factor for rating impact on similarity.

    Returns:
        list: A list of recommended movie keys.
    """

    # Define genre columns (should match those used during training)
    genre_columns = [col for col in df.columns if col in [
        'action', 'adventure', 'animation', 'comedy', 'crime', 'documentary',
        'drama', 'family', 'fantasy', 'history', 'horror', 'music', 'mystery',
        'romance', 'science_fiction', 'thriller', 'tv_movie', 'war', 'western'
    ]]

    # 1️⃣ Convert the user description to a TF-IDF vector
    enhanced_description = expand_description(user_description)
    user_tfidf = vectorizer.transform([enhanced_description]).toarray()

    # 2️⃣ Encode the user language.
    # If the provided language is not in the dataset, default to "English"
    if user_language in df["language"].values:
        lang_val = df.loc[df["language"] == user_language, "language_encoded"].values[0]
    else:
        default_language = "English"
        if default_language in df["language"].values:
            lang_val = df.loc[df["language"] == default_language, "language_encoded"].values[0]
        else:
            lang_val = 0  # Fallback value if English is not found
    user_language_encoded = np.array([[lang_val]], dtype=np.int32)

    # 3️⃣ Extract the one-hot encoded genre vector directly
    user_genre_vector = np.zeros((1, len(genre_columns)), dtype=np.int32)
    if user_genres in genre_columns:
        user_genre_vector[0, genre_columns.index(user_genres)] = 1

    # 4️⃣ Generate latent embedding using the trained encoder model
    user_embedding = encoder_model.predict([user_tfidf, user_language_encoded, user_genre_vector])

    # 5️⃣ Use KNN to find similar movies
    distances, indices = knn_model.kneighbors(user_embedding)
    indices = indices.flatten()
    distances = distances.flatten()

    # 6️⃣ Compute similarity scores (higher similarity = better match)
    similarity_scores = [1 / (1 + dist) for dist in distances]

    # 7️⃣ Extract movie names and their similarity scores
    df = df.reset_index(drop=True)
    similar_movies = [df.iloc[idx]["name"] for idx in indices][:n_recommendations]
    similar_similarities = [sim for sim in similarity_scores][:n_recommendations]

    # 8️⃣ Adjust similarity using ratings
    adjusted_similarities = []
    for movie, sim in zip(similar_movies, similar_similarities):
        rating = df.loc[df["name"] == movie, "combined_rating"]
        if not rating.empty:
            adjusted_sim = sim + (float(rating.values[0]) * alpha)  # Increase similarity with rating
        else:
            adjusted_sim = sim  # No rating adjustment if rating is missing
        adjusted_similarities.append(adjusted_sim)

    # 9️⃣ Sort by adjusted similarity scores (higher is better)
    sorted_recommendations = sorted(zip(similar_movies, adjusted_similarities),
                                    key=lambda x: x[1], reverse=True)[:n_recommendations]

    # 🔹 Return only the "key" column values for recommended movies
    recommended_keys = [df.loc[df["name"] == movie, "key"].values[0]
                        for movie, _ in sorted_recommendations]

    return recommended_keys if recommended_keys else {"message": "No recommendations found."}
