from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

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


def get_movie_recommendations(user_input, df, knn_model, latent_embeddings, n_recommendations=5):
    """
    Finds similar movies based on the KNN model and latent embeddings.

    Parameters:
        user_input (str): The name of the movie to find recommendations for.
        df (pd.DataFrame): DataFrame containing movie names.
        knn_model (NearestNeighbors): The trained KNN model.
        latent_embeddings (np.array): The extracted latent embeddings.
        n_recommendations (int): Number of movie recommendations to return.

    Returns:
        list: A list of tuples containing recommended movies and their distances.
    """
    # Ensure DataFrame index is reset to align with latent embeddings
    df = df.reset_index(drop=True)

    # Match movie name case-insensitively
    matched_rows = df[df["name"].str.lower() == user_input.lower()]

    if matched_rows.empty:
        print("Movie not found.")
        return []

    sample_index = matched_rows.index[0]
    print(f"Found movie '{user_input}'.")

    distances, indices = knn_model.kneighbors(latent_embeddings[sample_index].reshape(1, -1))
    indices = indices.flatten()
    distances = distances.flatten()

    # Extract the top 20 most similar movies (excluding the input movie)
    similar_movies = [df.iloc[idx]["name"] for idx in indices if idx != sample_index][:20]
    similar_distances = [dist for idx, dist in zip(indices, distances) if idx != sample_index][:20]

    # Compute rating effects inside this function
    rating_effects = []
    for movie in similar_movies:
        rating = df.loc[df["name"] == movie, "combined_rating"]
        if not rating.empty:
            rating_effects.append(float(rating.values[0]) * 0.05)  # Scale the rating
        else:
            rating_effects.append(0)  # Default to 0 if no rating found

    # Adjust similarity scores by adding rating effects
    adjusted_scores = [dist + rating for dist, rating in zip(similar_distances, rating_effects)]

    # Sort by adjusted scores (lower is better)
    sorted_recommendations = sorted(zip(similar_movies, adjusted_scores), key=lambda x: x[1])[:n_recommendations]

    return sorted_recommendations if sorted_recommendations else {"message": "No recommendations found."}

def recommend_movies_by_details(user_description, user_language, user_genres, df, encoder_model, knn_model, vectorizer, tfidf_dim=2500, n_recommendations=5):
    """
    Finds similar movies based on a user-provided description, language, and genres.

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

    Returns:
        list: A list of recommended movie names and distances.
    """
    # Define genre columns (should match those used during training)
    genre_columns = [col for col in df.columns if col in [
        'action', 'adventure', 'animation', 'comedy', 'crime', 'documentary',
        'drama', 'family', 'fantasy', 'history', 'horror', 'music', 'mystery',
        'romance', 'science_fiction', 'thriller', 'tv_movie', 'war', 'western'
    ]]

    # Create a language encoder mapping from language to integer (as used during training)
    language_encoder = {lang: idx for idx, lang in enumerate(df["language"].unique())}

    # 1. Convert the user description to a TF-IDF vector
    user_tfidf = vectorizer.transform([user_description]).toarray()

    # 2. Encode the user language (defaults to 0 if not found)
    user_language_encoded = np.array([[language_encoder.get(user_language, 0)]], dtype=np.int32)

    # 3. One-hot encode the genres
    user_genre_vector = np.zeros((1, len(genre_columns)), dtype=np.int32)
    for genre in user_genres:
        if genre in genre_columns:
            user_genre_vector[0, genre_columns.index(genre)] = 1

    # 4. Generate latent embedding using the trained encoder model
    user_embedding = encoder_model.predict([user_tfidf, user_language_encoded, user_genre_vector])

    # 5. Use KNN to find similar movies
    distances, indices = knn_model.kneighbors(user_embedding)
    indices = indices.flatten()
    distances = distances.flatten()

    # Limit recommendations to n_recommendations
    filtered_recs = list(zip(indices, distances))[:n_recommendations]

    # Retrieve movie names from the DataFrame
    df = df.reset_index(drop=True)
    recommendations = [(df.loc[idx, "name"], dist) for idx, dist in filtered_recs]

    return recommendations

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
