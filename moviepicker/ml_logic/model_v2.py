from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


# Vectorize descriptions function is changed to give array rather than matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

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



# Create np arrays for tfidf, language, genre to be utilized in autoencoder
def prepare_model_inputs(df, tfidf_dim=2500):
    """
    Prepares input features for the autoencoder model.

    Parameters:
        df (pd.DataFrame): The dataset containing movie descriptions, language, and genres.
        tfidf_dim (int): The maximum number of features for TF-IDF vectorization.

    Returns:
        tuple: (tfidf_array, num_languages, language_data_np, genres_data_np, num_genres)
    """
    # ---------------------------
    # 1. TF-IDF Vectorization
    # ---------------------------
    tfidf_array, vectorizer = vectorize_descriptions(df, 'description', tfidf_dim)

    # ---------------------------
    # 2. Language Encoding
    # ---------------------------
    num_languages = df['language_encoded'].nunique()
    language_data_np = df['language_encoded'].values.reshape(-1, 1).astype(np.int32)

    # ---------------------------
    # 3. Genre Extraction
    # ---------------------------
    genre_columns = ['action', 'adventure', 'animation', 'comedy', 'crime', 'documentary',
                     'drama', 'family', 'fantasy', 'history', 'horror', 'music', 'mystery',
                     'romance', 'science_fiction', 'thriller', 'tv_movie', 'war', 'western']
    genres_data_np = df[genre_columns].values.astype(np.int32)
    num_genres = len(genre_columns)  # Automatically detect number of genres

    return tfidf_array, vectorizer, num_languages, language_data_np, genres_data_np, num_genres

# Create encoder which will be used in autoencoder function and optimized during autoencoder.fit
def build_encoder(tfidf_dim, num_languages, num_genres):
    """
    Builds an encoder model that fuses:
      - A TF-IDF vector input (continuous, shape: [tfidf_dim])
      - A language input (integer, shape: [1])
      - A one-hot encoded genres input (shape: [num_genres])

    Parameters:
      tfidf_dim (int): Dimensionality of the TF-IDF vector (e.g., 2500).
      num_languages (int): Total number of language categories (max language index + 1).
      num_genres (int): Number of genres (should be 19 for your columns).

    Returns:
      encoder_model (tf.keras.Model): A model that outputs a fused latent embedding.
    """

    # -------------------------
    # TF-IDF Branch
    # -------------------------
    tfidf_input = tf.keras.layers.Input(shape=(tfidf_dim,), name="tfidf_input")
    tfidf_dense = tf.keras.layers.Dense(128, activation='relu', name="tfidf_dense")(tfidf_input)

    # -------------------------
    # Language Branch
    # -------------------------
    language_input = tf.keras.layers.Input(shape=(1,), name="language_input")
    language_embedding = tf.keras.layers.Embedding(
        input_dim=num_languages,
        output_dim=8,
        name="language_embedding"
    )(language_input)
    language_vector = tf.keras.layers.Flatten(name="language_flatten")(language_embedding)

    # -------------------------
    # Genres Branch (One-hot encoded)
    # -------------------------
    genre_input = tf.keras.layers.Input(shape=(num_genres,), name="genre_input")
    # Pass the one-hot vector through a dense layer to learn a compressed representation.
    genre_dense = tf.keras.layers.Dense(32, activation='relu', name="genre_dense")(genre_input)

    # -------------------------
    # Merge Branches
    # -------------------------
    # Concatenate the outputs of all branches.
    merged = tf.keras.layers.concatenate([tfidf_dense, language_vector, genre_dense], name="merged_features")
    x = tf.keras.layers.Dense(64, activation='relu', name="dense_1")(merged)
    final_embedding = tf.keras.layers.Dense(32, activation='relu', name="final_embedding")(x)

    # Build the encoder model
    encoder_model = tf.keras.models.Model(
        inputs=[tfidf_input, language_input, genre_input],
        outputs=final_embedding
    )

    return encoder_model


# Create autoencoder
def build_autoencoder(tfidf_dim, num_languages, num_genres):
    """
    Builds an autoencoder that uses:
      - The encoder from build_encoder to produce a 32-d latent embedding.
      - Three decoder branches to reconstruct:
          A. The original TF-IDF vector.
          B. The language (as a probability distribution over num_languages).
          C. The one-hot encoded genres vector.

    The autoencoder is compiled with MSE loss for TF-IDF, sparse categorical crossentropy for language,
    and binary crossentropy for genres.
    """

    # Define the inputs (they will be passed to both encoder and as targets later)
    tfidf_input = tf.keras.layers.Input(shape=(tfidf_dim,), name="tfidf_input")
    language_input = tf.keras.layers.Input(shape=(1,), name="language_input")
    genre_input = tf.keras.layers.Input(shape=(num_genres,), name="genre_input")

    # Build the encoder and get the latent representation.
    encoder = build_encoder(tfidf_dim, num_languages, num_genres)
    latent = encoder([tfidf_input, language_input, genre_input])

    # -------------------------
    # Decoder for TF-IDF reconstruction
    # -------------------------
    decoder_tfidf = tf.keras.layers.Dense(64, activation='relu', name="decoder_tfidf_dense")(latent)
    tfidf_output = tf.keras.layers.Dense(tfidf_dim, activation='relu', name="tfidf_output")(decoder_tfidf)

    # -------------------------
    # Decoder for Language reconstruction
    # -------------------------
    decoder_language = tf.keras.layers.Dense(16, activation='relu', name="decoder_language_dense")(latent)
    # Output is a probability distribution over languages
    language_output = tf.keras.layers.Dense(num_languages, activation='softmax', name="language_output")(decoder_language)

    # -------------------------
    # Decoder for Genres reconstruction
    # -------------------------
    decoder_genre = tf.keras.layers.Dense(16, activation='relu', name="decoder_genre_dense")(latent)
    # For multi-label, we use sigmoid activation; if it's strictly one-hot, you could use softmax.
    genre_output = tf.keras.layers.Dense(num_genres, activation='sigmoid', name="genre_output")(decoder_genre)

    # Build the autoencoder model.
    autoencoder_model = tf.keras.models.Model(
        inputs=[tfidf_input, language_input, genre_input],
        outputs=[tfidf_output, language_output, genre_output],
        name="autoencoder"
    )

    # Compile the autoencoder:
    # - For TF-IDF, we use mean squared error.
    # - For language, we use sparse categorical crossentropy (the target should be an integer).
    # - For genres, binary crossentropy is appropriate for multi-label reconstruction.
    autoencoder_model.compile(
        optimizer='adam',
        loss={
            'tfidf_output': 'mse',
            'language_output': 'sparse_categorical_crossentropy',
            'genre_output': 'binary_crossentropy'
        },
        loss_weights={
            'tfidf_output': 1.0,
            'language_output': 1.0,
            'genre_output': 1.0
        }
    )

    return autoencoder_model, encoder


def train_autoencoder(autoencoder_model, tfidf_array, language_data_np, genres_data_np, batch_size=16, epochs=100):
    """
    Trains the autoencoder model using the given input data.

    Parameters:
        autoencoder_model (tf.keras.Model): The compiled autoencoder model.
        tfidf_array (np.array): TF-IDF input data.
        language_data_np (np.array): Encoded language data.
        genres_data_np (np.array): One-hot encoded genres data.
        batch_size (int): Batch size for training. Default is 16.
        epochs (int): Number of training epochs. Default is 50.

    Returns:
        history (tf.keras.callbacks.History): Training history object containing loss values.
    """
    # Define early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor='val_loss',       # Monitor the validation loss
        patience=3,               # Stop training if no improvement for 3 epochs
        restore_best_weights=True # Restore the best model weights
    )

    # Train the model
    history = autoencoder_model.fit(
        x=[tfidf_array, language_data_np, genres_data_np],
        y=[tfidf_array, language_data_np, genres_data_np],
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stop]
    )

    return history

def extract_latent_embeddings(encoder_model, tfidf_array, language_data_np, genres_data_np):
    """
    Extracts latent embeddings from the encoder model.

    Parameters:
        encoder_model (tf.keras.Model): The trained encoder model.
        tfidf_array (np.array): TF-IDF input data.
        language_data_np (np.array): Encoded language data.
        genres_data_np (np.array): One-hot encoded genres data.

    Returns:
        latent_embeddings (np.array): The extracted latent representations.
    """
    latent_embeddings = encoder_model.predict([tfidf_array, language_data_np, genres_data_np])
    return latent_embeddings



def knn_fit(latent_embeddings, n_neighbors=10, metric='cosine'):
    """
    Fits a KNN model for similarity search using the latent embeddings.

    Parameters:
        latent_embeddings (np.array): The extracted latent embeddings from the encoder.
        n_neighbors (int): Number of nearest neighbors to find. Default is 5.
        metric (str): Distance metric for KNN. Default is 'cosine'.

    Returns:
        knn_model (NearestNeighbors): The trained KNN model.
    """
    knn_model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)  # +1 to exclude the queried movie later
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
        n_recommendations (int): Number of movie recommendations to return. Default is 5.

    Returns:
        list: A list of tuples containing recommended movies and their distances.
    """
    # Convert user input and DataFrame names to lowercase for case-insensitive matching
    matched_rows = df[df["name"].str.lower() == user_input.lower()]

    if matched_rows.empty:
        print("Movie not found.")
        return []

    # Get the first matching index
    sample_index = matched_rows.index[0]
    print(f"Found movie '{user_input}'.")

    # Retrieve KNN results
    distances, indices = knn_model.kneighbors(latent_embeddings[sample_index].reshape(1, -1))

    # Convert to 1D arrays
    indices = indices.flatten()
    distances = distances.flatten()

    # Filter out the queried movie
    filtered_recs = [(idx, dist) for idx, dist in zip(indices, distances) if idx != sample_index]

    # Handle case where no recommendations remain
    if not filtered_recs:
        print("No recommendations found after filtering out the queried movie.")
        return []

    # Limit recommendations to `n_recommendations`
    filtered_recs = filtered_recs[:n_recommendations]

    # Retrieve movie names for recommendations
    recommendations = [(df.loc[idx, "name"], dist) for idx, dist in filtered_recs]

    return recommendations


def recommend_movies_by_details(user_description, user_language, user_genres, df, encoder_model, knn_model, vectorizer, tfidf_dim=2500):
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
        tfidf_dim (int): The number of TF-IDF features (default: 2500).

    Returns:
        list: A list of recommended movie names and distances.
    """

    # 🔹 Auto-detect genre columns
    genre_columns = [col for col in df.columns if col in [
        'action', 'adventure', 'animation', 'comedy', 'crime', 'documentary',
        'drama', 'family', 'fantasy', 'history', 'horror', 'music', 'mystery',
        'romance', 'science_fiction', 'thriller', 'tv_movie', 'war', 'western'
    ]]

    # 🔹 Auto-create language encoder (maps language names to integer encodings)
    language_encoder = {lang: idx for idx, lang in enumerate(df["language"].unique())}

    # 1️⃣ Convert the user description to a TF-IDF vector
    user_tfidf = vectorizer.transform([user_description]).toarray()

    # 2️⃣ Encode the user language
    user_language_encoded = np.array([[language_encoder.get(user_language, 0)]], dtype=np.int32)

    # 3️⃣ One-hot encode the genres
    user_genre_vector = np.zeros((1, len(genre_columns)), dtype=np.int32)
    for genre in user_genres:
        if genre in genre_columns:
            user_genre_vector[0, genre_columns.index(genre)] = 1

    # 4️⃣ Generate latent embedding using the encoder model
    user_embedding = encoder_model.predict([user_tfidf, user_language_encoded, user_genre_vector])

    # 5️⃣ Use KNN to find similar movies
    distances, indices = knn_model.kneighbors(user_embedding)

    # Convert to 1D arrays
    indices = indices.flatten()
    distances = distances.flatten()

    # Retrieve movie names for recommendations
    df = df.reset_index(drop=True)
    recommendations = [(df.loc[idx, "name"], dist) for idx, dist in zip(indices, distances)]

    return recommendations
