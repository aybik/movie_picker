import pandas as pd
import string
import numpy as np
import re
import ast
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
# from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences

def text_preprocess(sentence):
    # Basic cleaning
    sentence = sentence.strip() ## remove whitespaces
    sentence = sentence.lower() ## lowercase
    sentence = ''.join(char for char in sentence if not char.isdigit()) ## remove numbers #TODO
    # Advanced cleaning
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '') ## remove punctuation
    tokenized_sentence = word_tokenize(sentence) ## tokenize
    stop_words = set(stopwords.words('english'))
    stopwords_removed = [w for w in tokenized_sentence if not w in stop_words]
    v_lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos = "v")
        for word in stopwords_removed
    ]
    n_lemmatized = [
        WordNetLemmatizer().lemmatize(word, pos = "n")
        for word in v_lemmatized
    ]
    cleaned_sentence = ' '.join(word for word in n_lemmatized)
    return cleaned_sentence

def num_preprocess_year(value):
    scaler = RobustScaler()
    result = scaler.fit_transform(value)
    return result

def num_preprocess_min(value):
    scaler = MinMaxScaler()
    result = scaler.fit_transform(value)
    return result

def num_preprocess_rating(value):
    value = value.copy()

    median_value = value.median()
    value.fillna(median_value, inplace=True) # Impute NaN values with median

    scaler = MinMaxScaler()
    result = scaler.fit_transform(value.to_numpy().reshape(-1, 1))  # Ensure 2D input

    return result.flatten()

def fix_data_from_csv(df):
    df[["language", "genre_list"]] = df[["language", "genre_list"]].fillna("")
    return df

######################### NEW INPUT #########################


def cat_processing_genre(df, column="genre_list"):
    # Initialize MultiLabelBinarizer and transform the data
    encoder = MultiLabelBinarizer()
    genre_df = pd.DataFrame(encoder.fit_transform(df[column].str.split(' ')),
                                  columns=encoder.classes_,
                                  index=df.index)
    df = pd.concat([df, genre_df], axis=1)

    return df

def cat_processing_lan(df, column="language"):
    """
    Cleans and encodes a single categorical column (e.g., language) using LabelEncoder.
    - Keeps only the first value before delimiters (comma, slash, semicolon, pipe).
    - Encodes categorical values into numerical labels.
    """

    df[column] = df[column].astype(str).str.split(r",|/|;|\|").str[0].str.strip()

    encoder = LabelEncoder()
    df[f"{column}_encoded"] = encoder.fit_transform(df[column])

    return df

def safe_eval_column(df, column_name="crew_dict"):
    """
    Safely converts a column containing string representations of dictionaries into actual dictionaries.
    - If the value is already a dictionary, it remains unchanged.
    - If the value is a valid string dictionary, it is converted using `ast.literal_eval`.
    - If conversion fails, an empty dictionary `{}` is returned.
    """
    def safe_eval(val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)  # Convert only if it's a valid string dictionary
            except (ValueError, SyntaxError):
                return {}  # Return empty dictionary if parsing fails
        return val  # Return as is if already a dict

    df[column_name] = df[column_name].apply(safe_eval)
    return df

def extract_roles(df, column_name="crew_dict", roles=None):
    """
    Extracts specific roles (e.g., Director, Writer) from a dictionary column.
    Creates new columns for each role with lists of names.
    """
    if roles is None:
        roles = ["Director", "Writer", "Cinematography", "Composer"]

    for role in roles:
        df[role.lower()] = df[column_name].apply(
            lambda x: x.get(role, []) if isinstance(x, dict) else []
        )
    return df


def encode_list_column_with_padding(df, column_name, padding_value=0, max_length=2):
    """
    Encodes a column containing lists of categorical values (e.g., directors) and applies padding.
    - Uses LabelEncoder to encode unique values.
    - Pads sequences to a fixed length.
    """
    # Flatten unique values for encoding
    unique_values = sorted(set(value for sublist in df[column_name] for value in sublist))

    # Fit LabelEncoder once
    encoder = LabelEncoder()
    encoder.fit(unique_values)

    # Create mapping dictionary for faster lookup
    encoding_map = {label: idx for idx, label in enumerate(encoder.classes_)}

    # Apply encoding efficiently
    df[f"{column_name}_encoded"] = df[column_name].apply(lambda x: [encoding_map[v] for v in x])

    # Apply padding to ensure fixed-length sequences
    df[f"{column_name}_encoded_padded"] = list(
        pad_sequences(df[f"{column_name}_encoded"], maxlen=max_length, padding='pre', value=padding_value)
    )
    return df, len(unique_values)


######################### NEW INPUT ENDs #########################


def data_preproc(df):
    df = fix_data_from_csv(df)
    df['description'] = df['description'].apply(text_preprocess)
    df['year'] = num_preprocess_year(df[['year']])
    df['minute'] = num_preprocess_min(df[['minute']])
    df.loc[:, 'combined_rating'] = num_preprocess_rating(df[['combined_rating']]).round(2)
    df = cat_processing_genre(df,'genre_list')
    df = cat_processing_lan(df, 'language')
    return df

def data_encode(df):
    # Dictionary Processing
    df = safe_eval_column(df, column_name="crew_dict")
    df = extract_roles(df, column_name="crew_dict")

    # INPROGRESS!
    # Encoding list columns with padding
    # df, director_length = encode_list_column_with_padding(df, "director")
    # df, writer_length = encode_list_column_with_padding(df, "writer")
    # df, cinematography_length = encode_list_column_with_padding(df, "cinematography")
    # df, composer_length = encode_list_column_with_padding(df, "composer")

    # return df, {
    #     "director_length": director_length,
    #     "writer_length": writer_length,
    #     "cinematography_length": cinematography_length,
    #     "composer_length": composer_length,
    # }
    return df
