import pandas as pd
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer

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

def cat_processing_genre(df, value):
    unique_genres = set(genre for genres in df[value] for genre in eval(genres))
    for genre in unique_genres:
        df[genre] = df[value].apply(lambda x: 1 if genre in x else 0)
    return df.rename(columns={
    'science fiction': 'science_fiction',
    'tv movie': 'tv_movie'
    })

def cat_processing_lan(df, value):
    temp_lang = (df[value].value_counts()/df[value].value_counts().sum())*100
    temp = list(temp_lang.head(20).keys())
    df_filtered = df[df[value].isin(temp)]
    return df_filtered

def data_preproc(df):
    df['description'] = df['description'].apply(text_preprocess)
    df['date'] = num_preprocess_year(df[['date']])
    df['minute'] = num_preprocess_min(df[['minute']])
    cat_processing_genre(df,'genre_list')
    cat_processing_lan(df, 'language')
    return df

def text_encode(df):
    vectorizer =TfidfVectorizer()
    X = vectorizer.fit_transform(df['description'])
    X = pd.DataFrame(
        X.toarray(),
        columns = vectorizer.get_feature_names_out()
        )
    return X

def test_func():
    import os
    print(os.path.dirname(__file__))
