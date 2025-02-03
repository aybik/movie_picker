import pandas as pd
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) #main directory movie_picker

def get_set_a():
    # Load csv file from raw_data folder
    movies = pd.read_csv(os.path.join(parent_dir,'raw_data/movies.csv'))
    actors = pd.read_csv(os.path.join(parent_dir,'raw_data/actors.csv'))
    crew = pd.read_csv(os.path.join(parent_dir,'raw_data/crew.csv'))
    languages = pd.read_csv(os.path.join(parent_dir,'raw_data/languages.csv'))
    genres = pd.read_csv(os.path.join(parent_dir,'raw_data/genres.csv'))
    studios = pd.read_csv(os.path.join(parent_dir,'raw_data/studios.csv'))
    # countries = pd.read_csv(os.path.join(parent_dir,'raw_data/countries.csv'))

    # Clean up movies df
    movies = movies.drop(columns='tagline', axis=1)
    movies = movies[movies['name'].notnull() & ~movies['name'].isin(['', 'No Title'])]
    movies = movies[movies['description'].notnull() & (movies['description'] != '')] # Remove movies without description

    movies = movies.rename(columns={"date": "year"})
    # movies["year"] = movies["year"].astype(float).apply(lambda x: str(int(x)) if not pd.isna(x) else "")
    movies["year"] = pd.to_numeric(movies["year"], errors='coerce').astype("Int64")

    movies = movies.dropna(subset=['minute']) # Remove NaN values in minute
    movies['minute'] = movies['minute'].astype(int) # Change minute dtype to int
    movies = movies[(movies['minute'] > 40) & (movies['minute'] <= 240)] # Remove short and too long movies

    movies['key'] = movies['name'] + movies['year'].apply(lambda x: '' if pd.isna(x) else f" ({int(x)})")

    unique_keys = movies['key'].value_counts()
    filter_a = unique_keys[unique_keys > 1].index
    filter_a = movies[movies['key'].isin(filter_a)] # Entries not unique
    filter_a = filter_a[['id','year','rating', 'key']]
    filter_a[["year", "rating"]] = filter_a[["year", "rating"]].fillna(0)
    filter_a = filter_a.sort_values(by=["key", "year", "rating", "id"], ascending=[True, False, False, True])
    filter_result = filter_a.drop_duplicates(subset=["key"], keep="first").drop(columns=["rating", "year"])
    keys_with_count_1 = unique_keys[unique_keys == 1].index
    new_movies = movies[(movies['key'].isin(keys_with_count_1)) | (movies['id'].isin(filter_result['id']))]

    # Clean up actors df
    actors = actors[actors['role'].notnull() & (actors['role'] != '')]  # Remove actors without role
    pattern = r'footage|uncredited|Ensemble/|\d'  # Matches specific terms or any digit
    actors = actors[~actors['role'].str.contains(pattern, case=False, regex=True)]
    actors = actors.drop(columns='role', axis=1) # Drop column role

    name_counts = actors['name'].value_counts().reset_index() # Count frequency
    name_counts = name_counts[name_counts['count']>=12] # Take only those appearing >= 12 times
    actors = actors[actors['name'].isin(name_counts['name'])] # Remove unpopular actors

    new_actors = (
        actors.groupby('id')['name']
        .apply(list)  # Aggregates genres into a list
        .reset_index(name='actor_list')  # Converts to DataFrame and renames the column
    )

    # Clean up crew df
    crew = crew[crew['role'].isin(['Director', 'Writer', 'Cinematography', 'Composer'])] #'Songs', 'Producer',
    new_crew = (
        crew.groupby('id')
        .apply(lambda x: x.groupby('role')['name'].apply(list).to_dict())
        .reset_index(name='crew_dict')
    )

    # Clean up languages df
    languages = languages[languages['type'].isin(['Language', 'Primary language'])].drop(columns='type')

    # Clean up genres df
    genres['genre'] = genres['genre'].apply(lambda x: x.lower().replace(' ', '_'))
    new_genres = (
        genres.groupby('id')['genre']
        .apply(' '.join)  # Aggregates genres into a list
        .reset_index(name='genre_list')  # Converts to DataFrame and renames the column
    )

    # Clean up studios df
    new_studios = (
        studios.groupby('id')['studio']
        .apply(list)  # Aggregates genres into a list
        .reset_index(name='studio_list')  # Converts to DataFrame and renames the column
    )

    # Merge into 1 df
    data = new_movies \
        .merge(new_genres, how='left', on='id') \
        .merge(new_actors, how='left', on='id') \
        .merge(languages, how='left', on='id') \
        .merge(new_studios, how='left', on='id') \
        .merge(new_crew, how='left', on='id')

    # data[["language", "genre_list"]] = data[["language", "genre_list"]].fillna("")
    data[["actor_list", "studio_list"]] = data[["actor_list", "studio_list"]].applymap(lambda x: x if isinstance(x, list) else [])

    return data
    # NOTICE:
        # data.crew_dict has NaN
        # data.actor_list, .studio_list has []
        # data.genre_list has ''

def get_set_b():
    pass
