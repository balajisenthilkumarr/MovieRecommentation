import pandas as pd
import ast
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the movie and credits data
def load_data():
    credits = pd.read_csv('./dataset/tmdb_5000_credits.csv')
    movies = pd.read_csv('./dataset/tmdb_5000_movies.csv')

    # Rename 'id' to 'movie_id' for consistency
    movies = movies.rename(columns={'id': 'movie_id'})

    # Log available columns
    logger.info(f"Movies columns: {movies.columns.tolist()}")
    logger.info(f"Credits columns: {credits.columns.tolist()}")

    # Merge the two dataframes on the title column
    movies = movies.merge(credits, left_on='title', right_on='title', how='left')

    # Select relevant columns, including release_date
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'original_language', 'release_date']]

    # Verify release_date exists
    if 'release_date' not in movies.columns:
        logger.error("release_date column missing in dataset")
        raise ValueError("release_date column missing in dataset")

    # Log missing release_date values
    logger.info(f"Missing release_date count: {movies['release_date'].isna().sum()}")
    logger.info(f"Sample release_date values: {movies['release_date'].dropna().head().tolist()}")

    return movies

# Convert the JSON-like string into a list of names
def convert(obj):
    L = []
    try:
        for i in ast.literal_eval(obj):
            L.append(i['name'])
    except (ValueError, SyntaxError) as e:
        logger.warning(f"Error parsing object: {e}")
    return L

# Process the movie data to extract useful features
def process_data(movies):
    # Convert genres and keywords to lists of names
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)

    # Convert cast and crew to only the top 3 actors and directors
    movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]] if pd.notna(x) else [])
    movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'] if pd.notna(x) else [])

    # Create a new column 'tags' which combines genres, keywords, cast, and crew
    movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
    
    # Convert tags to lowercase
    movies['tags'] = movies['tags'].apply(lambda x: x.lower())

    # Ensure release_date is preserved
    movies['release_date'] = movies['release_date'].fillna('')
    
    return movies

# Compute the TF-IDF matrix and cosine similarity
def compute_similarity(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Precompute recommendations for all movies
def precompute_recommendations(movies, cosine_sim):
    recommendation_cache = {}
    for idx, title in enumerate(movies['title']):
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Get top 10 similar movies
        movie_indices = [i[0] for i in sim_scores]
        recommendation_cache[title] = movies['title'].iloc[movie_indices].tolist()
    
    with open('recommendation_cache.pkl', 'wb') as file:
        pickle.dump(recommendation_cache, file)
    
    return recommendation_cache

# Get movie recommendations based on a movie title
def get_recommendations(title, cosine_sim=None, movies=None):
    if os.path.exists('recommendation_cache.pkl'):
        with open('recommendation_cache.pkl', 'rb') as file:
            recommendation_cache = pickle.load(file)
        if title in recommendation_cache:
            return recommendation_cache[title]
    
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Save movie data and cosine similarity matrix to pickle file
def save_data(movies, cosine_sim):
    with open('movie_data.pkl', 'wb') as file:
        pickle.dump((movies, cosine_sim), file)

# Load movie data and cosine similarity matrix from pickle file
def load_pickled_data():
    with open('movie_data.pkl', 'rb') as file:
        movies, cosine_sim = pickle.load(file)
    return movies, cosine_sim

# Main function to load, process data, and compute similarity
def main():
    try:
        movies, cosine_sim = load_pickled_data()
        if not os.path.exists('recommendation_cache.pkl'):
            precompute_recommendations(movies, cosine_sim)
    except FileNotFoundError:
        movies = load_data()
        movies = process_data(movies)
        cosine_sim = compute_similarity(movies)
        save_data(movies, cosine_sim)
        precompute_recommendations(movies, cosine_sim)

    return movies, cosine_sim

if __name__ == '__main__':
    movies, cosine_sim = main()
    print(get_recommendations('The Dark Knight Rises', cosine_sim, movies))