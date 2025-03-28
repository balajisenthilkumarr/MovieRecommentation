import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movie and credits data
def load_data():
    credits = pd.read_csv('./dataset/tmdb_5000_credits.csv')
    movies = pd.read_csv('./dataset/tmdb_5000_movies.csv')

    # Merge the two dataframes on the title column
    movies = movies.merge(credits, left_on='title', right_on='title')

    # Select relevant columns
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'original_language']]

    return movies

# Convert the JSON-like string into a list of names
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# Process the movie data to extract useful features
def process_data(movies):
    # Convert genres and keywords to lists of names
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)

    # Convert cast and crew to only the top 3 actors and directors
    movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]])  # Only top 3 actors
    movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'])

    # Create a new column 'tags' which combines genres, keywords, cast, and crew
    movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
    
    # Convert tags to lowercase
    movies['tags'] = movies['tags'].apply(lambda x: x.lower())

    return movies

# Compute the TF-IDF matrix and cosine similarity
def compute_similarity(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Get movie recommendations based on a movie title
def get_recommendations(title, cosine_sim, movies):
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
        # Try loading the processed movie data and similarity matrix from the pickle file
        movies, cosine_sim = load_pickled_data()
    except FileNotFoundError:
        # If the pickle file does not exist, process the raw data and compute similarity
        movies = load_data()
        movies = process_data(movies)
        cosine_sim = compute_similarity(movies)
        # Save the data to the pickle file for future use
        save_data(movies, cosine_sim)

    return movies, cosine_sim

if __name__ == '__main__':
    movies, cosine_sim = main()
    print(get_recommendations('The Dark Knight Rises', cosine_sim, movies))
    