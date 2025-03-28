import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging
from functools import lru_cache
import random
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import time
from rapidfuzz import process, fuzz
from model import main, get_recommendations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})  # Allow frontend origin

# Download necessary NLTK resources
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

# Load NLP models
classifier = pipeline("zero-shot-classification")  # For intent detection
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # For semantic similarity
ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Load data for content-based filtering (TMDB dataset)
try:
    movies, cosine_sim = main()
except Exception as e:
    logger.error(f"Failed to load movie data: {e}")
    raise

# Load data for collaborative filtering (MovieLens dataset)
movies_df = pd.read_csv(
    "dataset/ml-1m/movies.dat",
    sep="::",
    names=["movieId", "title", "genres"],
    engine="python",
    encoding="latin1"
)

# Load the links.csv file for MovieLens to TMDB mapping
links_df = pd.read_csv(
    "dataset/dataset2/links.csv",
    dtype={"movieId": int, "imdbId": str, "tmdbId": str}
)

# Create mapping dictionary
movielens_to_tmdb = dict(zip(links_df["movieId"], links_df["tmdbId"].astype(str)))

# Load the trained collaborative filtering model
with open("svd_model.pkl", "rb") as file:
    collab_model = pickle.load(file)

# TMDB API Key (replace with your valid key)
TMDB_API_KEY = "5bd7d31b6e1466d5799253aa07b28a02"

# Configure session with retry strategy
def create_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

session = create_session()
poster_cache = {}

# Load movie dataset for chatbot
MOVIE_TITLES = movies["title"].tolist()

@lru_cache(maxsize=1000)
def search_tmdb_movie(title):
    """Search for a movie in TMDB by title and return its TMDB ID."""
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "query": title,
            "language": "en-US",
            "page": 1,
        }
        response = session.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data["results"]:
            return data["results"][0]["id"]
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching TMDB for movie {title}: {e}")
        return None

@lru_cache(maxsize=1000)
def fetch_poster(movie_id):
    """Fetch movie poster with caching and error handling."""
    if movie_id in poster_cache:
        return poster_cache[movie_id]
    
    time.sleep(0.5)  # Delay to avoid TMDB rate limiting
    
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {"api_key": TMDB_API_KEY}
        response = session.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if "poster_path" in data and data["poster_path"]:
            poster_path = data["poster_path"]
            full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
            poster_cache[movie_id] = full_path
            return full_path
        return "https://placehold.co/500x750?text=No+Poster"
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching poster for movie {movie_id}: {e}")
        return "https://placehold.co/500x750?text=No+Poster"

def map_movielens_to_tmdb(movielens_movie_id):
    """Map MovieLens movieId to TMDB tmdbId."""
    try:
        tmdb_id = movielens_to_tmdb.get(int(movielens_movie_id))
        if tmdb_id and tmdb_id != "nan":
            return int(tmdb_id)
        return None
    except Exception as e:
        logger.error(f"Error mapping MovieLens ID {movielens_movie_id} to TMDB: {e}")
        return None

def get_collaborative_recommendations(user_id, n=10):
    """Generate top-N collaborative filtering recommendations for a given user."""
    try:
        all_movie_ids = movies_df["movieId"].unique()
        predictions = [(movie_id, collab_model.predict(user_id, movie_id).est) for movie_id in all_movie_ids]

        # Sort by highest predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_movies = predictions[:n]

        recommendations = []
        for movie_id, rating in top_movies:
            movie_title = movies_df[movies_df["movieId"] == movie_id]["title"].values[0]
            tmdb_id = map_movielens_to_tmdb(movie_id)
            poster_url = fetch_poster(tmdb_id) if tmdb_id else "https://placehold.co/500x750?text=No+Poster"
            recommendations.append({
                "title": movie_title,
                "movie_id": int(tmdb_id) if tmdb_id else movie_id,
                "predicted_rating": round(rating, 2),
                "poster": poster_url,
            })
        return recommendations
    except Exception as e:
        logger.error(f"Error in collaborative filtering: {e}")
        return []

@app.route("/api/collaborative_recommendations/<int:user_id>", methods=["GET"])
def collaborative_recommend(user_id):
    """API endpoint for collaborative filtering recommendations."""
    recommendations = get_collaborative_recommendations(user_id)
    if recommendations:
        return jsonify({"user_id": user_id, "recommendations": recommendations})
    else:
        return jsonify({"error": "Could not generate recommendations"}), 404

@app.route("/api/movies", methods=["GET"])
def get_movies():
    try:
        page = int(request.args.get("page", 1))
        page_size = int(request.args.get("page_size", 10))
        
        if page < 1 or page_size < 1:
            return jsonify({"error": "Page and page_size must be positive integers."}), 400
            
        # Create a copy and clean the data
        movies_cleaned = movies.copy()
        movies_cleaned['overview'] = movies_cleaned['overview'].fillna('No overview available')
        movies_cleaned['tags'] = movies_cleaned['tags'].fillna('')
        movies_cleaned = movies_cleaned.dropna(subset=['title'])

        # Pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        total_movies = len(movies_cleaned)
        
        if start_idx >= total_movies:
            return jsonify({"error": "Page number exceeds available data."}), 404

        paginated_movies = movies_cleaned.iloc[start_idx:end_idx]

        # Add posters to movies
        movies_with_posters = []
        for _, movie in paginated_movies.iterrows():
            movie_data = movie.to_dict()
            poster_url = fetch_poster(movie['movie_id'])
            movie_data['poster'] = poster_url if poster_url else "https://placehold.co/500x750?text=No+Poster"
            movies_with_posters.append(movie_data)

        response_data = {
            "movies": movies_with_posters,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_movies": total_movies,
                "total_pages": (total_movies + page_size - 1) // page_size
            }
        }

        return jsonify(response_data)

    except ValueError:
        return jsonify({"error": "Invalid page or page_size parameters."}), 400
    except Exception as e:
        logger.error(f"Error processing movies request: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/recommendations/<int:movie_id>', methods=['GET'])
def recommendations(movie_id):
    try:
        # Find the movie in the dataframe
        movie_row = movies[movies['movie_id'] == movie_id]
        
        if movie_row.empty:
            return jsonify({"error": "Movie not found"}), 404
        
        # Get the title of the movie
        movie_title = movie_row['title'].iloc[0]
        
        # Get recommendations using your existing model function
        recommended_titles = get_recommendations(movie_title, cosine_sim, movies)
        
        if not recommended_titles:
            return jsonify({"error": "Could not generate recommendations"}), 404

        # Convert recommended titles to full movie data
        recommended_movies = []
        for title in recommended_titles:
            movie_data = movies[movies['title'] == title].iloc[0].to_dict()
            poster_url = fetch_poster(movie_data['movie_id'])
            movie_data['poster'] = poster_url if poster_url else "https://placehold.co/500x750?text=No+Poster"
            recommended_movies.append(movie_data)

        response_data = {
            "movie_id": movie_id,
            "movie_title": movie_title,
            "recommendations": recommended_movies
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error processing recommendation request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/movie/<int:movie_id>', methods=['GET'])
def get_movie(movie_id):
    try:
        # Find the movie in the dataframe
        movie = movies[movies['movie_id'] == movie_id]
        
        if movie.empty:
            return jsonify({"error": "Movie not found"}), 404
        
        # Convert movie data to dict
        movie_data = movie.iloc[0].to_dict()
        
        # Add poster URL
        poster_url = fetch_poster(movie_data['movie_id'])
        movie_data['poster'] = poster_url if poster_url else "https://placehold.co/500x750?text=No+Poster"
        
        return jsonify(movie_data)
        
    except Exception as e:
        logger.error(f"Error processing movie request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/hybrid_recommendations/<int:user_id>/<int:movie_id>', methods=['GET'])
def hybrid_recommend(user_id, movie_id):
    """API that combines Content-Based and Collaborative Filtering recommendations."""
    
    # Get Collaborative Recommendations
    collab_recommendations = get_collaborative_recommendations(user_id, n=5)

    try:
        # Ensure movie_id exists in the content-based dataset
        movie_row = movies[movies['movie_id'] == movie_id]
        if movie_row.empty:
            return jsonify({"error": f"Movie ID {movie_id} not found in content-based dataset"}), 404

        # Get the movie title for content-based recommendations
        movie_title = movie_row['title'].iloc[0]

        # Get content-based recommendations
        recommended_titles = get_recommendations(movie_title, cosine_sim, movies)[:5]

        # Convert recommended titles to full movie data with posters
        content_recommendations = []
        for title in recommended_titles:
            movie_data = movies[movies['title'] == title].iloc[0].to_dict()
            poster_url = fetch_poster(movie_data['movie_id'])
            content_recommendations.append({
                "title": movie_data['title'],
                "movie_id": int(movie_data['movie_id']),
                "poster": poster_url if poster_url else "https://placehold.co/500x750?text=No+Poster",
                "genres": movie_data.get('genres', []),
                "overview": movie_data.get('overview', "No overview available"),
            })

        # Format response
        response_data = {
            "user_id": user_id,
            "movie_id": movie_id,
            "movie_title": movie_title,
            "hybrid_recommendations": {
                "collaborative": collab_recommendations,
                "content_based": content_recommendations
            }
        }
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error in hybrid recommendation: {e}")
        return jsonify({"error": "Could not generate hybrid recommendations"}), 500

# Chatbot Functionality
def extract_genre(message):
    genres = [
        "action", "adventure", "animation", "comedy", "crime", "documentary",
        "drama", "family", "fantasy", "history", "horror", "music", "mystery",
        "romance", "science fiction", "sci-fi", "thriller", "war", "western",
    ]
    found_genres = []
    message_lower = message.lower()
    for genre in genres:
        if genre in message_lower:
            found_genres.append(genre)
    return found_genres

def extract_movie_title(message, movie_titles):
    for title in movie_titles:
        if title.lower() in message.lower():
            return title
    return None

def get_movies_by_genre(genre, movies_df, count=5):
    try:
        genre = genre.lower()
        filtered_movies = []
        
        for _, movie in movies_df.iterrows():
            if isinstance(movie["genres"], list):
                movie_genres = [g.lower() if isinstance(g, str) else "" for g in movie["genres"]]
                if genre in movie_genres:
                    filtered_movies.append(movie)
            elif isinstance(movie["genres"], str):
                if genre.lower() in movie["genres"].lower():
                    filtered_movies.append(movie)

        if filtered_movies:
            selected_movies = random.sample(filtered_movies, min(count, len(filtered_movies)))
            recommendations = []
            for movie in selected_movies:
                poster_url = fetch_poster(movie['movie_id'])
                recommendations.append({
                    "title": movie["title"],
                    "movie_id": int(movie["movie_id"]),
                    "poster": poster_url if poster_url else "https://placehold.co/500x750?text=No+Poster",
                })
            return recommendations
        return []
    except Exception as e:
        logger.error(f"Error getting movies by genre: {e}")
        return []

def get_recommendations_from_input(message, movies_df, cosine_sim, count=5):
    try:
        # Check for genre recommendations
        genres = extract_genre(message)
        if genres:
            return get_movies_by_genre(genres[0], movies_df, count)
        
        # Check for movie similarity
        all_titles = list(movies_df['title'].values)
        movie_title = extract_movie_title(message, all_titles)
        
        if movie_title:
            recommended_titles = get_recommendations(movie_title, cosine_sim, movies_df)
            recommended_movies = []
            
            for title in recommended_titles[:count]:
                movie = movies_df[movies_df['title'] == title].iloc[0]
                poster_url = fetch_poster(movie['movie_id'])
                recommended_movies.append({
                    'title': movie['title'],
                    'movie_id': int(movie['movie_id']),
                    'poster': poster_url if poster_url else "https://placehold.co/500x750?text=No+Poster",
                })
            return recommended_movies
        
        # Default to random recommendations
        random_movies = movies_df.sample(count)
        recommendations = []
        for _, movie in random_movies.iterrows():
            poster_url = fetch_poster(movie['movie_id'])
            recommendations.append({
                'title': movie['title'],
                'movie_id': int(movie['movie_id']),
                'poster': poster_url if poster_url else "https://placehold.co/500x750?text=No+Poster",
            })
        return recommendations
                
    except Exception as e:
        logger.error(f"Error getting recommendations from input: {e}")
        return []

def detect_intent(user_message):
    labels = {
        "recommend_movies": "User wants movie recommendations",
        "find_similar": "User is asking for movies similar to another movie",
        "get_movie_details": "User wants information about a specific movie",
        "extract_genre": "User wants movie recommendations based on genre",
    }
    result = classifier(user_message, list(labels.values()))
    best_match = result["labels"][0]
    for key, value in labels.items():
        if value == best_match:
            return key
    return "unknown"

def extract_entities(user_message):
    entities = {"MOVIE": []}
    results = ner_model(user_message)

    current_entity = ""
    movie_candidates = []

    for res in results:
        entity_text = res["word"].replace("##", "")
        entity_label = res["entity"]
        if entity_label.startswith("B-") or entity_label.startswith("I-"):
            if res["word"].startswith("##"):
                current_entity += entity_text
            else:
                if current_entity:
                    movie_candidates.append(current_entity)
                current_entity = entity_text

    if current_entity:
        movie_candidates.append(current_entity)

    for movie in movie_candidates:
        fuzzy_result = process.extractOne(
            movie.lower(),
            [title.lower() for title in MOVIE_TITLES],
            scorer=fuzz.ratio,
            score_cutoff=75,
        )
        if fuzzy_result:
            match = fuzzy_result[0]
            validated_movie = search_tmdb_movie(match) if match else None
            if validated_movie:
                entities["MOVIE"].append(validated_movie)

    if len(entities["MOVIE"]) == 1:
        entities["MOVIE"] = entities["MOVIE"][0]

    return entities

def safe_json_response(data):
    if isinstance(data, np.int64):
        return int(data)
    elif isinstance(data, dict):
        return {k: safe_json_response(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [safe_json_response(i) for i in data]
    return data

def format_recommendations(recommendations):
    if recommendations:
        movie_titles = [movie["title"] for movie in recommendations]
        return f"Here are some movies you might enjoy: {', '.join(movie_titles)}"
    return "I couldn't find any recommendations."

def format_genre_movies(genre_movies, genre):
    if genre_movies:
        movie_titles = [movie["title"] for movie in genre_movies]
        return f"Here are some {genre} movies: {', '.join(movie_titles)}"
    return f"No movies found for genre '{genre}'."

def format_similar_movies(similar_movies, movie_title):
    if similar_movies:
        movie_titles = [movie["title"] for movie in similar_movies]
        return f"If you liked {movie_title}, you might also enjoy: {', '.join(movie_titles)}"
    return f"Couldn't find similar movies for '{movie_title}'."

def get_movie_details(movie_title):
    try:
        movie = movies[movies['title'].str.lower() == movie_title.lower()].iloc[0]
        overview = movie['overview'] if 'overview' in movie and movie['overview'] else "No overview available."
        poster_url = fetch_poster(movie['movie_id'])
        return (
            f"Here's what I know about '{movie_title}': {overview}",
            {
                "title": movie_title,
                "movie_id": int(movie["movie_id"]),
                "overview": overview,
                "poster": poster_url if poster_url else "https://placehold.co/500x750?text=No+Poster",
            },
        )
    except IndexError:
        return f"Sorry, I couldn't find details for '{movie_title}'.", None

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # Detect user intent
        action = detect_intent(user_message)
        
        # Extract movie title and genre from user input
        entities = extract_entities(user_message)
        movie_title = entities.get("MOVIE")
        genre = extract_genre(user_message)

        # Handle different chatbot actions
        if action == 'recommend_movies':
            recommendations = get_recommendations_from_input(user_message, movies, cosine_sim)
            response_text = format_recommendations(recommendations)
            action_result = recommendations

        elif action == 'extract_genre' and genre:
            genre_movies = get_movies_by_genre(genre[0], movies)
            response_text = format_genre_movies(genre_movies, genre[0])
            action_result = genre_movies

        elif action == 'get_movie_details' and movie_title:
            response_text, action_result = get_movie_details(movie_title)
        
        elif action == 'find_similar' and movie_title:
            similar_movies = get_recommendations_from_input(f"like {movie_title}", movies, cosine_sim)
            response_text = format_similar_movies(similar_movies, movie_title)
            action_result = similar_movies

        else:
            response_text = "I'm not sure what you're asking. Try asking for movie recommendations or details about a specific movie!"
            action_result = []

        return jsonify({"response": response_text, "action": action, "result": action_result})

    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("\n🚀 Available API Routes:")
    for rule in app.url_map.iter_rules():
        print(rule)
    print("\n")
    app.run(debug=True, host="0.0.0.0", port=5000)