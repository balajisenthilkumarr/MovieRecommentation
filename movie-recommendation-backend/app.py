import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging
from functools import lru_cache, wraps
import random
import pandas as pd
import pickle
import time
import threading
import json
from pymongo import MongoClient  # Import MongoClient for MongoDB
from datetime import datetime  # For tracking login timestamps
import os
from model import main, get_recommendations
from transformers import pipeline
import nltk 
from sentence_transformers import SentenceTransformer
from rapidfuzz import process, fuzz
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})


try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

# Load NLP models (add after data loading in app.py)
classifier = pipeline("zero-shot-classification")  # For intent detection
ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")  # For entity extraction

# MongoDB setup
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["movie_recommendation_db"]  # Database name
users_collection = db["users"]  # Collection name
movies_collection=db["movies_collection"]

# Initialize the database with a default admin user if not exists
def init_db():
    """Initialize the database with a default admin user."""
    admin_user = {
        "id": 1,  # Ensure the id field is present
        "username": "admin",
        "password": "admin123",
        "role": "admin",
        "last_login": None
    }
    if not users_collection.find_one({"username": "admin"}):
        users_collection.insert_one(admin_user)
        logger.info("Default admin user created.")
    else:
        # Update the existing admin user to ensure it has an id field
        users_collection.update_one(
            {"username": "admin"},
            {"$set": {"id": 1}},
            upsert=True
        )
        logger.info("Ensured admin user has an id field.")

init_db()

# Load data for content-based filtering (TMDB dataset)
try:
    movies, cosine_sim = main()
except Exception as e:
    logger.error(f"Failed to load movie data: {e}")
    raise

# Load ratings for collaborative filtering
ratings_df = pd.read_csv(
    "dataset/dataset2/ratings.csv",
    dtype={"userId": int, "movieId": int, "rating": float, "timestamp": int}
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

# TMDB API Key
TMDB_API_KEY = "5bd7d31b6e1466d5799253aa07b28a02"

# Store user preferences in memory (replace with a database in production)
user_preferences = {}

# Load or initialize the local poster cache
POSTER_CACHE_FILE = "poster_cache.json"
if os.path.exists(POSTER_CACHE_FILE):
    with open(POSTER_CACHE_FILE, "r") as f:
        poster_cache = json.load(f)
else:
    poster_cache = {}

# Configure session with retry strategy
def create_session():
    session = requests.Session()
    retries = Retry(
        total=6,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

session = create_session()

MOVIE_TITLES = movies["title"].tolist()


# Global lock for TMDB requests
tmdb_lock = threading.Lock()

# Rate limiting decorator with logging
def rate_limit(max_per_second):
    min_interval = 1.0 / float(max_per_second)
    def decorator(f):
        last_called = [0.0]
        @wraps(f)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait_time = min_interval - elapsed
            if wait_time > 0:
                logger.info(f"Rate limiting: Waiting {wait_time:.2f} seconds before making request")
                time.sleep(wait_time)
            result = f(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator

@rate_limit(max_per_second=3)
@lru_cache(maxsize=1000)
def search_tmdb_movie(title):
    """Search for a movie in TMDB by title and return its TMDB ID."""
    with tmdb_lock:
        try:
            url = "https://api.themoviedb.org/3/search/movie"
            params = {
                "api_key": TMDB_API_KEY,
                "query": title,
                "language": "en-US",
                "page": 1,
            }
            logger.info(f"Searching TMDB for movie title: {title}")
            response = session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data["results"]:
                tmdb_id = data["results"][0]["id"]
                logger.info(f"Found TMDB ID {tmdb_id} for movie title: {title}")
                return tmdb_id
            logger.warning(f"No TMDB ID found for movie title: {title}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching TMDB for movie {title}: {e}")
            return None

@rate_limit(max_per_second=3)
@lru_cache(maxsize=1000)
def fetch_poster(movie_id):
    """Fetch movie poster URL directly from TMDB."""
    if not movie_id:
        logger.warning("No movie ID provided for fetching poster.")
        return None

    if str(movie_id) in poster_cache:
        return poster_cache[str(movie_id)]
    
    with tmdb_lock:
        try:
            url = f"https://api.themoviedb.org/3/movie/{movie_id}"
            params = {"api_key": TMDB_API_KEY}
            logger.info(f"Fetching poster for movie ID {movie_id}")
            response = session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            if "poster_path" in data and data["poster_path"]:
                poster_path = data["poster_path"]
                full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
                poster_cache[str(movie_id)] = full_path
                with open(POSTER_CACHE_FILE, "w") as f:
                    json.dump(poster_cache, f)
                logger.info(f"Successfully fetched and cached poster for movie ID {movie_id}")
                return full_path
            logger.warning(f"No poster found for movie ID {movie_id}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching poster for movie {movie_id}: {e}")
            return None

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

def get_popular_movies(n=15):
    """Get popular movies based on average ratings from ratings_df."""
    movie_ratings = ratings_df.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    popular_movies = movie_ratings[movie_ratings["count"] > 50].sort_values("mean", ascending=False)
    top_movie_ids = popular_movies["movieId"].head(n).tolist()
    
    recommendations = []
    for movie_id in top_movie_ids:
        tmdb_id = map_movielens_to_tmdb(movie_id)
        if tmdb_id:
            movie_row = movies[movies["movie_id"] == tmdb_id]
            if not movie_row.empty:
                movie_title = movie_row["title"].values[0]
                poster_url = fetch_poster(tmdb_id)
                recommendations.append({
                    "title": movie_title,
                    "movie_id": int(tmdb_id),
                    "predicted_rating": round(popular_movies[popular_movies["movieId"] == movie_id]["mean"].iloc[0], 2),
                    "poster": poster_url if poster_url else "Poster unavailable",
                })
    logger.info(f"Popular movies recommendations: {recommendations}")
    return recommendations

def get_content_based_fallback(user_id, n=10):
    """Fallback to content-based recommendations using user preferences."""
    if user_id not in user_preferences:
        logger.info(f"No preferences found for user {user_id}, returning popular movies")
        return get_popular_movies(n)

    user_prefs = user_preferences[user_id]
    user_genres = [genre.lower() for genre in user_prefs.get("genres", [])]
    user_actors = [actor.lower() for actor in user_prefs.get("actors", [])]
    user_directors = [director.lower() for director in user_prefs.get("directors", [])]

    if not (user_genres or user_actors or user_directors):
        logger.info(f"No specific preferences (genres, actors, directors) for user {user_id}, returning popular movies")
        return get_popular_movies(n)

    movies_with_scores = []
    for _, movie in movies.iterrows():
        score = 0
        movie_genres = [genre.lower() for genre in movie.get("genres", [])]
        movie_actors = [actor.lower() for actor in movie.get("cast", [])]
        movie_directors = [director.lower() for director in movie.get("directors", [])]

        genre_overlap = sum(1 for genre in user_genres if genre in movie_genres)
        score += genre_overlap * 3

        actor_overlap = sum(1 for actor in user_actors if actor in movie_actors)
        score += actor_overlap * 2

        director_overlap = sum(1 for director in user_directors if director in movie_directors)
        score += director_overlap * 2

        if score > 0:
            movies_with_scores.append((movie, score))

    movies_with_scores.sort(key=lambda x: x[1], reverse=True)
    top_movies = movies_with_scores[:n]

    recommendations = []
    for movie, score in top_movies:
        poster_url = fetch_poster(movie["movie_id"])
        recommendations.append({
            "title": movie["title"],
            "movie_id": int(movie["movie_id"]),
            "predicted_rating": round(score, 2),
            "poster": poster_url if poster_url else "Poster unavailable",
        })
    logger.info(f"Content-based fallback recommendations for user {user_id}: {recommendations}")
    return recommendations

def get_collaborative_recommendations(user_id, n=10):
    try:
        # Check if the user has rated movies in ratings_df
        user_ratings = ratings_df[ratings_df["userId"] == user_id]
        if len(user_ratings) >= 5:  # Require at least 5 ratings for reliable collaborative filtering
            logger.info(f"User {user_id} has sufficient ratings in ratings_df: {len(user_ratings)} ratings")
            # Use SVD model for recommendations
            all_movie_ids = ratings_df["movieId"].unique()
            rated_movie_ids = set(user_ratings["movieId"].values)
            predictions = []
            for movie_id in all_movie_ids:
                if movie_id not in rated_movie_ids:
                    predicted_rating = collab_model.predict(user_id, movie_id).est
                    predictions.append((movie_id, predicted_rating))
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_movies = predictions[:n]
        else:
            logger.info(f"User {user_id} has insufficient ratings in ratings_df, checking user preferences")
            # Check user preferences for ratings
            if user_id in user_preferences and "ratings" in user_preferences[user_id] and len(user_preferences[user_id]["ratings"]) >= 5:
                user_prefs = user_preferences[user_id]
                user_ratings = user_prefs["ratings"]
                user_genres = [genre.lower() for genre in user_prefs.get("genres", [])]
                user_actors = [actor.lower() for actor in user_prefs.get("actors", [])]
                user_directors = [director.lower() for director in user_prefs.get("directors", [])]
                logger.info(f"User {user_id} ratings from preferences: {user_ratings}")
                logger.info(f"User {user_id} preferred genres: {user_genres}")
                logger.info(f"User {user_id} preferred actors: {user_actors}")
                logger.info(f"User {user_id} preferred directors: {user_directors}")
                
                # Create a dictionary of user-provided ratings
                user_ratings_dict = {rating["movieId"]: rating["rating"] for rating in user_ratings}
                rated_movie_ids = set(user_ratings_dict.keys())
                
                # Predict ratings for all movies
                all_movie_ids = ratings_df["movieId"].unique()
                predictions = []
                
                # Use user-provided ratings for movies the user rated
                for movie_id in rated_movie_ids:
                    if movie_id in all_movie_ids:
                        user_rating = user_ratings_dict[movie_id]
                        predictions.append((movie_id, user_rating))
                        logger.info(f"Using user-provided rating for user {user_id}, movie {movie_id}: {user_rating}")
                
                # Predict ratings for movies the user hasn't rated
                for movie_id in all_movie_ids:
                    if movie_id not in rated_movie_ids:
                        predicted_rating = collab_model.predict(user_id, movie_id).est
                        # Boost the rating based on user preferences
                        tmdb_id = map_movielens_to_tmdb(movie_id)
                        if tmdb_id:
                            movie_row = movies[movies["movie_id"] == tmdb_id]
                            if not movie_row.empty:
                                movie_genres = [genre.lower() for genre in movie_row["genres"].iloc[0]]
                                movie_actors = [actor.lower() for actor in movie_row.get("cast", [])]
                                movie_directors = [director.lower() for director in movie_row.get("directors", [])]

                                # Boost based on genre overlap
                                genre_overlap = sum(1 for genre in user_genres if genre in movie_genres)
                                predicted_rating += genre_overlap * 1.0  # Increased boost to 1.0 per matching genre

                                # Boost based on actor overlap
                                actor_overlap = sum(1 for actor in user_actors if actor in movie_actors)
                                predicted_rating += actor_overlap * 0.5  # Increased boost to 0.5 per matching actor

                                # Boost based on director overlap
                                director_overlap = sum(1 for director in user_directors if director in movie_directors)
                                predicted_rating += director_overlap * 0.5  # Increased boost to 0.5 per matching director

                                if genre_overlap > 0 or actor_overlap > 0 or director_overlap > 0:
                                    logger.info(f"Boosted rating for movie {movie_id} (genres: {movie_genres}, actors: {movie_actors}, directors: {movie_directors}) to {predicted_rating}")
                        predictions.append((movie_id, predicted_rating))
                
                # Sort predictions by rating
                predictions.sort(key=lambda x: x[1], reverse=True)
                top_movies = predictions[:n]
            else:
                logger.info(f"Insufficient ratings for user {user_id}, falling back to content-based recommendations")
                return get_content_based_fallback(user_id, n)

        recommendations = []
        for movie_id, rating in top_movies:
            tmdb_id = map_movielens_to_tmdb(movie_id)
            if tmdb_id:
                movie_row = movies[movies["movie_id"] == tmdb_id]
                if not movie_row.empty:
                    movie_title = movie_row["title"].values[0]
                    poster_url = fetch_poster(tmdb_id)
                    recommendations.append({
                        "title": movie_title,
                        "movie_id": int(tmdb_id),
                        "predicted_rating": round(rating, 2),
                        "poster": poster_url if poster_url else "Poster unavailable",
                    })
        logger.info(f"Final collaborative recommendations for user {user_id}: {recommendations}")
        return recommendations
    except Exception as e:
        logger.error(f"Error in collaborative filtering: {e}")
        return get_content_based_fallback(user_id, n)

@app.route("/api/user/preferences", methods=["POST"])
def save_user_preferences():
    """Save user preferences (genres, actors, directors, ratings) for collaborative filtering."""
    try:
        data = request.json
        user_id = data.get("userId")
        if not user_id:
            return jsonify({"error": "userId is required"}), 400

        user_preferences[user_id] = {
            "genres": data.get("genres", []),
            "actors": data.get("actors", []) if data.get("actors") else [],
            "directors": data.get("directors", []) if data.get("directors") else [],
            "ratings": data.get("ratings", [])
        }
        logger.info(f"Saved preferences for user {user_id}: {user_preferences[user_id]}")
        return jsonify({"message": "Preferences saved successfully"})
    except Exception as e:
        logger.error(f"Error saving user preferences: {e}")
        return jsonify({"error": "Internal server error"}), 500
# Chatbot Functions
def extract_genre(message):
    """Extract genres from the user message."""
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

def get_movies_by_genre(genre, movies_df, count=5):
    """Get movies based on a specific genre."""
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
                    "poster": poster_url if poster_url else "Poster unavailable",
                })
            return recommendations
        return []
    except Exception as e:
        logger.error(f"Error getting movies by genre: {e}")
        return []

def get_recommendations_from_input(message, movies_df, cosine_sim, count=5):
    """Generate recommendations based on user input."""
    try:
        # Check for genre recommendations
        genres = extract_genre(message)
        if genres:
            return get_movies_by_genre(genres[0], movies_df, count)
        
        # Default to random recommendations if no specific intent
        random_movies = movies_df.sample(count)
        recommendations = []
        for _, movie in random_movies.iterrows():
            poster_url = fetch_poster(movie['movie_id'])
            recommendations.append({
                'title': movie['title'],
                'movie_id': int(movie['movie_id']),
                'poster': poster_url if poster_url else "Poster unavailable",
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
    """Extract entities (e.g., movie titles) from the user message."""
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
                # Convert TMDB ID to movie_id from your dataset if needed
                movie_row = movies[movies['movie_id'] == validated_movie]
                if movie_row.empty:
                    continue
                entities["MOVIE"].append(validated_movie)

    if len(entities["MOVIE"]) == 1:
        entities["MOVIE"] = entities["MOVIE"][0]

    return entities

def get_movie_details(movie_id_or_title):
    """Get details for a specific movie based on ID or title."""
    try:
        if isinstance(movie_id_or_title, (int, str)) and str(movie_id_or_title).isdigit():
            movie = movies[movies['movie_id'] == int(movie_id_or_title)]
            if movie.empty:
                return f"Sorry, I couldn't find details for the movie with ID {movie_id_or_title}.", None
        else:
            movie = movies[movies['title'].str.lower() == movie_id_or_title.lower()]
            if movie.empty:
                return f"Sorry, I couldn't find details for '{movie_id_or_title}'.", None

        movie_data = movie.iloc[0].to_dict()
        overview = movie_data.get('overview', "No overview available.")
        poster_url = fetch_poster(movie_data['movie_id'])
        return (
            f"Here's what I know about '{movie_data['title']}': {overview}",
            {
                "title": movie_data["title"],
                "movie_id": int(movie_data["movie_id"]),
                "overview": overview,
                "poster": poster_url if poster_url else "Poster unavailable",
            },
        )
    except Exception as e:
        logger.error(f"Error getting movie details: {e}")
        return f"Error retrieving movie details.", None

def format_recommendations(recommendations):
    """Format the recommendation response."""
    if recommendations:
        movie_titles = [movie["title"] for movie in recommendations]
        return f"Here are some movies you might enjoy: {', '.join(movie_titles)}"
    return "I couldn't find any recommendations."

def format_genre_movies(genre_movies, genre):
    """Format the genre-based movie response."""
    if genre_movies:
        movie_titles = [movie["title"] for movie in genre_movies]
        return f"Here are some {genre} movies: {', '.join(movie_titles)}"
    return f"No movies found for genre '{genre}'."

def format_similar_movies(similar_movies, movie_title):
    """Format the similar movies response."""
    if similar_movies:
        movie_titles = [movie["title"] for movie in similar_movies]
        return f"If you liked {movie_title}, you might also enjoy: {', '.join(movie_titles)}"
    return f"Couldn't find similar movies for '{movie_title}'."
    
@app.route("/api/login", methods=["POST"])
def login():
    """Authenticate a user and update their last_login in MongoDB."""
    try:
        data = request.json
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"error": "Username and password are required"}), 400

        # Find the user in MongoDB
        user = users_collection.find_one({"username": username, "password": password})
        if not user:
            return jsonify({"error": "Invalid username or password"}), 401

        # Update last_login timestamp
        users_collection.update_one(
            {"username": username},
            {"$set": {"last_login": datetime.utcnow().isoformat()}}
        )

        # Prepare user data to return (excluding password)
        user_data = {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"],
            "last_login": user["last_login"]
        }

        logger.info(f"User logged in: username={username}, role={user['role']}")
        return jsonify({"message": "Login successful", "user": user_data})
    except Exception as e:
        logger.error(f"Error during login: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/users", methods=["GET"])
def get_users():
    """Fetch all users from MongoDB for the admin dashboard."""
    try:
        users = list(users_collection.find({}, {"_id": 0}))  # Exclude MongoDB's _id field
        return jsonify(users)
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        return jsonify({"error": "Internal server error"}), 500
    
@app.route('/api/user/<int:user_id>', methods=['GET'])
def get_user(user_id):
    try:
        user = users_collection.find_one({"id": user_id}, {"_id": 0})  # Match by id field
        if user:
            return jsonify(user)
        else:
            abort(404, description="Resource not found")
    except Exception as e:
        logger.error(f"Error fetching user {user_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500
    
    
@app.route("/api/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    """Delete a user by their ID."""
    try:
        result = users_collection.delete_one({"id": user_id})
        if result.deleted_count == 0:
            return jsonify({"error": "User not found"}), 404
        logger.info(f"Deleted user with ID {user_id}")
        return jsonify({"message": "User deleted successfully"})
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        return jsonify({"error": "Internal server error"}), 500
    
@app.route("/api/register_user", methods=["POST"])
def register_user():
    """Register a new user and save to MongoDB with default values for missing fields."""
    try:
        data = request.json
        logger.info(f"Received registration request: {data}")

        # Get the highest user ID in the database and increment it for a new user
        highest_user = users_collection.find_one(sort=[("id", -1)])  # Sort by id descending
        logger.info(f"Highest user: {highest_user}")
        # Safely get the id; default to 0 if not present
        highest_id = highest_user.get("id", 0) if highest_user else 0
        next_user_id = highest_id + 1  # Increment the highest id
        logger.info(f"Next user ID: {next_user_id}")

        # Set default values if fields are missing
        user_id = int(data.get("userId", next_user_id))  # Use next_user_id if userId is not provided
        username = data.get("username", f"user_{user_id}")
        password = data.get("password", "default_password_123")
        role = data.get("role", "user")

        logger.info(f"User data: userId={user_id}, username={username}, role={role}")

        # Validate critical fields
        if not username or username == f"user_{user_id}":
            logger.warning("Validation failed: Username is required and cannot be the default value")
            return jsonify({"error": "Username is required and cannot be the default value"}), 400
        if not password or password == "default_password_123":
            logger.warning("Validation failed: Password is required and cannot be the default value")
            return jsonify({"error": "Password is required and cannot be the default value"}), 400

        # Check if the user already exists
        existing_user = users_collection.find_one({"username": username})
        if existing_user:
            logger.warning(f"Username already exists: {username}")
            return jsonify({"error": "Username already exists"}), 400

        # Create new user document
        new_user = {
            "id": user_id,
            "username": username,
            "password": password,  # In production, hash the password!
            "role": role,
            "last_login": None
        }

        # Insert into MongoDB
        users_collection.insert_one(new_user)
        logger.info(f"Registered new user: userId={user_id}, username={username}, role={role}")
        return jsonify({"message": "User registered successfully"})
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500
    
# app.py
@app.route("/api/clear_preferences/<int:user_id>", methods=["DELETE"])
def clear_preferences(user_id):
    if user_id in user_preferences:
        del user_preferences[user_id]
        logger.info(f"Cleared preferences for user {user_id}")
        return jsonify({"message": "Preferences cleared"})
    return jsonify({"error": "User not found"}), 404


@app.route("/api/user/preferences/<int:user_id>", methods=["GET"])
def get_user_preferences(user_id):
    """Retrieve user preferences for a given user."""
    try:
        if user_id in user_preferences:
            return jsonify({"user_id": user_id, "preferences": user_preferences[user_id]})
        else:
            return jsonify({"error": "User preferences not found"}), 404
    except Exception as e:
        logger.error(f"Error retrieving user preferences: {e}")
        return jsonify({"error": "Internal server error"}), 500

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
            
        movies_cleaned = movies.copy()
        movies_cleaned['overview'] = movies_cleaned['overview'].fillna('No overview available')
        movies_cleaned['tags'] = movies_cleaned['tags'].fillna('')
        movies_cleaned = movies_cleaned.dropna(subset=['title'])

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        total_movies = len(movies_cleaned)
        
        if start_idx >= total_movies:
            return jsonify({"error": "Page number exceeds available data."}), 404

        paginated_movies = movies_cleaned.iloc[start_idx:end_idx]

        movies_with_posters = []
        for _, movie in paginated_movies.iterrows():
            movie_data = movie.to_dict()
            poster_url = fetch_poster(movie['movie_id'])
            movie_data['poster'] = poster_url if poster_url else "Poster unavailable"
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
        movie_row = movies[movies['movie_id'] == movie_id]
        
        if movie_row.empty:
            return jsonify({"error": "Movie not found"}), 404
        
        movie_title = movie_row['title'].iloc[0]
        
        recommended_titles = get_recommendations(movie_title, cosine_sim, movies)
        
        recommended_titles = recommended_titles.tolist() if isinstance(recommended_titles, pd.Series) else recommended_titles
        if not recommended_titles:
            return jsonify({"error": "Could not generate recommendations"}), 404

        recommended_movies = []
        for title in recommended_titles:
            movie_data = movies[movies['title'] == title].iloc[0].to_dict()
            poster_url = fetch_poster(movie_data['movie_id'])
            movie_data['poster'] = poster_url if poster_url else "Poster unavailable"
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
        movie = movies[movies['movie_id'] == movie_id]
        
        if movie.empty:
            return jsonify({"error": "Movie not found"}), 404
        
        movie_data = movie.iloc[0].to_dict()
        
        poster_url = fetch_poster(movie_data['movie_id'])
        movie_data['poster'] = poster_url if poster_url else "Poster unavailable"
        
        return jsonify(movie_data)
        
    except Exception as e:
        logger.error(f"Error processing movie request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/hybrid_recommendations/<int:user_id>/<int:movie_id>', methods=['GET'])
def hybrid_recommend(user_id, movie_id):
    """API that combines Content-Based and Collaborative Filtering recommendations."""
    
    collab_recommendations = get_collaborative_recommendations(user_id, n=10)

    try:
        movie_row = movies[movies['movie_id'] == movie_id]
        if movie_row.empty:
            return jsonify({"error": f"Movie ID {movie_id} not found in content-based dataset"}), 404

        movie_title = movie_row['title'].iloc[0]

        recommended_titles = get_recommendations(movie_title, cosine_sim, movies)[:10]

        content_recommendations = []
        for title in recommended_titles:
            movie_data = movies[movies['title'] == title].iloc[0].to_dict()
            poster_url = fetch_poster(movie_data['movie_id'])
            content_recommendations.append({
                "title": movie_data['title'],
                "movie_id": int(movie_data['movie_id']),
                "poster": poster_url if poster_url else "Poster unavailable",
                "genres": movie_data.get('genres', []),
                "overview": movie_data.get('overview', "No overview available"),
            })

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

@app.route('/api/initial_recommendations', methods=['GET'])
def initial_recommendations():
    """API endpoint for initial recommendations (popular movies)."""
    try:
        recommendations = get_popular_movies(n=20)
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        logger.error(f"Error fetching initial recommendations: {e}")
        return jsonify({"error": "Could not fetch initial recommendations"}), 500
    
@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chatbot interactions and return responses."""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # Detect user intent
        action = detect_intent(user_message)
        
        # Extract movie title and genre from user input
        entities = extract_entities(user_message)
        movie_id_or_title = entities.get("MOVIE")
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

        elif action == 'get_movie_details' and movie_id_or_title:
            response_text, action_result = get_movie_details(movie_id_or_title)
        
        elif action == 'find_similar' and movie_id_or_title:
            similar_movies = get_recommendations_from_input(f"like {movie_id_or_title}", movies, cosine_sim)
            response_text = format_similar_movies(similar_movies, movie_id_or_title)
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