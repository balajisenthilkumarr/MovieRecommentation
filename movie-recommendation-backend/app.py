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
import os
from model import main, get_recommendations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

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

def get_popular_movies(n=10):
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


@app.route("/api/register_user", methods=["POST"])
def register_user():
    try:
        data = request.json
        user_id = data.get("userId")
        username = data.get("username")
        if not user_id or not username:
            return jsonify({"error": "userId and username are required"}), 400
        logger.info(f"Registered new user: userId={user_id}, username={username}")
        return jsonify({"message": "User registered successfully"})
    except Exception as e:
        logger.error(f"Error registering user: {e}")
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
    
    collab_recommendations = get_collaborative_recommendations(user_id, n=5)

    try:
        movie_row = movies[movies['movie_id'] == movie_id]
        if movie_row.empty:
            return jsonify({"error": f"Movie ID {movie_id} not found in content-based dataset"}), 404

        movie_title = movie_row['title'].iloc[0]

        recommended_titles = get_recommendations(movie_title, cosine_sim, movies)[:5]

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
        recommendations = get_popular_movies(n=10)
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        logger.error(f"Error fetching initial recommendations: {e}")
        return jsonify({"error": "Could not fetch initial recommendations"}), 500

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