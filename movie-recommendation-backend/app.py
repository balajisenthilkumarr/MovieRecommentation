# import requests
# from flask import Flask, jsonify, request
# from model import main, get_recommendations
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# # Load data and compute similarity (from model.py)
# movies, cosine_sim = main()

# import time
# import requests
# from requests.exceptions import ConnectionError

# TMDB_API_KEY = '5bd7d31b6e1466d5799253aa07b28a02'

# def fetch_poster(movie_id):
#     # api_key = '5bd7d31b6e1466d5799253aa07b28a02'  # Replace with your TMDB API key
#     url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}'
#     response = requests.get(url)
#     data = response.json()
#     poster_path = data['poster_path']
#     full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
#     return full_path


# @app.route("/api/movies", methods=["GET"])
# def get_movies():
#     try:
#         page = int(request.args.get("page", 1))  # Default to page 1 if not provided
#         page_size = int(request.args.get("page_size", 10))  # Default to 10 items per page
#     except ValueError:
#         return jsonify({"error": "Invalid page or page_size parameters."}), 400

#     # Replace NaN in 'overview' with a default string (or empty string)
#     movies_cleaned = movies.copy()

#     # Replace NaN values in 'overview' with a default message
#     movies_cleaned['overview'] = movies_cleaned['overview'].fillna('No overview available')
#     movies_cleaned['tags'] = movies_cleaned['tags'].fillna('')
    
#     # Drop any rows where the 'title' is missing (essential field)
#     movies_cleaned = movies_cleaned.dropna(subset=['title'])

#     # Calculate the start and end index based on the page number and page size
#     start_idx = (page - 1) * page_size
#     end_idx = start_idx + page_size

#     # Slice the DataFrame to get the movies for the current page
#     paginated_movies = movies_cleaned.iloc[start_idx:end_idx]

#     # Add poster to each movie
#     movies_with_posters = []
#     for _, movie in paginated_movies.iterrows():
#         movie_data = movie.to_dict()
#         movie_data['poster'] = fetch_poster(movie['movie_id'])  # Fetch poster image using movie_id
#         movies_with_posters.append(movie_data)

#     # Convert the movies list with posters to JSON
#     return jsonify(movies_with_posters)


# @app.route('/api/recommendations/<movie_id>', methods=['GET'])
# def recommendations(movie_id):
#     """
#     Endpoint to get movie recommendations based on a given movie ID.
#     Returns a list of recommended movies.
#     """
#     # Find the movie by its movie_id
#     movie = movies[movies['movie_id'] == int(movie_id)].iloc[0]
#     title = movie['title']
    
#     # Get the recommendations using the model's function
#     recommended_titles = get_recommendations(title, cosine_sim, movies)
    
#     # Return the recommended movies as a JSON response
#     recommended_movies = movies[movies['title'].isin(recommended_titles)][['movie_id', 'title', 'overview']].to_dict(orient='records')
    
#     # Add poster to each recommended movie
#     for recommended_movie in recommended_movies:
#         recommended_movie['poster'] = fetch_poster(recommended_movie['movie_id'])
    
#     return jsonify(recommended_movies)

# if __name__ == '__main__':
#     app.run(debug=True)












# import requests
# from flask import Flask, jsonify, request
# from model import main, get_recommendations
# from flask_cors import CORS
# from requests.adapters import HTTPAdapter
# from requests.packages.urllib3.util.retry import Retry
# import logging
# from functools import lru_cache

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)
# CORS(app)

# # Load data and compute similarity (from model.py)
# try:
#     movies, cosine_sim = main()
# except Exception as e:
#     logger.error(f"Failed to load movie data: {e}")
#     raise

# TMDB_API_KEY = '5bd7d31b6e1466d5799253aa07b28a02'

# # Configure session with retry strategy
# def create_session():
#     session = requests.Session()
#     retries = Retry(
#         total=5,
#         backoff_factor=0.1,
#         status_forcelist=[500, 502, 503, 504],
#         allowed_methods=["GET"]
#     )
#     session.mount('https://', HTTPAdapter(max_retries=retries))
#     return session

# # Create a session to be reused
# session = create_session()

# # Cache for storing poster URLs
# poster_cache = {}

# @lru_cache(maxsize=1000)
# def fetch_poster(movie_id):
#     """Fetch movie poster with caching and error handling."""
#     if movie_id in poster_cache:
#         return poster_cache[movie_id]
    
#     try:
#         url = f'https://api.themoviedb.org/3/movie/{movie_id}'
#         params = {'api_key': TMDB_API_KEY}
        
#         response = session.get(url, params=params, timeout=10)
#         response.raise_for_status()
        
#         data = response.json()
#         if 'poster_path' in data and data['poster_path']:
#             poster_path = data['poster_path']
#             full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
#             poster_cache[movie_id] = full_path
#             return full_path
#         return None
#     except requests.exceptions.RequestException as e:
#         logger.error(f"Error fetching poster for movie {movie_id}: {e}")
#         return None

# @app.route("/api/movies", methods=["GET"])
# def get_movies():
#     try:
#         page = int(request.args.get("page", 1))
#         page_size = int(request.args.get("page_size", 10))
        
#         if page < 1 or page_size < 1:
#             return jsonify({"error": "Page and page_size must be positive integers."}), 400
            
#     except ValueError:
#         return jsonify({"error": "Invalid page or page_size parameters."}), 400

#     try:
#         # Create a copy and clean the data
#         movies_cleaned = movies.copy()
#         movies_cleaned['overview'] = movies_cleaned['overview'].fillna('No overview available')
#         movies_cleaned['tags'] = movies_cleaned['tags'].fillna('')
#         movies_cleaned = movies_cleaned.dropna(subset=['title'])

#         # Pagination
#         start_idx = (page - 1) * page_size
#         end_idx = start_idx + page_size
#         total_movies = len(movies_cleaned)
        
#         if start_idx >= total_movies:
#             return jsonify({"error": "Page number exceeds available data."}), 404

#         paginated_movies = movies_cleaned.iloc[start_idx:end_idx]

#         # Add posters to movies
#         movies_with_posters = []
#         for _, movie in paginated_movies.iterrows():
#             movie_data = movie.to_dict()
#             poster_url = fetch_poster(movie['movie_id'])
#             movie_data['poster'] = poster_url if poster_url else "https://via.placeholder.com/500x750?text=No+Poster"
#             movies_with_posters.append(movie_data)

#         response_data = {
#             "movies": movies_with_posters,
#             "pagination": {
#                 "page": page,
#                 "page_size": page_size,
#                 "total_movies": total_movies,
#                 "total_pages": (total_movies + page_size - 1) // page_size
#             }
#         }

#         return jsonify(response_data)

#     except Exception as e:
#         logger.error(f"Error processing movies request: {e}")
#         return jsonify({"error": "Internal server error"}), 500

# @app.route('/api/recommendations/<movie_id>', methods=['GET'])
# def recommendations(movie_id):
#     try:
#         movie_id = int(movie_id)
#         movie = movies[movies['movie_id'] == movie_id]
        
#         if movie.empty:
#             return jsonify({"error": "Movie not found"}), 404
            
#         title = movie.iloc[0]['title']
#         recommended_titles = get_recommendations(title, cosine_sim, movies)
        
#         if not recommended_titles:
#             return jsonify({"error": "Could not generate recommendations"}), 404

#         recommended_movies = []
#         for title in recommended_titles:
#             movie_data = movies[movies['title'] == title].iloc[0].to_dict()
#             poster_url = fetch_poster(movie_data['movie_id'])
#             movie_data['poster'] = poster_url if poster_url else "https://via.placeholder.com/500x750?text=No+Poster"
#             recommended_movies.append(movie_data)

#         return jsonify({
#             "movie_id": movie_id,
#             "movie_title": title,
#             "recommendations": recommended_movies
#         })

#     except ValueError:
#         return jsonify({"error": "Invalid movie ID format"}), 400
#     except Exception as e:
#         logger.error(f"Error processing recommendation request: {e}")
#         return jsonify({"error": "Internal server error"}), 500

# @app.errorhandler(404)
# def not_found(error):
#     return jsonify({"error": "Resource not found"}), 404

# @app.errorhandler(500)
# def internal_error(error):
#     return jsonify({"error": "Internal server error"}), 500

# if __name__ == '__main__':
#     app.run(debug=True)



import requests
from flask import Flask, jsonify, request
from model import main, get_recommendations
from flask_cors import CORS
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load data and compute similarity (from model.py)
try:
    movies, cosine_sim = main()
except Exception as e:
    logger.error(f"Failed to load movie data: {e}")
    raise

TMDB_API_KEY = '5bd7d31b6e1466d5799253aa07b28a02'

# Configure session with retry strategy
def create_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

session = create_session()
poster_cache = {}

@lru_cache(maxsize=1000)
def fetch_poster(movie_id):
    """Fetch movie poster with caching and error handling."""
    if movie_id in poster_cache:
        return poster_cache[movie_id]
    
    try:
        url = f'https://api.themoviedb.org/3/movie/{movie_id}'
        params = {'api_key': TMDB_API_KEY}
        
        response = session.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if 'poster_path' in data and data['poster_path']:
            poster_path = data['poster_path']
            full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
            poster_cache[movie_id] = full_path
            return full_path
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching poster for movie {movie_id}: {e}")
        return None

@app.route("/api/movies", methods=["GET"])
def get_movies():
    try:
        page = int(request.args.get("page", 1))
        page_size = int(request.args.get("page_size", 10))
        
        if page < 1 or page_size < 1:
            return jsonify({"error": "Page and page_size must be positive integers."}), 400
            
    except ValueError:
        return jsonify({"error": "Invalid page or page_size parameters."}), 400

    try:
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
            movie_data['poster'] = poster_url if poster_url else "https://via.placeholder.com/500x750?text=No+Poster"
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

    except Exception as e:
        logger.error(f"Error processing movies request: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/recommendations/<movie_id>', methods=['GET'])
def recommendations(movie_id):
    try:
        # Convert movie_id to integer
        movie_id = int(movie_id)
        
        # Find the movie in the dataframe
        movie_row = movies[movies['movie_id'] == movie_id]
        
        if movie_row.empty:
            return jsonify({"error": "Movie not found"}), 404
        
        # Get the title of the movie
        movie_title = movie_row['title'].iloc[0]
        
        # Get recommendations using your existing model function
        recommended_titles = get_recommendations(movie_title, cosine_sim, movies)
        
        if recommended_titles.empty:
            return jsonify({"error": "Could not generate recommendations"}), 404

        # Convert recommended titles to full movie data
        recommended_movies = []
        for title in recommended_titles:
            movie_data = movies[movies['title'] == title].iloc[0].to_dict()
            
            # Add poster URL
            poster_url = fetch_poster(movie_data['movie_id'])
            movie_data['poster'] = poster_url if poster_url else "https://via.placeholder.com/500x750?text=No+Poster"
            
            recommended_movies.append(movie_data)
           # print("dddddddddd",recommended_movies)

        response_data = {
            "movie_id": movie_id,
            "movie_title": movie_title,
            "recommendations": recommended_movies
        }

        #print(recommended_movies)

        return jsonify(response_data)

    except ValueError as e:
        logger.error(f"Invalid movie ID format: {movie_id}")
        return jsonify({"error": "Invalid movie ID format"}), 400
    except Exception as e:
        logger.error(f"Error processing recommendation request: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/movie/<movie_id>', methods=['GET'])
def get_movie(movie_id):
    try:
        # Convert movie_id to integer
        movie_id = int(movie_id)
        
        # Find the movie in the dataframe
        movie = movies[movies['movie_id'] == movie_id]
        
        if movie.empty:
            return jsonify({"error": "Movie not found"}), 404
        
        # Convert movie data to dict
        movie_data = movie.iloc[0].to_dict()
        
        # Add poster URL
        poster_url = fetch_poster(movie_data['movie_id'])
        movie_data['poster'] = poster_url if poster_url else "https://via.placeholder.com/500x750?text=No+Poster"
        
        return jsonify(movie_data)
        
    except ValueError as e:
        logger.error(f"Invalid movie ID format: {movie_id}")
        return jsonify({"error": "Invalid movie ID format"}), 400
    except Exception as e:
        logger.error(f"Error processing movie request: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500



if __name__ == '__main__':
    app.run(debug=True)