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
from flask import Flask, jsonify, request
import random
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import requests
import logging
from transformers import pipeline
import spacy
from sentence_transformers import SentenceTransformer
from chatbaot import search_movie_in_tmdb



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load necessary NLTK data
# Download necessary NLTK resources (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    
# Load NLP models
# nlp = spacy.load("en_core_web_sm")  # For entity extraction
classifier = pipeline("zero-shot-classification")  # For intent detection
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # For semantic similarity
from transformers import pipeline

ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

text = "Show me some movies like Inception and Titanic"
entities = ner_model(text)
print(entities,"jkjksjs")

    
    
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')  # Ensures WordNet is fully available



# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

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


# Extract genres from message
def extract_genre(message):
    genres = ["action", "adventure", "animation", "comedy", "crime", "documentary", 
              "drama", "family", "fantasy", "history", "horror", "music", "mystery",
              "romance", "science fiction", "sci-fi", "thriller", "war", "western"]
    
    found_genres = []
    message_lower = message.lower()
    
    for genre in genres:
        if genre in message_lower:
            found_genres.append(genre)
    
    return found_genres

# Extract actor names from message
def extract_actor(message):
    # This is a simple implementation; a more sophisticated NER would be better
    words = message.split()
    potential_names = []
    
    for i in range(len(words)-1):
        if words[i][0].isupper() and words[i+1][0].isupper():
            potential_names.append(f"{words[i]} {words[i+1]}")
    
    return potential_names

# Extract movie titles from message
def extract_movie_title(message, movie_titles):
    for title in movie_titles:
        if title.lower() in message.lower():
            return title
    return None



# Function to get movies by genre
def get_movies_by_genre(genre, movies_df, count=5):
    try:
        # Convert genre to lowercase for case-insensitive matching
        genre = genre.lower()
        
        # Filter movies that contain the genre in their genres list
        filtered_movies = []
        
        for _, movie in movies_df.iterrows():
            # Check if genres is a list-like object and contains strings
            if isinstance(movie['genres'], list):
                movie_genres = [g.lower() if isinstance(g, str) else '' for g in movie['genres']]
                if genre in movie_genres:
                    filtered_movies.append(movie)
            # If genres is a string (like a JSON string), check if genre is in it
            elif isinstance(movie['genres'], str):
                if genre.lower() in movie['genres'].lower():
                    filtered_movies.append(movie)
        
        # Sort by popularity if available, otherwise return random selection
        if filtered_movies:
            import random
            selected_movies = random.sample(filtered_movies, min(count, len(filtered_movies)))
            return [{'title': movie['title'], 'movie_id': int(movie['movie_id'])} for movie in selected_movies]
        
        return []
    except Exception as e:
        print(f"Error getting movies by genre: {e}")
        return []


# Function to recommend movies based on input
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
                recommended_movies.append({
                    'title': movie['title'],
                    'movie_id': int(movie['movie_id']),  
                })
            
            return recommended_movies
        
        # Default to random recommendations
        random_movies = movies_df.sample(count)
        return [{'title': movie['title'], 'movie_id': movie['movie_id']} 
                for _, movie in random_movies.iterrows()]
                
    except Exception as e:
        print(f"Error getting recommendations from input: {e}")
        return []



def detect_intent(user_message):
    labels = {
        "recommend_movies": "User wants movie recommendations",
        "find_similar": "User is asking for movies similar to another movie",
        "get_movie_details": "User wants information about a specific movie",
        "extract_genre": "User wants movie recommendations based on genre"
    }
    
    result = classifier(user_message, list(labels.values()))
    best_match = result["labels"][0]
    
    # Convert back to intent label
    for key, value in labels.items():
        if value == best_match:
            return key
    
    return "unknown"

from rapidfuzz import process, fuzz

# Load movie dataset
MOVIE_TITLES = movies['title'].tolist()  # Ensure movie titles are in a list
from rapidfuzz import process, fuzz

def extract_entities(user_message):
    """Extract movie names from user input using NER and validate with TMDB API, 
       with fuzzy matching to handle typos.
    """
    entities = {"MOVIE": []}
    results = ner_model(user_message)

    current_entity = ""
    movie_candidates = []

    # 🔹 Extract movie names using NER
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

    # 🔹 Debugging Output: Print extracted words before validation
    print("Extracted Raw Movie Names:", movie_candidates)

    # 🔹 Validate and correct movie names using fuzzy matching + TMDB API
    for movie in movie_candidates:
        fuzzy_result = process.extractOne(movie.lower(), [title.lower() for title in MOVIE_TITLES], scorer=fuzz.ratio, score_cutoff=75)

        if fuzzy_result:
            match = fuzzy_result[0]  # Extract only the matched title
            validated_movie = search_movie_in_tmdb(match) if match else None

            if validated_movie:
                entities["MOVIE"].append(validated_movie)
            else:
                print(f"❌ No valid match found for: {movie}")

    # Convert to string if only one movie found
    if len(entities["MOVIE"]) == 1:
        entities["MOVIE"] = entities["MOVIE"][0]

    return entities







import numpy as np

def safe_json_response(data):
    if isinstance(data, np.int64):  # Convert int64 to Python int
        return int(data)
    elif isinstance(data, dict):  # Convert values inside dictionary
        return {k: safe_json_response(v) for k, v in data.items()}
    elif isinstance(data, list):  # Convert values inside list
        return [safe_json_response(i) for i in data]
    return data



from flask import Flask, request, jsonify
import logging

def format_recommendations(recommendations):
    if recommendations:
        movie_titles = [movie['title'] for movie in recommendations]
        return f"Here are some movies you might enjoy: {', '.join(movie_titles)}"
    return "I couldn't find any recommendations."

def format_genre_movies(genre_movies, genre):
    if genre_movies:
        movie_titles = [movie['title'] for movie in genre_movies]
        return f"Here are some {genre} movies: {', '.join(movie_titles)}"
    return f"No movies found for genre '{genre}'."

def format_similar_movies(similar_movies, movie_title):
    if similar_movies:
        movie_titles = [movie['title'] for movie in similar_movies]
        return f"If you liked {movie_title}, you might also enjoy: {', '.join(movie_titles)}"
    return f"Couldn't find similar movies for '{movie_title}'."

def get_movie_details(movie_title):
    try:
        movie = movies[movies['title'].str.lower() == movie_title.lower()].iloc[0]
        overview = movie['overview'] if 'overview' in movie and movie['overview'] else "No overview available."
        return f"Here's what I know about '{movie_title}': {overview}", {
            'title': movie_title,
            'movie_id': int(movie['movie_id']),
            'overview': overview
        }
    except IndexError:
        return f"Sorry, I couldn't find details for '{movie_title}'.", None


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        print(f"User Query: {user_message}") 

        # Detect user intent
        action = detect_intent(user_message)
        print(f"Detected Intent: {action}")

        # Extract movie title and genre from user input
        entities = extract_entities(user_message)
        print(f"Extracted Entities: {entities}")

        movie_title = entities.get("MOVIE")
        genre = entities.get("GENRE")

        # Handle different chatbot actions
        if action == 'recommend_movies':
            recommendations = get_recommendations_from_input(user_message, movies, cosine_sim)
            response_text = format_recommendations(recommendations)  # Use helper function
            action_result = recommendations

        elif action == 'extract_genre' and genre:
            genre_movies = get_movies_by_genre(genre, movies)
            response_text = format_genre_movies(genre_movies, genre)  # Use helper function
            action_result = genre_movies

        elif action == 'get_movie_details' and movie_title:
            response_text, action_result = get_movie_details(movie_title)
        
        elif action == 'find_similar' and movie_title:
            similar_movies = get_recommendations_from_input(f"like {movie_title}", movies, cosine_sim)
            response_text = format_similar_movies(similar_movies, movie_title)  # Use helper function
            action_result = similar_movies


        return jsonify({"response": response_text, "action": action, "result": action_result})

    except Exception as e:
        logging.error(f"Error processing chat request: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)