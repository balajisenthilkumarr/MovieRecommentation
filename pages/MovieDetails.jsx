// src/pages/MovieDetails.jsx
import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import LoadingSpinner from "../components/LoadingSpinner";
import MovieGrid from "../components/MovieGrid";
import { ArrowLeft, Star, Calendar, Globe } from "lucide-react";
import axios from "axios";



const MovieDetails = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [movie, setMovie] = useState(null);
  const [hybridResponse, setHybridResponse] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const userId = 1; // Replace with actual user ID from auth (hardcoded for now)

  const isValidUrl = (url) => {
    try {
      new URL(url);
      return true;
    } catch (e) {
      return false;
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch movie details
        const movieResponse = await axios.get(`http://localhost:5000/api/movie/${id}`);
        console.log("Movie response:", movieResponse.data);
        setMovie(movieResponse.data);

        // Fetch hybrid recommendations
        const hybridResponse = await axios.get(`http://localhost:5000/api/hybrid_recommendations/${userId}/${id}`);
        console.log("Hybrid recommendations response:", hybridResponse.data);
        setHybridResponse(hybridResponse);
      } catch (err) {
        setError(err.message);
        console.error("Error fetching data:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [id]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-900 to-black flex items-center justify-center">
        <LoadingSpinner />
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-900 to-black flex items-center justify-center">
        <div className="bg-red-500/10 border border-red-500/20 text-red-500 text-center p-6 rounded-lg max-w-2xl mx-auto">
          <h3 className="text-xl font-semibold mb-2">Error</h3>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (!movie) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-900 to-black flex items-center justify-center">
        <div className="text-white text-center">Movie not found</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-900 to-black pt-16">
      {/* Back Button */}
      <motion.button
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        onClick={() => navigate(-1)}
        className="fixed top-4 left-4 z-50 flex items-center space-x-2 bg-gray-800/90 text-white px-4 py-2 rounded-lg 
                   backdrop-blur-sm shadow-lg hover:bg-gray-700/90 transition-all duration-200"
      >
        <ArrowLeft className="w-5 h-5" />
        <span>Back</span>
      </motion.button>

      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="container mx-auto px-4 py-8"
      >
        <div className="bg-gray-800/50 rounded-2xl shadow-2xl overflow-hidden backdrop-blur-sm">
          <div className="md:flex">
            {/* Movie Poster */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="md:w-1/3 relative"
            >
              {movie.poster && isValidUrl(movie.poster) ? (
                <img
                  src={movie.poster}
                  alt={movie.title}
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    e.target.src = "https://via.placeholder.com/500x750?text=No+Poster";
                  }}
                />
              ) : (
                <img
                  src="https://via.placeholder.com/500x750?text=No+Poster"
                  alt={movie.title}
                  className="w-full h-full object-cover"
                />
              )}
              <div className="absolute inset-0 bg-gradient-to-t from-gray-900 via-transparent to-transparent md:hidden" />
            </motion.div>

            {/* Movie Info */}
            <div className="md:w-2/3 p-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
              >
                <h1 className="text-4xl font-bold text-white mb-4">{movie.title}</h1>

                {/* Movie Stats */}
                <div className="flex flex-wrap items-center gap-4 mb-6 text-gray-300">
                  {movie.vote_average ? (
                    <div className="flex items-center">
                      <Star className="w-5 h-5 text-yellow-500 mr-1" />
                      <span>{Number(movie.vote_average).toFixed(1)}</span>
                    </div>
                  ) : (
                    <div className="flex items-center">
                      <Star className="w-5 h-5 text-gray-500 mr-1" />
                      <span>N/A</span>
                    </div>
                  )}
                  {movie.release_date ? (
                    <div className="flex items-center">
                      <Calendar className="w-5 h-5 text-blue-500 mr-1" />
                      <span>{new Date(movie.release_date).getFullYear()}</span>
                    </div>
                  ) : (
                    <div className="flex items-center">
                      <Calendar className="w-5 h-5 text-gray-500 mr-1" />
                      <span>N/A</span>
                    </div>
                  )}
                  {movie.original_language ? (
                    <div className="flex items-center">
                      <Globe className="w-5 h-5 text-green-500 mr-1" />
                      <span>{movie.original_language.toUpperCase()}</span>
                    </div>
                  ) : (
                    <div className="flex items-center">
                      <Globe className="w-5 h-5 text-gray-500 mr-1" />
                      <span>N/A</span>
                    </div>
                  )}
                </div>

                {/* Genres */}
                <div className="flex flex-wrap gap-2 mb-6">
                  {movie.genres && Array.isArray(movie.genres) ? (
                    movie.genres.map((genre) => (
                      <span
                        key={genre}
                        className="px-3 py-1 bg-blue-600/20 border border-blue-500/30 text-blue-400 
                                 text-sm rounded-full backdrop-blur-sm"
                      >
                        {genre}
                      </span>
                    ))
                  ) : (
                    <span className="text-gray-500">No genres available</span>
                  )}
                </div>

                {/* Overview */}
                <p className="text-gray-300 text-lg mb-8 leading-relaxed">{movie.overview}</p>

                {/* Cast & Crew */}
                <div className="grid md:grid-cols-2 gap-8">
                  {/* Cast */}
                  <div className="bg-gray-800/30 rounded-xl p-6 backdrop-blur-sm">
                    <h3 className="text-xl font-semibold text-white mb-4 flex items-center">Cast</h3>
                    <ul className="space-y-2">
                      {movie.cast?.slice(0, 6).map((actor) => (
                        <li key={actor} className="text-gray-300 hover:text-white transition-colors">
                          {actor}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Crew */}
                  <div className="bg-gray-800/30 rounded-xl p-6 backdrop-blur-sm">
                    <h3 className="text-xl font-semibold text-white mb-4">Crew</h3>
                    <ul className="space-y-2">
                      {movie.crew?.slice(0, 6).map((member) => (
                        <li key={member} className="text-gray-300 hover:text-white transition-colors">
                          {member}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                {/* Keywords */}
                <div className="mt-8">
                  <h3 className="text-xl font-semibold text-white mb-4">Keywords</h3>
                  <div className="flex flex-wrap gap-2">
                    {movie.keywords?.map((keyword) => (
                      <span
                        key={keyword}
                        className="px-3 py-1 bg-gray-700/50 text-gray-300 text-sm rounded-lg
                                 hover:bg-gray-700 transition-colors"
                      >
                        {keyword}
                      </span>
                    ))}
                  </div>
                </div>
              </motion.div>
            </div>
          </div>
        </div>

        {/* Recommendations Section */}
        {hybridResponse && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="mt-16"
          >
            {/* Collaborative Recommendations */}
            {hybridResponse.data.hybrid_recommendations.collaborative.length > 0 && (
              <>
                <h2 className="text-3xl font-bold text-white mb-8">Based on Your Ratings</h2>
                <MovieGrid
                  movies={hybridResponse.data.hybrid_recommendations.collaborative
                    .filter((rec) => rec.movie_id !== parseInt(id))
                    .slice(0, 4)}
                />
              </>
            )}

            {/* Content-Based Recommendations */}
            {hybridResponse.data.hybrid_recommendations.content_based.length > 0 && (
              <>
                <h2 className="text-3xl font-bold text-white mb-8 mt-12">Similar Movies</h2>
                <MovieGrid
                  movies={hybridResponse.data.hybrid_recommendations.content_based
                    .filter((rec) => rec.movie_id !== parseInt(id))
                    .slice(0, 4)}
                />
              </>
            )}
          </motion.div>
        )}
      </motion.div>
    </div>
  );
};

export default MovieDetails;