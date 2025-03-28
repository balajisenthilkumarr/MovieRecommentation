// src/components/MovieList.js
import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronLeft, ChevronRight, Play } from "lucide-react";

const MovieList = () => {
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [scrollPosition, setScrollPosition] = useState(0);
  const containerRef = useRef(null);

  // Fetch initial recommendations
  useEffect(() => {
    const fetchInitialRecommendations = async () => {
      try {
        setLoading(true);
        const response = await axios.get("http://localhost:5000/api/initial_recommendations");
        setMovies(response.data.recommendations);
      } catch (err) {
        setError("Failed to fetch initial recommendations");
        console.error("Error fetching initial recommendations:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchInitialRecommendations();
  }, []);

  // Handle scrolling left and right
  const scrollLeft = () => {
    const container = containerRef.current;
    if (container) {
      const newPosition = Math.max(scrollPosition - 300, 0);
      container.scrollTo({ left: newPosition, behavior: "smooth" });
      setScrollPosition(newPosition);
    }
  };

  const scrollRight = () => {
    const container = containerRef.current;
    if (container) {
      const maxScroll = container.scrollWidth - container.clientWidth;
      const newPosition = Math.min(scrollPosition + 300, maxScroll);
      container.scrollTo({ left: newPosition, behavior: "smooth" });
      setScrollPosition(newPosition);
    }
  };

  // Check if buttons should be disabled
  const isAtStart = scrollPosition <= 0;
  const isAtEnd = () => {
    const container = containerRef.current;
    if (container) {
      return scrollPosition >= container.scrollWidth - container.clientWidth;
    }
    return false;
  };

  if (loading) {
    return (
      <div className="text-white text-center py-10">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
          className="inline-block w-8 h-8 border-4 border-t-transparent border-blue-500 rounded-full"
        />
        <p className="mt-2">Loading...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-red-500 text-center py-10">
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className="relative px-2 py-8">
      {/* Scrollable container for movie cards */}
      <div
        ref={containerRef}
        id="movie-list-scroll"
        className="flex overflow-x-hidden space-x-6 scrollbar-hide snap-x snap-mandatory"
        style={{ scrollBehavior: "smooth" }}
      >
        <AnimatePresence>
          {movies.map((movie, index) => (
            <motion.div
              key={movie.movie_id}
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -50 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="flex-none w-72 bg-gray-900 rounded-xl overflow-hidden shadow-xl transform transition-all duration-300 hover:shadow-2xl hover:-translate-y-2 snap-start"
            >
              {/* Movie Poster */}
              <div className="relative h-96 group">
                <img
                  src={movie.poster}
                  alt={movie.title}
                  className="w-full h-full object-cover transition-opacity duration-300 group-hover:opacity-70"
                  onError={(e) => (e.target.src = "https://via.placeholder.com/256x384?text=No+Image")}
                />
                {/* Gradient Overlay */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/50 to-transparent opacity-80 group-hover:opacity-100 transition-opacity duration-300"></div>
                {/* Predicted Rating */}
                <div className="absolute top-4 right-4 bg-yellow-400 text-black text-sm font-bold px-3 py-1 rounded-full shadow-md">
                  {movie.predicted_rating.toFixed(1)} ★
                </div>
                {/* Play Button on Hover */}
                <motion.div
                  className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                  whileHover={{ scale: 1.1 }}
                >
                  <button className="bg-red-600 text-white p-4 rounded-full shadow-lg hover:bg-red-700 transition-colors duration-200">
                    <Play className="w-6 h-6" />
                  </button>
                </motion.div>
              </div>
              {/* Movie Details */}
              <div className="p-5">
                <h3 className="text-white text-xl font-bold truncate">{movie.title}</h3>
                <p className="text-gray-400 text-sm mt-2">
                  {movie.genres?.join(", ") || "No genres available"}
                </p>
                <p className="text-gray-500 text-xs mt-1">Year: {movie.year || "N/A"}</p>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Navigation Buttons */}
      <motion.button
        onClick={scrollLeft}
        disabled={isAtStart}
        className={`absolute left-0 top-1/2 transform -translate-y-1/2 bg-gradient-to-r from-blue-600 to-blue-800 text-white p-4 rounded-full shadow-lg transition-all duration-300 ${
          isAtStart ? "opacity-50 cursor-not-allowed" : "hover:bg-blue-700 hover:scale-110"
        }`}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
      >
        <ChevronLeft className="w-8 h-8" />
      </motion.button>
      <motion.button
        onClick={scrollRight}
        disabled={isAtEnd()}
        className={`absolute right-0 top-1/2 transform -translate-y-1/2 bg-gradient-to-r from-blue-600 to-blue-800 text-white p-4 rounded-full shadow-lg transition-all duration-300 ${
          isAtEnd() ? "opacity-50 cursor-not-allowed" : "hover:bg-blue-700 hover:scale-110"
        }`}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
      >
        <ChevronRight className="w-8 h-8" />
      </motion.button>
    </div>
  );
};

export default MovieList;