// src/pages/OnboardingPage.js
import { useState, useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

const OnboardingPage = () => {
  const [genres, setGenres] = useState([]);
  const [actors, setActors] = useState([]);
  const [directors, setDirectors] = useState([]);
  const [ratings, setRatings] = useState([]);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  // Get userId from localStorage
  const userId = parseInt(localStorage.getItem("userId"));

  // Check if user has already set preferences
  useEffect(() => {
    const checkPreferences = async () => {
      if (!userId) {
        console.log("No userId found, redirecting to login");
        navigate("/login");
        return;
      }
      try {
        console.log(`Fetching preferences for userId: ${userId}`);
        const response = await axios.get(`http://localhost:5000/api/user/preferences/${userId}`);
        console.log("Preferences response:", response.data);

        // Check if preferences exist and are non-empty
        const hasPreferences =
          response.data.preferences &&
          (response.data.preferences.genres?.length > 0 || response.data.preferences.ratings?.length > 0);

        if (hasPreferences) {
          console.log(`User ${userId} already has preferences, redirecting to homepage`);
          navigate("/");
        } else {
          console.log(`No preferences found for user ${userId}, staying on onboarding page`);
        }
      } catch (err) {
        if (err.response?.status === 404) {
          console.log(`No preferences found for user ${userId} (404), staying on onboarding page`);
        } else {
          setError("Failed to check preferences");
          console.error("Error checking preferences:", err);
        }
      }
    };

    checkPreferences();
  }, [userId, navigate]);

  // Redirect to login if no userId is found
  if (!userId) {
    return null;
  }

  // List of available genres, actors, directors, and movies for selection
  const availableGenres = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller"];
  const availableActors = ["Keanu Reeves", "Tom Cruise", "Leonardo DiCaprio", "Emma Stone"];
  const availableDirectors = ["Christopher Nolan", "Martin Scorsese", "Quentin Tarantino"];
  const availableMovies = [
    { movieId: 1, title: "Toy Story" },
    { movieId: 2, title: "Jumanji" },
    { movieId: 3, title: "Grumpier Old Men" },
    { movieId: 4, title: "Waiting to Exhale" },
    { movieId: 5, title: "Father of the Bride Part II" },
  ];

  const handleGenreChange = (genre) => {
    setGenres((prev) =>
      prev.includes(genre) ? prev.filter((g) => g !== genre) : [...prev, genre]
    );
  };

  const handleActorChange = (actor) => {
    setActors((prev) =>
      prev.includes(actor) ? prev.filter((a) => a !== actor) : [...prev, actor]
    );
  };

  const handleDirectorChange = (director) => {
    setDirectors((prev) =>
      prev.includes(director) ? prev.filter((d) => d !== director) : [...prev, director]
    );
  };

  const handleRatingChange = (movieId, rating) => {
    setRatings((prev) => {
      const existingRating = prev.find((r) => r.movieId === movieId);
      if (existingRating) {
        return prev.map((r) =>
          r.movieId === movieId ? { ...r, rating: parseFloat(rating) } : r
        );
      }
      return [...prev, { movieId, rating: parseFloat(rating) }];
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // Save preferences to the backend
      const preferences = {
        userId,
        genres,
        actors,
        directors,
        ratings,
      };
      console.log(`Saving preferences for userId: ${userId}`, preferences);
      await axios.post("http://localhost:5000/api/user/preferences", preferences);

      // Redirect to homepage
      navigate("/");
    } catch (err) {
      setError("Failed to save preferences");
      console.error("Error saving preferences:", err);
    }
  };

  const handleSkip = () => {
    navigate("/");
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-900 to-black flex items-center justify-center">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="bg-gray-800 p-8 rounded-lg shadow-lg max-w-lg w-full"
      >
        <h2 className="text-3xl font-bold text-white mb-6 text-center">Tell Us About Yourself</h2>
        {error && (
          <div className="bg-red-500/10 border border-red-500/20 text-red-500 text-center p-4 rounded-lg mb-6">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          {/* Genres */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-white mb-2">Favorite Genres</h3>
            <div className="flex flex-wrap gap-2">
              {availableGenres.map((genre) => (
                <label key={genre} className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={genres.includes(genre)}
                    onChange={() => handleGenreChange(genre)}
                    className="form-checkbox h-5 w-5 text-blue-600"
                  />
                  <span className="text-gray-300">{genre}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Actors */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-white mb-2">Favorite Actors</h3>
            <div className="flex flex-wrap gap-2">
              {availableActors.map((actor) => (
                <label key={actor} className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={actors.includes(actor)}
                    onChange={() => handleActorChange(actor)}
                    className="form-checkbox h-5 w-5 text-blue-600"
                  />
                  <span className="text-gray-300">{actor}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Directors */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-white mb-2">Favorite Directors</h3>
            <div className="flex flex-wrap gap-2">
              {availableDirectors.map((director) => (
                <label key={director} className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={directors.includes(director)}
                    onChange={() => handleDirectorChange(director)}
                    className="form-checkbox h-5 w-5 text-blue-600"
                  />
                  <span className="text-gray-300">{director}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Ratings */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-white mb-2">Rate Some Movies</h3>
            {availableMovies.map((movie) => (
              <div key={movie.movieId} className="flex items-center space-x-4 mb-2">
                <span className="text-gray-300 w-40">{movie.title}</span>
                <select
                  onChange={(e) => handleRatingChange(movie.movieId, e.target.value)}
                  className="bg-gray-700 text-white p-2 rounded"
                >
                  <option value="">Select Rating</option>
                  {[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5].map((rating) => (
                    <option key={rating} value={rating}>
                      {rating}
                    </option>
                  ))}
                </select>
              </div>
            ))}
          </div>

          <div className="flex space-x-4">
            <button
              type="submit"
              className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg transition-colors"
            >
              Save Preferences
            </button>
            <button
              type="button"
              onClick={handleSkip}
              className="flex-1 bg-gray-600 hover:bg-gray-700 text-white font-bold py-3 px-4 rounded-lg transition-colors"
            >
              Skip
            </button>
          </div>
        </form>
      </motion.div>
    </div>
  );
};

export default OnboardingPage;