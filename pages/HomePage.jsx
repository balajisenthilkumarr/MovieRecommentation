// src/pages/HomePage.js
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import MovieGrid from "../components/MovieGrid";
import SearchBar from "../components/SearchBar";
import LoadingSpinner from "../components/LoadingSpinner";
import axios from "axios";
import { Film, Heart, BookMarked, User, Menu, X, Bell } from "lucide-react";
import MovieList from "../components/MovieList";
import { useNavigate } from "react-router-dom";

// Navbar Component (unchanged)
const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        isScrolled ? "bg-gray-900/95 backdrop-blur-md shadow-lg" : "bg-transparent"
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-4">
            <Film className="w-8 h-8 text-blue-500" />
            <span className="text-xl font-bold text-white">Recommendation System</span>
          </div>
          <div className="hidden md:flex items-center space-x-8">
            <NavLink icon={<Film className="w-5 h-5" />} text="Movies" active />
            <NavLink icon={<Heart className="w-5 h-5" />} text="Watchlist" />
            <NavLink icon={<BookMarked className="w-5 h-5" />} text="Collections" />
            <button className="relative p-2 text-gray-400 hover:text-white transition-colors">
              <Bell className="w-5 h-5" />
              <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
            </button>
            <div className="flex items-center space-x-3 text-gray-300 hover:text-white cursor-pointer">
              <div className="w-8 h-8 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center">
                <User className="w-5 h-5 text-white" />
              </div>
            </div>
          </div>
          <div className="md:hidden">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="text-gray-400 hover:text-white transition-colors"
            >
              {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </div>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden bg-gray-900/95 backdrop-blur-md"
          >
            <div className="px-4 pt-2 pb-3 space-y-1">
              <MobileNavLink text="Movies" active />
              <MobileNavLink text="Watchlist" />
              <MobileNavLink text="Collections" />
              <MobileNavLink text="Profile" />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
};

const NavLink = ({ icon, text, active }) => (
  <a
    href="#"
    className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors ${
      active ? "text-white bg-gray-800" : "text-gray-400 hover:text-white hover:bg-gray-800/50"
    }`}
  >
    {icon}
    <span>{text}</span>
  </a>
);

const MobileNavLink = ({ text, active }) => (
  <a
    href="#"
    className={`block px-3 py-2 rounded-lg text-base font-medium transition-colors ${
      active ? "text-white bg-gray-800" : "text-gray-400 hover:text-white hover:bg-gray-800/50"
    }`}
  >
    {text}
  </a>
);

const HomePage = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const [movies, setMovies] = useState([]);
  const [collaborativeMovies, setCollaborativeMovies] = useState([]);
  const [contentBasedMovies, setContentBasedMovies] = useState([]);
  const [filteredMovies, setFilteredMovies] = useState([]);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasMore, setHasMore] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);

  // Get userId from localStorage
  const userId = parseInt(localStorage.getItem("userId"));
  const navigate = useNavigate();

  // Redirect to login if no userId is found
  useEffect(() => {
    if (!userId) {
      console.log("No userId found in localStorage, redirecting to login");
      navigate("/login");
    }
  }, [userId, navigate]);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
      setShowScrollButton(window.scrollY > 500); // Same threshold as old code
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Fetch Collaborative Recommendations
  const fetchCollaborativeMovies = async () => {
    if (!userId) return;
    try {
      setLoading(true);
      console.log(`Fetching collaborative recommendations for userId: ${userId}`);
      const response = await axios.get(`http://localhost:5000/api/collaborative_recommendations/${userId}`);
      setCollaborativeMovies(response.data.recommendations);
    } catch (err) {
      setError("Failed to fetch collaborative recommendations");
      console.error("Error fetching collaborative recommendations:", err);
    } finally {
      setLoading(false);
    }
  };

  const fetchContentBasedMovies = async () => {
    if (!userId) return;
    try {
      setLoading(true);
      const preferencesResponse = await axios.get(`http://localhost:5000/api/user/preferences/${userId}`);
      const userGenres = preferencesResponse.data.preferences?.genres || [];
      console.log(`User ${userId} preferred genres:`, userGenres);

      let movieId;
      if (userGenres.length > 0) {
        const moviesResponse = await axios.get(`http://localhost:5000/api/movies?page=1&page_size=100`);
        const movies = moviesResponse.data.movies;
        const matchingMovie = movies.find((movie) =>
          movie.genres?.some((genre) => userGenres.includes(genre))
        );
        if (matchingMovie) {
          movieId = matchingMovie.movie_id;
          console.log(`Selected movie for content-based recommendations: ${matchingMovie.title} (ID: ${movieId}, Genres: ${matchingMovie.genres})`);
        } else {
          console.log("No movie found matching user genres, falling back to a popular movie");
          const popularMovieResponse = await axios.get(`http://localhost:5000/api/movies?page=1&page_size=1`);
          movieId = popularMovieResponse.data.movies[0].movie_id;
          console.log(`Fallback movie ID: ${movieId}`);
        }
      } else {
        console.log("No user genres specified, using a default movie");
        const movieResponse = await axios.get(`http://localhost:5000/api/movies?page=1&page_size=1`);
        movieId = movieResponse.data.movies[0].movie_id;
        console.log(`Default movie ID: ${movieId}`);
      }

      const response = await axios.get(`http://localhost:5000/api/recommendations/${movieId}`);
      setContentBasedMovies(response.data.recommendations);
    } catch (err) {
      setError("Failed to fetch content-based recommendations");
      console.error("Error fetching content-based recommendations:", err);
    } finally {
      setLoading(false);
    }
  };

  const fetchMovies = async (pageNum) => {
    try {
      setLoading(true);
      const response = await axios.get(`http://localhost:5000/api/movies`, {
        params: { page: pageNum, page_size: 12 },
      });
      const newMovies = pageNum === 1 ? response.data.movies : [...movies, ...response.data.movies];
      setMovies(newMovies);
      filterMovies(newMovies, searchQuery);
      setHasMore(pageNum < response.data.pagination.total_pages);
    } catch (err) {
      setError("Failed to fetch movies");
      console.error("Error fetching movies:", err);
    } finally {
      setLoading(false);
    }
  };

  const filterMovies = (movieList, query) => {
    if (!query.trim()) {
      setFilteredMovies(movieList);
      setIsSearching(false);
      return;
    }

    setIsSearching(true);
    const searchTerms = query.toLowerCase().split(" ").filter((term) => term.length > 0);

    const getMatchScore = (movie) => {
      let score = 0;
      searchTerms.forEach((term) => {
        if (movie.title.toLowerCase().includes(term)) {
          score += 10;
          if (movie.title.toLowerCase().startsWith(term)) score += 5;
        }
        if (movie.genres?.some((genre) => genre.toLowerCase().includes(term))) score += 8;
        if (movie.cast?.some((actor) => actor.toLowerCase().includes(term))) score += 6;
        if (movie.keywords?.some((keyword) => keyword.toLowerCase().includes(term))) score += 4;
        if (movie.overview?.toLowerCase().includes(term)) score += 2;
      });
      return score;
    };

    const scored = movieList
      .map((movie) => ({ movie, score: getMatchScore(movie) }))
      .filter((item) => item.score > 0)
      .sort((a, b) => b.score - a.score);

    setFilteredMovies(scored.map((item) => item.movie));
  };

  const handleSearch = (query) => {
    setSearchQuery(query);
    setPage(1);
    filterMovies(movies, query);
  };

  const handleScroll = () => {
    setShowScrollButton(window.scrollY > 500);
  };

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const loadMore = () => {
    if (!loading && hasMore && !isSearching) {
      setPage((prev) => prev + 1);
    }
  };

  useEffect(() => {
    if (userId) {
      fetchCollaborativeMovies();
      fetchContentBasedMovies();
      fetchMovies(page);
    }
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, [page, userId]);

  if (!userId) {
    return null; // Render nothing while redirecting
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-900 to-black">
      <Navbar />

      <main className="container mx-auto px-0 pt-24 pb-8">        {/* Search Bar Section */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="relative z-10 mb-8"
        >
          <div className="max-w-2xl mx-auto">
            <SearchBar onSearch={handleSearch} />
          </div>
        </motion.div>

        {/* Error Message */}
        <AnimatePresence mode="wait">
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="bg-red-500/10 border border-red-500/20 text-red-500 text-center p-4 rounded-lg mb-6 max-w-2xl mx-auto"
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Initial Recommendations (MovieList) */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="mb-12" // Slightly larger margin to separate sections
        >
          <h2 className="text-3xl font-bold text-white mb-6">Initial Recommendations</h2>
          <MovieList />
        </motion.div>

        {/* Search Results */}
        <AnimatePresence mode="wait">
          {isSearching && (
            <motion.div

              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="mb-8"
            >
              {filteredMovies.length > 0 ? (
                <div className="text-center space-y-2">
                  <div className="text-lg text-white/90">
                    Found <span className="font-semibold text-blue-400">{filteredMovies.length}</span> results
                  </div>
                  <div className="text-sm text-gray-400">
                    for "<span className="italic text-blue-400">{searchQuery}</span>"
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 bg-gray-800/30 rounded-xl backdrop-blur-sm max-w-2xl mx-auto">
                  <h3 className="text-2xl text-white/90 mb-3">No results found</h3>
                  <p className="text-gray-400">Try adjusting your search terms or browse our collection</p>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Collaborative Recommendations */}
        {Array.isArray(collaborativeMovies) && collaborativeMovies.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="mb-12" // Consistent margin with other sections
          >
            <h2 className="text-3xl font-bold text-white mb-6">Movies You Might Like (Collaborative)</h2>
            <MovieGrid movies={collaborativeMovies.slice(0, 8)} />
          </motion.div>
        )}

        {/* Content-Based Recommendations */}
        {Array.isArray(contentBasedMovies) && contentBasedMovies.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="mb-12" // Consistent margin with other sections
          >
            <h2 className="text-3xl font-bold text-white mb-6">Similar Movies (Content-Based)</h2>
            <MovieGrid movies={contentBasedMovies.slice(0, 8)} />
          </motion.div>
        )}

        {/* Explore Movies / Search Results */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="mb-8"
        >
          <h2 className="text-3xl font-bold text-white mb-6">
            {isSearching ? "Search Results" : "Explore Movies"}
          </h2>
          <MovieGrid movies={filteredMovies} />
        </motion.div>

        {/* Loading Spinner and Load More Button */}
        <div className="flex justify-center mt-8">
          {loading && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
            >
              <LoadingSpinner />
            </motion.div>
          )}

          {hasMore && !loading && !isSearching && (
            <motion.button
              onClick={loadMore}
              className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-8 rounded-lg transform transition-all duration-200 hover:scale-105 hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Load More
            </motion.button>
          )}
        </div>

        {/* Scroll to Top Button */}
        <AnimatePresence>
          {showScrollButton && (
            <motion.button
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              onClick={scrollToTop}
              className="fixed bottom-8 right-8 bg-blue-600 hover:bg-blue-700 text-white p-4 rounded-full shadow-lg transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
            >
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 10l7-7m0 0l7 7m-7-7v18" />
              </svg>
            </motion.button>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
};

export default HomePage;