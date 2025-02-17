// // src/pages/HomePage.jsx
// import { useState, useEffect } from 'react';
// import MovieGrid from '../components/MovieGrid';
// import SearchBar from '../components/SearchBar';
// import LoadingSpinner from '../components/LoadingSpinner';

// const HomePage = () => {
//   const [movies, setMovies] = useState([]);
//   const [filteredMovies, setFilteredMovies] = useState([]);
//   const [page, setPage] = useState(1);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState(null);
//   const [hasMore, setHasMore] = useState(true);
//   const [searchQuery, setSearchQuery] = useState('');
//   const [isSearching, setIsSearching] = useState(false);

//   const fetchMovies = async () => {
//     try {
//       setLoading(true);
//       const response = await fetch(`http://localhost:5000/api/movies?page=${page}&page_size=12`);
//       if (!response.ok) throw new Error('Failed to fetch movies');
      
//       const data = await response.json();
//       const newMovies = page === 1 ? data.movies : [...movies, ...data.movies];
//       setMovies(newMovies);
//       filterMovies(newMovies, searchQuery);
//       setHasMore(page < data.pagination.total_pages);
//     } catch (err) {
//       setError(err.message);
//     } finally {
//       setLoading(false);
//     }
//   };

//   // Enhanced search function that checks multiple fields and ranks results
//   const filterMovies = (movieList, query) => {
//     if (!query.trim()) {
//       setFilteredMovies(movieList);
//       setIsSearching(false);
//       return;
//     }

//     setIsSearching(true);
//     const searchTerms = query.toLowerCase().split(' ').filter(term => term.length > 0);
    
//     const getMatchScore = (movie) => {
//       let score = 0;
      
//       searchTerms.forEach(term => {
//         // Title matches (highest priority)
//         if (movie.title.toLowerCase().includes(term)) {
//           score += 10;
//           if (movie.title.toLowerCase().startsWith(term)) score += 5;
//         }
        
//         // Genre matches
//         if (movie.genres?.some(genre => genre.toLowerCase().includes(term))) {
//           score += 8;
//         }
        
//         // Cast matches
//         if (movie.cast?.some(actor => actor.toLowerCase().includes(term))) {
//           score += 6;
//         }
        
//         // Keyword matches
//         if (movie.keywords?.some(keyword => keyword.toLowerCase().includes(term))) {
//           score += 4;
//         }
        
//         // Overview matches
//         if (movie.overview?.toLowerCase().includes(term)) {
//           score += 2;
//         }
//       });
      
//       return score;
//     };

//     const scored = movieList
//       .map(movie => ({
//         movie,
//         score: getMatchScore(movie)
//       }))
//       .filter(item => item.score > 0)
//       .sort((a, b) => b.score - a.score);

//     const filteredResults = scored.map(item => item.movie);
//     setFilteredMovies(filteredResults);
//   };

//   const handleSearch = (query) => {
//     setSearchQuery(query);
//     setPage(1); // Reset to first page when searching
//     filterMovies(movies, query);
//   };

//   useEffect(() => {
//     fetchMovies();
//   }, [page]);

//   const loadMore = () => {
//     if (!loading && hasMore && !isSearching) {
//       setPage(prev => prev + 1);
//     }
//   };

//   return (
//     <div className="container mx-auto px-4 py-8">
//       <SearchBar onSearch={handleSearch} />
      
//       {error && (
//         <div className="text-red-500 text-center mb-4">{error}</div>
//       )}
      
//       {isSearching && filteredMovies.length > 0 && (
//         <div className="text-white text-center mb-6">
//           Found {filteredMovies.length} results for "{searchQuery}"
//         </div>
//       )}
      
//       {isSearching && filteredMovies.length === 0 && (
//         <div className="text-center py-12">
//           <h3 className="text-xl text-white mb-2">No results found for "{searchQuery}"</h3>
//           <p className="text-gray-400">Try adjusting your search terms or browse our movie collection below</p>
//         </div>
//       )}
      
//       <MovieGrid movies={filteredMovies} />
      
//       {loading && <LoadingSpinner />}
      
//       {hasMore && !loading && !isSearching && (
//         <div className="text-center mt-8">
//           <button 
//             onClick={loadMore}
//             className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-6 rounded-lg transition-colors duration-200"
//           >
//             Load More
//           </button>
//         </div>
//       )}
//     </div>
//   );
// };

// export default HomePage;






import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import MovieGrid from '../components/MovieGrid';
import SearchBar from '../components/SearchBar';
import LoadingSpinner from '../components/LoadingSpinner';
import { Film, Heart, BookMarked, User, Menu, X, Bell } from 'lucide-react';

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
      isScrolled ? 'bg-gray-900/95 backdrop-blur-md shadow-lg' : 'bg-transparent'
    }`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo and Brand */}
          <div className="flex items-center space-x-4">
            <Film className="w-8 h-8 text-blue-500" />
            <span className="text-xl font-bold text-white">Recommendation System </span>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            <NavLink icon={<Film className="w-5 h-5" />} text="Movies" active />
            <NavLink icon={<Heart className="w-5 h-5" />} text="Watchlist" />
            <NavLink icon={<BookMarked className="w-5 h-5" />} text="Collections" />
            
            {/* Notification Bell */}
            <button className="relative p-2 text-gray-400 hover:text-white transition-colors">
              <Bell className="w-5 h-5" />
              <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
            </button>
            
            {/* User Profile */}
            <div className="flex items-center space-x-3 text-gray-300 hover:text-white cursor-pointer">
              <div className="w-8 h-8 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center">
                <User className="w-5 h-5 text-white" />
              </div>
            </div>
          </div>

          {/* Mobile Menu Button */}
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

      {/* Mobile Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
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
      active
        ? 'text-white bg-gray-800'
        : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
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
      active
        ? 'text-white bg-gray-800'
        : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
    }`}
  >
    {text}
  </a>
);

const HomePage = () => {
  const [movies, setMovies] = useState([]);
  const [filteredMovies, setFilteredMovies] = useState([]);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasMore, setHasMore] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);

  const fetchMovies = async () => {
    try {
      setLoading(true);
      const response = await fetch(`http://localhost:5000/api/movies?page=${page}&page_size=12`);
      if (!response.ok) throw new Error('Failed to fetch movies');
      
      const data = await response.json();
      const newMovies = page === 1 ? data.movies : [...movies, ...data.movies];
      setMovies(newMovies);
      filterMovies(newMovies, searchQuery);
      setHasMore(page < data.pagination.total_pages);
    } catch (err) {
      setError(err.message);
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
    const searchTerms = query.toLowerCase().split(' ').filter(term => term.length > 0);
    
    const getMatchScore = (movie) => {
      let score = 0;
      searchTerms.forEach(term => {
        if (movie.title.toLowerCase().includes(term)) {
          score += 10;
          if (movie.title.toLowerCase().startsWith(term)) score += 5;
        }
        if (movie.genres?.some(genre => genre.toLowerCase().includes(term))) score += 8;
        if (movie.cast?.some(actor => actor.toLowerCase().includes(term))) score += 6;
        if (movie.keywords?.some(keyword => keyword.toLowerCase().includes(term))) score += 4;
        if (movie.overview?.toLowerCase().includes(term)) score += 2;
      });
      return score;
    };

    const scored = movieList
      .map(movie => ({ movie, score: getMatchScore(movie) }))
      .filter(item => item.score > 0)
      .sort((a, b) => b.score - a.score);

    setFilteredMovies(scored.map(item => item.movie));
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
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  useEffect(() => {
    fetchMovies();
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [page]);

  const loadMore = () => {
    if (!loading && hasMore && !isSearching) {
      setPage(prev => prev + 1);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-900 to-black">
      <Navbar />
      
      <main className="container mx-auto px-4 pt-24 pb-8">
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

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="mb-8"
        >
          <MovieGrid movies={filteredMovies} />
        </motion.div>
        
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
              className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-8 rounded-lg
                         transform transition-all duration-200 hover:scale-105 hover:shadow-lg
                         focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Load More
            </motion.button>
          )}
        </div>

        <AnimatePresence>
          {showScrollButton && (
            <motion.button
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              onClick={scrollToTop}
              className="fixed bottom-8 right-8 bg-blue-600 hover:bg-blue-700 text-white p-4 rounded-full
                         shadow-lg transition-colors duration-200 focus:outline-none focus:ring-2
                         focus:ring-blue-500 focus:ring-opacity-50"
            >
              <svg
                className="w-6 h-6"
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path d="M5 10l7-7m0 0l7 7m-7-7v18" />
              </svg>
            </motion.button>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
};

export default HomePage;