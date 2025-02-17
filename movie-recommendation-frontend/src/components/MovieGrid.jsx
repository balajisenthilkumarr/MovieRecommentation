// src/components/MovieGrid.jsx
import { Link } from 'react-router-dom';
import MovieCard from './MovieCard';

const MovieGrid = ({ movies }) => {
  //console.log("ddddddddd",movies);
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
      {movies.map(movie => (
        <Link key={movie.movie_id} to={`/movie/${movie.movie_id}`}>
          <MovieCard movie={movie} />
        </Link>
      ))}
    </div>
  );
};

export default MovieGrid;