// // src/components/MovieCard.jsx
// const MovieCard = ({ movie }) => {
//     return (
//       <div className="bg-gray-800 rounded-lg overflow-hidden shadow-lg transform transition-all duration-300 hover:scale-105">
//         <div className="relative pb-[150%]">
//           <img
//             src={movie.poster}
//             alt={movie.title}
//             className="absolute inset-0 w-full h-full object-cover"
//             onError={(e) => {
//               e.target.src = "https://via.placeholder.com/500x750?text=No+Poster";
//             }}
//           />
//           <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent" />
//         </div>
//         <div className="p-4">
//           <h3 className="text-xl font-semibold text-white mb-2 line-clamp-2">{movie.title}</h3>
//           <div className="flex items-center justify-between text-gray-300">
//             <span>{new Date(movie.release_date).getFullYear()}</span>
//             <div className="flex items-center">
//               <span className="text-yellow-400 mr-1">★</span>
//               <span>{movie.vote_average?.toFixed(1)}</span>
//             </div>
//           </div>
//         </div>
//       </div>
//     );
//   };
  
//   export default MovieCard;
  



import React from 'react';

const MovieCard = ({ movie }) => {
  return (
    <div className="group bg-gray-800 rounded-lg overflow-hidden shadow-lg transform transition-all duration-300 hover:scale-105 h-full">
      {/* Poster Container */}
      <div className="relative aspect-[2/3]">
        <img
          src={movie.poster}
          alt={movie.title}
          className="absolute inset-0 w-full h-full object-cover transform transition-transform duration-300 group-hover:scale-105"
          onError={(e) => {
            e.target.src = "https://via.placeholder.com/500x750?text=No+Poster";
          }}
        />
        
        {/* Overlay with gradient and content */}
        <div className="absolute inset-0 bg-gradient-to-t from-black via-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300">
          {/* Movie Overview */}
          <div className="absolute bottom-0 left-0 right-0 p-4">
            <p className="text-white text-sm line-clamp-3 mb-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300 delay-100">
              {movie.overview || 'No overview available'}
            </p>
          </div>
        </div>

        {/* Genre Tags */}
        <div className="absolute top-2 left-2 right-2 flex flex-wrap gap-1">
          {movie.genres?.slice(0, 2).map((genre) => (
            <span
              key={genre}
              className="bg-blue-600/80 text-white text-xs px-2 py-1 rounded-full backdrop-blur-sm"
            >
              {genre}
            </span>
          ))}
        </div>
      </div>

      {/* Content Section */}
      <div className="p-4">
        {/* Title */}
        <h3 className="text-lg font-semibold text-white mb-2 line-clamp-2 min-h-[3.5rem]">
          {movie.title}
        </h3>

        {/* Movie Details */}
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center space-x-2">
            {/* Release Year */}
            <span className="text-gray-400">
              {movie.release_date ? new Date(movie.release_date).getFullYear() : 'N/A'}
            </span>
            
            {/* Separator */}
            <span className="text-gray-600">•</span>
            
            {/* Original Language */}
            <span className="text-gray-400 uppercase">
              {movie.original_language || 'N/A'}
            </span>
          </div>

          {/* Rating */}
          {movie.vote_average && (
            <div className="flex items-center bg-gray-700/50 px-2 py-1 rounded">
              <span className="text-yellow-400 mr-1">★</span>
              <span className="text-white">
                {Number(movie.vote_average).toFixed(1)}
              </span>
            </div>
          )}
        </div>

        {/* Cast Preview */}
        {movie.cast && movie.cast.length > 0 && (
          <div className="mt-3 text-sm text-gray-400">
            <p className="line-clamp-1">
              With: {movie.cast.slice(0, 3).join(', ')}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default MovieCard;