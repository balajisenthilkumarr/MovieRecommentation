// // src/components/SearchBar.jsx
// const SearchBar = ({ onSearch }) => {
//     return (
//       <div className="max-w-2xl mx-auto mb-8">
//         <input
//           type="text"
//           placeholder="Search movies..."
//           onChange={(e) => onSearch(e.target.value)}
//           className="w-full px-4 py-2 rounded-lg bg-gray-800 text-white border border-gray-700 focus:outline-none focus:border-blue-500"
//         />
//       </div>
//     );
//   };
  
//   export default SearchBar;


// src/components/SearchBar.jsx
import React, { useState } from 'react';
import { Search } from 'lucide-react';

const SearchBar = ({ onSearch }) => {
  const [searchTerm, setSearchTerm] = useState('');

  const handleSearch = (e) => {
    const value = e.target.value;
    setSearchTerm(value);
    if (onSearch && typeof onSearch === 'function') {
      onSearch(value);
    }
  };

  return (
    <div className="max-w-2xl mx-auto mb-8 relative">
      <div className="relative">
        <input
          type="text"
          value={searchTerm}
          placeholder="Search movies by title, genre, cast, or keywords..."
          onChange={handleSearch}
          className="w-full px-4 py-3 pl-12 rounded-lg bg-gray-800 text-white border border-gray-700 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500"
        />
        <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
      </div>
    </div>
  );
};

export default SearchBar;