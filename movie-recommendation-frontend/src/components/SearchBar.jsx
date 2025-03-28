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


import { useState } from "react";
import { Search } from "lucide-react";

const SearchBar = ({ onSearch }) => {
  const [query, setQuery] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    onSearch(query);
  };

  return (
    <form onSubmit={handleSubmit} className="relative w-full">
      <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
        <Search className="w-5 h-5 text-gray-400" />
      </div>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search movies, genres..."
        className="w-full pl-12 pr-4 py-3 rounded-lg bg-gray-800/50 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 border border-gray-700"
      />
    </form>
  );
};

export default SearchBar;