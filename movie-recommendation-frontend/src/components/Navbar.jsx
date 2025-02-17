// src/components/Navbar.jsx
import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <nav className="bg-gray-800 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center">
            {/* <span className="text-white text-xl font-bold">Recommendation System </span> */}
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;