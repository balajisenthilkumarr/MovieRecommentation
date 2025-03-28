// src/components/Navbar.jsx
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Film, User, Menu, X, Bell } from "lucide-react";

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
          {/* Logo and Brand */}
          <div className="flex items-center space-x-4">
            <Film className="w-8 h-8 text-blue-500" />
            <span className="text-xl font-bold text-white">Recommendation System</span>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            <NavLink icon={<Film className="w-5 h-5" />} text="Movies" active />

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
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden bg-gray-900/95 backdrop-blur-md"
          >
            <div className="px-4 pt-2 pb-3 space-y-1">
              <MobileNavLink text="Movies" active />
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

export default Navbar;