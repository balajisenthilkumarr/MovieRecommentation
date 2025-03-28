// src/pages/LoginPage.js
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import axios from "axios";

const LoginPage = () => {
  const [username, setUsername] = useState("");
  const [userId, setUserId] = useState(""); // Add userId state
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      // Validate userId
      const parsedUserId = parseInt(userId);
      if (isNaN(parsedUserId) || parsedUserId <= 0) {
        setError("Please enter a valid user ID (positive integer)");
        return;
      }

      // Store the userId in localStorage
      localStorage.setItem("userId", parsedUserId.toString());
      console.log(`Logging in with username: ${username}, userId: ${parsedUserId}`);

      // Save the user to the backend
      await axios.post("http://localhost:5000/api/register_user", {
        userId: parsedUserId,
        username,
      });

      // Navigate to the OnboardingPage after successful login
      navigate("/preference"); // Changed to /onboarding for consistency
    } catch (err) {
      setError("Failed to log in");
      console.error("Login error:", err);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-900 to-black flex items-center justify-center">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="bg-gray-800 p-8 rounded-lg shadow-lg max-w-sm w-full"
      >
        <h2 className="text-3xl font-bold text-white mb-6 text-center">Login</h2>
        {error && (
          <div className="bg-red-500/10 border border-red-500/20 text-red-500 text-center p-4 rounded-lg mb-6">
            {error}
          </div>
        )}
        <form onSubmit={handleLogin}>
          <div className="mb-4">
            <label className="block text-gray-300 mb-2">Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full bg-gray-700 text-white p-2 rounded"
              placeholder="Enter username (e.g., user1)"
              required
            />
          </div>
          <div className="mb-4">
            <label className="block text-gray-300 mb-2">User ID</label>
            <input
              type="number"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              className="w-full bg-gray-700 text-white p-2 rounded"
              placeholder="Enter your user ID (e.g., 1)"
              required
            />
          </div>
          <button
            type="submit"
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg transition-colors"
          >
            Login
          </button>
        </form>
      </motion.div>
    </div>
  );
};

export default LoginPage;