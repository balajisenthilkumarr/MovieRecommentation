// src/pages/LoginPage.js
import { useState } from "react";
import { useNavigate, Link } from "react-router-dom"; // Add Link for navigation
import { motion } from "framer-motion";
import axios from "axios";

const LoginPage = () => {
  const [username, setUsername] = useState("");
  const [userId, setUserId] = useState("");
  const [password, setPassword] = useState("");
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

      // Authenticate the user using the /api/login endpoint
      const response = await axios.post("http://localhost:5000/api/login", {
        username,
        password,
      });

      // Store user data in localStorage
      const user = response.data.user;
      localStorage.setItem("userId", parsedUserId.toString());
      localStorage.setItem("user", JSON.stringify(user));

      // Redirect based on role
      if (user.role === "admin") {
        navigate("/admin");
      } else {
        navigate("/preference");
      }
    } catch (err) {
      setError(err.response?.data?.error || "Failed to log in");
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
          <div className="mb-4">
            <label className="block text-gray-300 mb-2">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-gray-700 text-white p-2 rounded"
              placeholder="Enter your password"
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
        <div className="mt-4 text-center">
          <p className="text-gray-400">
            Don't have an account?{" "}
            <Link to="/register" className="text-blue-500 hover:underline">
              Register here
            </Link>
          </p>
        </div>
      </motion.div>
    </div>
  );
};

export default LoginPage;