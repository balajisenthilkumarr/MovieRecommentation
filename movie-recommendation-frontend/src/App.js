import { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import UserOnboarding from "./pages/UserOnboarding ";
import HomePage from "./pages/HomePage";
import MovieDetails from "./pages/MovieDetails";
import LoginPage from "./pages/LoginPage";
import AdminDashboard from "./pages/AdminDashboard"
import RegisterPage from "./pages/RegisterPage";
import axios from "axios";

function App() {
  const [userPreferences, setUserPreferences] = useState(null);

  useEffect(() => {
    const fetchPreferences = async () => {
      try {
        const userId = 1; // Replace with actual user ID from auth
        const response = await axios.get(`http://localhost:5000/api/user_preferences/${userId}`);
        setUserPreferences(response.data.preferences);
      } catch (err) {
        console.error("Failed to fetch user preferences:", err);
      }
    };
    fetchPreferences();
  }, []);

  return (
    <Router>
      <Routes>
        <Route
          path="/preference"
          element={<UserOnboarding setUserPreferences={setUserPreferences} />}
        />
        <Route
          path="/"
          element={<HomePage userPreferences={userPreferences} />}
        />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/movie/:id" element={<MovieDetails />} />
        <Route path="/admin" element={<AdminDashboard />} />
      </Routes>
    </Router>
  );
}

export default App;