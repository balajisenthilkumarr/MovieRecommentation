// src/pages/AdminDashboard.js
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { User, Trash2, LogOut } from "lucide-react";

const AdminDashboard = () => {
  // Mocked user data (instead of fetching from API)
  const [users, setUsers] = useState([
    { id: 1, username: "admin", role: "admin" },
    { id: 2, username: "user1", role: "user" },
    { id: 3, username: "user2", role: "user" },
  ]);
  
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const navigate = useNavigate();





  const user = JSON.parse(localStorage.getItem("user"));
  useEffect(() => {
    // Comment out the redirect for now to see the dashboard
    // if (!user || user.role !== "admin") {
    //   navigate("/AdminDasboard");
    // }
  }, [user, navigate]);

  // Remove the API fetch (since we're using mocked data)
  // useEffect(() => {
  //   const fetchUsers = async () => {
  //     try {
  //       const response = await axios.get("http://localhost:5000/api/users");
  //       setUsers(response.data);
  //     } catch (err) {
  //       setError("Failed to fetch users");
  //     }
  //   };
  //   fetchUsers();
  // }, []);

  // Mock the delete functionality (without API call)
  const handleDelete = (id) => {
    if (window.confirm("Are you sure you want to delete this user?")) {
      try {
        // Simulate a successful delete by filtering the user out
        setUsers(users.filter((user) => user.id !== id));
        setSuccess("User deleted successfully");
        setError("");
      } catch (err) {
        setError("Failed to delete user");
        setSuccess("");
      }
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("user");
    localStorage.removeItem("userId");
    navigate("/login"); // Fixed the typo from "/AdminDashboart" to "/login"
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-black text-white">
      {/* Navbar */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-gray-900/95 backdrop-blur-md shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <User className="w-8 h-8 text-blue-500" />
              <span className="text-xl font-bold text-white">Admin Dashboard</span>
            </div>
            <button
              onClick={handleLogout}
              className="flex items-center space-x-2 text-gray-400 hover:text-white transition-colors"
            >
              <LogOut className="w-5 h-5" />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="container mx-auto px-2 pt-24 pb-8">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-8"
        >
          <h2 className="text-3xl font-bold mb-6">Manage Users</h2>
          {error && (
            <div className="bg-red-500/10 border border-red-500/20 text-red-500 text-center p-3 rounded-lg mb-4">
              {error}
            </div>
          )}
          {success && (
            <div className="bg-green-500/10 border border-green-500/20 text-green-500 text-center p-3 rounded-lg mb-4">
              {success}
            </div>
          )}
          <div className="bg-gray-800 rounded-lg shadow-lg p-6">
            {users.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-left">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="py-3 px-4">ID</th>
                      <th className="py-3 px-4">Username</th>
                      <th className="py-3 px-4">Role</th>
                      <th className="py-3 px-4">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {users.map((user) => (
                      <motion.tr
                        key={user.id}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.3 }}
                        className="border-b border-gray-700 hover:bg-gray-700 transition-colors"
                      >
                        <td className="py-3 px-4">{user.id}</td>
                        <td className="py-3 px-4">{user.username}</td>
                        <td className="py-3 px-4">{user.role}</td>
                        <td className="py-3 px-4">
                          <button
                            onClick={() => handleDelete(user.id)}
                            className="text-red-500 hover:text-red-400 transition-colors"
                          >
                            <Trash2 className="w-5 h-5" />
                          </button>
                        </td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-gray-400 text-center">No users found.</p>
            )}
          </div>
        </motion.div>
      </main>
    </div>
  );
};

export default AdminDashboard;