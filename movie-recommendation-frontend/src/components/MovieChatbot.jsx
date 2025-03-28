import { useState, useEffect, useRef } from 'react';
import { Send, Film, X, Maximize2, Minimize2, ChevronUp, ChevronDown } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion'; // Added missing import

// Define a minimal addEmojisToResponse function to avoid errors
const addEmojisToResponse = (text) => {
  return text; // Simply return the text as-is to avoid errors
};

export default function MovieChatbot() {
  const [messages, setMessages] = useState([
    { text: "Hello! 👋 I'm your movie assistant. Ask me for recommendations, information about specific movies, or discover movies by genre! 🎬", sender: "bot" }
  ]);
  const [input, setInput] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [size, setSize] = useState("medium"); // Options: small, medium, large
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { text: input, sender: "user" };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:5000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input })
      });
      
      const data = await response.json();
      // Add emojis to certain keywords in the response
      const enhancedResponse = addEmojisToResponse(data.response);
      setMessages((prev) => [...prev, { text: enhancedResponse, sender: "bot" }]);
    } catch (error) {
      console.error("Error fetching response:", error);
      setMessages((prev) => [...prev, { 
        text: "Sorry, I couldn't connect to the movie database. Please try again later. 😔", 
        sender: "bot" 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <motion.button
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        onClick={() => setIsOpen(true)}
        className="fixed bottom-8 right-8 bg-blue-600 text-white p-4 rounded-full shadow-lg hover:bg-blue-700 transition-colors"
      >
        <Film className="w-6 h-6" /> {/* Replaced MessageCircle with Film */}
      </motion.button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="fixed bottom-24 right-8 w-80 bg-gray-800 rounded-lg shadow-xl overflow-hidden"
          >
            <div className="flex justify-between items-center p-4 bg-gray-900">
              <h3 className="text-white font-semibold">Movie Chatbot</h3>
              <button onClick={() => setIsOpen(false)} className="text-gray-400 hover:text-white">
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="h-64 overflow-y-auto p-4 space-y-4">
              {messages.map((msg, index) => (
                <div
                  key={index}
                  className={`flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-xs p-3 rounded-lg ${
                      msg.sender === "user"
                        ? "bg-blue-600 text-white"
                        : "bg-gray-700 text-gray-200"
                    }`}
                  >
                    {msg.text}
                  </div>
                </div>
              ))}
              {isLoading && ( // Fixed: Changed 'loading' to 'isLoading'
                <div className="flex justify-start">
                  <div className="bg-gray-700 text-gray-200 p-3 rounded-lg">
                    Thinking...
                  </div>
                </div>
              )}
            </div>
            <div className="p-4 border-t border-gray-700">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && sendMessage()} // Fixed: Changed 'handleSend' to 'sendMessage'
                placeholder="Ask about movies..."
                className="w-full p-2 rounded-lg bg-gray-900 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};