import { useState, useEffect, useRef } from 'react';
import { Send, Film, X, Maximize2, Minimize2, ChevronUp, ChevronDown } from 'lucide-react';

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
  const [size, setSize] = useState("large"); // Changed default size to large
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  // Focus input when chat opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  
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

  // Get size dimensions
  const getSizeStyles = () => {
    switch(size) {
      case "small": 
        return { width: "w-72", height: "h-64" };
      case "large": 
        return { width: "w-96", height: "h-96" };
      default: // medium
        return { width: "w-80", height: "h-72" };
    }
  };

  const { width, height } = getSizeStyles();

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-8 right-8 bg-blue-600 text-white p-4 rounded-full shadow-lg hover:bg-blue-700 transition-colors"
      >
        <Film className="w-6 h-6" />
      </button>

      {isOpen && (
        <div
          className={`fixed bottom-24 right-8 ${width} bg-gray-800 rounded-lg shadow-xl overflow-hidden`}
        >
          <div className="flex justify-between items-center p-4 bg-gray-900">
            <div className="flex items-center gap-2">
              <Film className="w-5 h-5 text-blue-400" />
              <h3 className="text-white font-semibold">Movie Chatbot</h3>
            </div>
            <div className="flex gap-2">
              <button 
                onClick={() => setSize(size === "medium" ? "large" : "medium")} 
                className="text-gray-400 hover:text-white"
              >
                {size === "large" ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
              </button>
              <button onClick={() => setIsOpen(false)} className="text-gray-400 hover:text-white">
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>
          
          <div className={`${height} overflow-y-auto p-4 space-y-4 bg-gray-800`}>
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
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-700 text-gray-200 p-3 rounded-lg flex items-center gap-2">
                  <span className="animate-pulse">•</span>
                  <span className="animate-pulse delay-100">•</span>
                  <span className="animate-pulse delay-200">•</span>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
          
          <div className="p-4 border-t border-gray-700 bg-gray-900">
            <div className="flex gap-2">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && sendMessage()}
                placeholder="Ask about movies..."
                className="w-full p-2 rounded-lg bg-gray-800 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button 
                onClick={sendMessage}
                disabled={!input.trim()}
                className="p-2 bg-blue-600 rounded-lg text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}