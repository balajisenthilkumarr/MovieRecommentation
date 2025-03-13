import { useState, useEffect, useRef } from 'react';
import { Send, Film, X, Maximize2, Minimize2, ChevronUp, ChevronDown } from 'lucide-react';

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

  // Add emojis to certain keywords in the responses
  const addEmojisToResponse = (text) => {
    if (typeof text !== 'string') return text;
    
    // Add emojis based on keywords
    return text
      .replace(/comedy/gi, "comedy 😄")
      .replace(/horror/gi, "horror 😱")
      .replace(/action/gi, "action 💥")
      .replace(/drama/gi, "drama 😢")
      .replace(/sci-fi/gi, "sci-fi 🚀")
      .replace(/romance/gi, "romance ❤️")
      .replace(/thriller/gi, "thriller 😰")
      .replace(/recommend/gi, "recommend 👍")
      .replace(/best/gi, "best ⭐")
      .replace(/watch/gi, "watch 👀")
      .replace(/great/gi, "great 🔥")
      .replace(/award/gi, "award 🏆");
  };

  // Calculate sizes based on current size state
  const getSizes = () => {
    switch(size) {
      case "small":
        return {
          width: "w-72 md:w-80",
          height: "h-80"
        };
      case "large":
        return {
          width: "w-96 md:w-120",
          height: "h-112"
        };
      default: // medium
        return {
          width: "w-80 md:w-96",
          height: "h-96"
        };
    }
  };

  // Cycle through sizes
  const cycleSize = () => {
    const sizes = ["small", "medium", "large"];
    const currentIndex = sizes.indexOf(size);
    const nextIndex = (currentIndex + 1) % sizes.length;
    setSize(sizes[nextIndex]);
  };

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Focus input field when chat opens
  useEffect(() => {
    if (isOpen && !isMinimized) {
      inputRef.current?.focus();
    }
  }, [isOpen, isMinimized]);

  // Format messages with movie suggestions
  const formatMessage = (text) => {
    if (typeof text !== 'string') return text;
    
    // Simple regex to detect movie titles with year pattern
    const moviePattern = /([A-Za-z0-9\s:,\-&!'.]+)(\([0-9]{4}\))/g;
    const parts = [];
    let lastIndex = 0;
    let match;
    
    while ((match = moviePattern.exec(text)) !== null) {
      // Add text before the match
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }
      
      // Add the movie title with styling
      parts.push(
        <span key={match.index} className="px-1 mx-1 bg-blue-700 rounded text-blue-100 font-medium">
          {match[0]}
        </span>
      );
      
      lastIndex = match.index + match[0].length;
    }
    
    // Add remaining text
    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }
    
    return parts.length > 0 ? parts : text;
  };

  // Get dimensions based on size setting
  const { width, height } = getSizes();

  // Render the chatbot UI
  return (
    <>
      {/* Chat toggle button (always visible) */}
      {!isOpen && (
        <button 
          onClick={() => {setIsOpen(true); setIsMinimized(false);}}
          className="fixed bottom-6 right-6 bg-blue-600 text-white p-4 rounded-full shadow-lg hover:bg-blue-700 transition-all duration-300 z-50 flex items-center justify-center"
          aria-label="Open movie chat"
        >
          <Film size={24} />
        </button>
      )}

      {/* Chat interface */}
      {isOpen && (
        <div className={`fixed ${isMinimized ? 'bottom-6 right-6 h-auto w-auto' : `bottom-0 right-0 md:bottom-6 md:right-6 ${height} ${width}`} bg-gray-900 rounded-lg shadow-2xl overflow-hidden transition-all duration-300 z-50 flex flex-col border border-blue-700`}>
          {/* Chat header */}
          <div className="bg-blue-600 px-4 py-3 flex justify-between items-center">
            {!isMinimized && (
              <div className="flex items-center text-white font-medium">
                <Film size={20} className="mr-2" />
                <span>Movie Assistant 🎬</span>
              </div>
            )}
            <div className="flex items-center">
              {/* Size toggle button (only visible when not minimized) */}
              {!isMinimized && (
                <button 
                  onClick={cycleSize} 
                  className="text-white p-1 hover:bg-blue-700 rounded mr-1"
                  title={`Current size: ${size}. Click to change size.`}
                >
                  {size === "small" ? (
                    <ChevronUp size={18} />
                  ) : size === "large" ? (
                    <ChevronDown size={18} />
                  ) : (
                    <div className="flex">
                      <ChevronUp size={18} className="mr-1" />
                      <ChevronDown size={18} />
                    </div>
                  )}
                </button>
              )}
              <button 
                onClick={() => setIsMinimized(!isMinimized)} 
                className="text-white p-1 hover:bg-blue-700 rounded"
                title={isMinimized ? "Maximize" : "Minimize"}
              >
                {isMinimized ? <Maximize2 size={18} /> : <Minimize2 size={18} />}
              </button>
              <button 
                onClick={() => setIsOpen(false)} 
                className="text-white p-1 hover:bg-blue-700 rounded ml-1"
                title="Close"
              >
                <X size={18} />
              </button>
            </div>
          </div>

          {/* Chat messages area (hidden when minimized) */}
          {!isMinimized && (
            <>
              <div className="flex-1 overflow-y-auto p-4 bg-gray-800">
                {messages.map((msg, index) => (
                  <div 
                    key={index} 
                    className={`mb-3 p-3 rounded-lg max-w-4/5 ${
                      msg.sender === "user" 
                        ? "ml-auto bg-blue-600 text-white" 
                        : "bg-gray-700 text-gray-100"
                    } shadow-md`}
                  >
                    {formatMessage(msg.text)}
                  </div>
                ))}
                {isLoading && (
                  <div className="bg-gray-700 text-gray-100 p-3 rounded-lg w-fit flex items-center">
                    <div className="mr-2">Thinking</div>
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: "0ms" }}></div>
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: "150ms" }}></div>
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: "300ms" }}></div>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Input area */}
              <div className="p-3 bg-gray-900 border-t border-gray-700 flex items-center">
                <input
                  ref={inputRef}
                  className="flex-1 p-2 bg-gray-800 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder-gray-400"
                  placeholder="Ask about movies... 🍿"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
                />
                <button 
                  className={`ml-3 p-2 rounded-lg transition-colors ${
                    input.trim() 
                      ? "bg-blue-600 hover:bg-blue-700 transform hover:scale-105" 
                      : "bg-gray-700 cursor-not-allowed"
                  }`} 
                  onClick={sendMessage}
                  disabled={!input.trim() || isLoading}
                  title="Send message"
                >
                  <Send size={20} className="text-white" />
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </>
  );
}