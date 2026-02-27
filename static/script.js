document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const sendButton = document.getElementById("send-button");

    let isRequestInProgress = false;

    // --- 1. Connect to the Socket.IO server ---
    const socket = io("http://178.63.15.33:8000")

    // --- Event listener for successful connection ---
    socket.on("connect", () => {
        console.log("✅ Successfully connected to the server with SID:", socket.id);
    });

    // --- 2. Listen for 'chat_response' from the server ---
    socket.on("chat_response", (data) => {
        hideLoadingIndicator();
        appendMessage(data.content, "bot-message");
        isRequestInProgress = false;
        sendButton.disabled = false;
        userInput.focus();
    });
    
    // --- Event listener for potential server errors ---
    socket.on("error", (data) => {
        console.error("Server Error:", data.message);
        hideLoadingIndicator();
        appendMessage(`Sorry, an error occurred: ${data.message}`, "bot-message");
        isRequestInProgress = false;
        sendButton.disabled = false;
    });

    // --- Function to handle sending a message ---
    const handleSendMessage = () => {
        if (isRequestInProgress) return;
        const userMessage = userInput.value.trim();

        if (userMessage) {
            isRequestInProgress = true;
            sendButton.disabled = true;
            appendMessage(userMessage, "user-message");
            userInput.value = "";
            showLoadingIndicator();
            socket.emit("chat_request", { message: userMessage });
        }
    };

    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        handleSendMessage();
    });

    /**
     * Appends a message to the chat box.
     */
    function appendMessage(content, className) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", className);

        if (className === 'bot-message') {
            // Bot messages are formatted to handle Markdown
            const formattedContent = formatBotResponse(content);
            messageElement.appendChild(formattedContent);
        } else {
            // User messages are plain text
            const p = document.createElement("p");
            p.textContent = content;
            messageElement.appendChild(p);
        }
        
        chatBox.appendChild(messageElement);
        scrollToBottom();
    }

    // static/script.js
    // static/script.js

    /**
     * Parses the bot's entire Markdown response into a single, formatted HTML block.
     * @param {string} markdownText - The raw markdown string from the bot.
     * @returns {HTMLElement} - A div element containing the formatted HTML.
     */
    function formatBotResponse(markdownText) {
        const container = document.createElement('div');
        container.className = 'formatted-response';

        // Sanitize for real newlines and tabs
        const sanitizedText = markdownText.replace(/\\n/g, '\n').replace(/\\t/g, '\t');

        // Use the marked.js library to convert the entire response into HTML
        container.innerHTML = marked.parse(sanitizedText);
        
        return container;
    }
    

    // --- Helper functions ---
    function showLoadingIndicator() {
        const loadingElement = document.createElement("div");
        loadingElement.classList.add("message", "bot-message", "loading-indicator");
        loadingElement.id = "loading-indicator";
        loadingElement.innerHTML = `<span></span><span></span><span></span>`;
        chatBox.appendChild(loadingElement);
        scrollToBottom();
    }

    function hideLoadingIndicator() {
        const loadingElement = document.getElementById("loading-indicator");
        if (loadingElement) {
            loadingElement.remove();
        }
    }

    function scrollToBottom() {
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});


