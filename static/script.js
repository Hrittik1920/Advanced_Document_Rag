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

    // --- 2. Listen for streaming chunks from the server ---
    let currentBotMessageElement = null;
    
    socket.on("chat_stream_chunk", (data) => {
        if (!currentBotMessageElement) {
            hideLoadingIndicator();
            currentBotMessageElement = createBotMessageElement("");
            chatBox.appendChild(currentBotMessageElement);
        }
        
        const chunk = data.chunk || "";
        appendChunkToBotMessage(currentBotMessageElement, chunk);
        scrollToBottom();
    });
    
    socket.on("chat_stream_end", (data) => {
        if (currentBotMessageElement) {
            currentBotMessageElement = null;
        }
        isRequestInProgress = false;
        sendButton.disabled = false;
        userInput.focus();
    });
    
    // --- Legacy support for non-streaming responses ---
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
     * Creates a bot message element.
     */
    function createBotMessageElement(content) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", "bot-message");
        
        const container = document.createElement('div');
        container.className = 'formatted-response';
        container.dataset.rawContent = "";
        messageElement.appendChild(container);
        
        return messageElement;
    }
    
    /**
     * Appends a chunk to the bot message and updates the formatted display.
     */
    function appendChunkToBotMessage(messageElement, chunk) {
        const container = messageElement.querySelector('.formatted-response');
        if (container) {
            const currentRawContent = container.dataset.rawContent || "";
            const newRawContent = currentRawContent + chunk;
            container.dataset.rawContent = newRawContent;
            
            // Update the displayed markdown
            const sanitizedText = newRawContent.replace(/\\n/g, '\n').replace(/\\t/g, '\t');
            container.innerHTML = marked.parse(sanitizedText);
        }
    }
    
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


