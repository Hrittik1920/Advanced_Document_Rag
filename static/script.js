document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const sendButton = document.getElementById("send-button");

    let isRequestInProgress = false;

    // --- 1. Connect to the Socket.IO server ---
    const socket = io("http://127.0.0.1:8000")

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
    // /**
    //  * Parses the bot's response into a conversational part and a collapsible details section.
    //  * @param {string} markdownText - The raw markdown string from the bot.
    //  * @returns {HTMLElement} - A div element containing the formatted HTML.
    //  */
    // function formatBotResponse(markdownText) {
    //     const container = document.createElement('div');
    //     container.className = 'formatted-response';

    //     // Sanitize the input for real newlines
    //     const sanitizedText = markdownText.replace(/\\n/g, '\n').replace(/\\t/g, '\t');

    //     // Split the response into the conversational part and the details part
    //     const parts = sanitizedText.split(/\n---\n/);
        
    //     const conversationalPart = parts[0];
    //     const detailsPart = parts.length > 1 ? parts[1] : '';

    //     // Render the main conversational answer
    //     // Use marked.parse to handle any markdown in the conversational part
    //     const conversationalDiv = document.createElement('div');
    //     conversationalDiv.innerHTML = marked.parse(conversationalPart);
    //     container.appendChild(conversationalDiv);

    //     // If there is a details part, create a collapsible element
    //     if (detailsPart.trim()) {
    //         const detailsElement = document.createElement('details');
            
    //         const summaryElement = document.createElement('summary');
    //         summaryElement.textContent = 'Show Details';
            
    //         const contentElement = document.createElement('div');
    //         contentElement.className = 'details-content';
    //         // Use marked.parse to render the markdown for the details (lists, headings)
    //         contentElement.innerHTML = marked.parse(detailsPart);

    //         detailsElement.appendChild(summaryElement);
    //         detailsElement.appendChild(contentElement);
    //         container.appendChild(detailsElement);
    //     }
        
    //     return container;
    // }
    // // --- THIS IS THE CORRECT, SIMPLIFIED FUNCTION ---
    // /**
    //  * Parses Markdown from the bot into HTML using the marked.js library.
    //  * @param {string} markdownText - The raw markdown string from the bot.
    //  * @returns {HTMLElement} - A div element containing the formatted HTML.
    //  */
    // function formatBotResponse(markdownText) {
    //     const container = document.createElement('div');
    //     container.className = 'formatted-response';

    //     // 1. Sanitize the input: Convert literal '\\n' strings into real newlines.
    //     const sanitizedText = markdownText.replace(/\\n/g, '\n').replace(/\\t/g, '\t');

    //     // 2. Use the marked.js library to safely convert Markdown into HTML.
    //     // This will correctly handle lists, bolding, and everything else.
    //     container.innerHTML = marked.parse(sanitizedText);
        
    //     return container;
    // }

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


// document.addEventListener("DOMContentLoaded", () => {
//     const chatForm = document.getElementById("chat-form");
//     const userInput = document.getElementById("user-input");
//     const chatBox = document.getElementById("chat-box");
//     const sendButton = document.getElementById("send-button");

//     let isRequestInProgress = false;

//     // --- 1. Connect to the Socket.IO server ---
//     // This replaces the API_BASE_URL constant.
//     const socket = io("http://127.0.0.1:8000");

//     // --- Event listener for successful connection ---
//     socket.on("connect", () => {
//         console.log("✅ Successfully connected to the server with SID:", socket.id);
//     });

//     // --- 2. Listen for 'chat_response' from the server ---
//     // This replaces the 'await response.json()' part of the fetch call.
//     socket.on("chat_response", (data) => {
//         // When the server sends a response, this function runs.
//         hideLoadingIndicator(); // Remove the loading animation
//         appendMessage(data.content, "bot-message"); // Add the bot's message to the chat

//         // Reset the state to allow the user to send another message.
//         isRequestInProgress = false;
//         sendButton.disabled = false;
//         userInput.focus();
//     });
    
//     // --- Event listener for potential server errors ---
//     socket.on("error", (data) => {
//         console.error("Server Error:", data.message);
//         hideLoadingIndicator();
//         appendMessage(`Sorry, an error occurred: ${data.message}`, "bot-message");

//         // Reset the state.
//         isRequestInProgress = false;
//         sendButton.disabled = false;
//     });

//     // This function now uses Socket.IO to send messages.
//     const handleSendMessage = () => {
//         if (isRequestInProgress) return;
//         const userMessage = userInput.value.trim();

//         if (userMessage) {
//             isRequestInProgress = true;
//             sendButton.disabled = true;
//             appendMessage(userMessage, "user-message");
//             userInput.value = "";
//             showLoadingIndicator();

//             // --- 3. Send message using socket.emit() instead of fetch() ---
//             // The event is named 'chat_request' to match the Python backend.
//             socket.emit("chat_request", {
//                 message: userMessage
//             });
//         }
//     };

//     chatForm.addEventListener("submit", (e) => {
//         e.preventDefault();
//         handleSendMessage();
//     });

//     /**
//      * Appends a message to the chat box, formatting it if it's a special response.
//      * @param {string} content - The message content.
//      * @param {string} className - The CSS class for the message.
//      */
//     function appendMessage(content, className) {
//         const messageElement = document.createElement("div");
//         messageElement.classList.add("message", className);

//         if (className === 'bot-message') {
//             const formattedContent = formatBotResponse(content);
//             messageElement.appendChild(formattedContent);
//         } else {
//             const p = document.createElement("p");
//             p.textContent = content;
//             messageElement.appendChild(p);
//         }
        
//         chatBox.appendChild(messageElement);
//         scrollToBottom();
//     }

//     /**
//      * Parses different markdown formats from the bot into beautiful HTML.
//      * This function remains unchanged as its job is purely for display.
//      * @param {string} markdownText - The raw markdown string from the bot.
//      * @returns {HTMLElement} - A div element containing the formatted HTML.
//      */
//     function formatBotResponse(markdownText) {
//         // --- This entire complex formatting function remains exactly the same. ---
//         // --- No changes are needed here. ---
//         try {
//             const container = document.createElement('div');
//             container.className = 'formatted-response';
//             const text = markdownText.replace(/\*\*/g, '');

//             if (text.includes('Assistant Response:') && /\n\d+\.\s*\w+:/.test(text)) {
//                 // ... (your existing address formatting logic) ...
//                 const sections = text.split('Assistant Response:');
//                 if (sections[0] && sections[0].includes('Key Takeaways:')) { /* ... */ }
//                 if (sections[1]) { /* ... */ }
//                 return container;
//             }

//             const lines = text.split('\n');
//             let currentList = null;
//             lines.forEach(line => {
//                 const trimmedLine = line.trim();
//                 if (!trimmedLine) {
//                     if (currentList) { container.appendChild(currentList); currentList = null; }
//                     return;
//                 }
//                 const olMatch = trimmedLine.match(/^(\d+)\.\s+(.*)/);
//                 const ulMatch = trimmedLine.match(/^[\*\+]\s+(.*)/);
//                 if (olMatch) {
//                     if (!currentList || currentList.tagName !== 'OL') { if (currentList) container.appendChild(currentList); currentList = document.createElement('ol'); container.appendChild(currentList); }
//                     const listItem = document.createElement('li'); listItem.textContent = olMatch[2]; currentList.appendChild(listItem);
//                 } else if (ulMatch) {
//                     if (!currentList || currentList.tagName !== 'UL') { if (currentList) container.appendChild(currentList); currentList = document.createElement('ul'); container.appendChild(currentList); }
//                     const listItem = document.createElement('li'); listItem.textContent = ulMatch[1]; currentList.appendChild(listItem);
//                 } else {
//                     if (currentList) { container.appendChild(currentList); currentList = null; }
//                     const p = document.createElement('p'); p.textContent = trimmedLine; container.appendChild(p);
//                 }
//             });
//             if (currentList) container.appendChild(currentList);
//             return container;

//         } catch (formatError) {
//             console.error("Failed to format bot response:", formatError);
//             const fallbackContainer = document.createElement('div');
//             const p = document.createElement('p');
//             p.textContent = markdownText;
//             fallbackContainer.appendChild(p);
//             return fallbackContainer;
//         }
//     }
//     // /**
//     //  * Parses different markdown formats from the bot into beautiful HTML
//     //  * using the marked.js library.
//     //  * @param {string} markdownText - The raw markdown string from the bot.
//     //  * @returns {HTMLElement} - A div element containing the formatted HTML.
//     //  */
//     // function formatBotResponse(markdownText) {
//     //     const container = document.createElement('div');
//     //     container.className = 'formatted-response';

//     //     // Use the marked library to safely convert the bot's markdown into HTML
//     //     // This will correctly handle lists, bolding, and everything else.
//     //     container.innerHTML = marked.parse(markdownText);
        
//     //     return container;
//     // }

//     // --- The remaining helper functions are also unchanged. ---
//     function showLoadingIndicator() {
//         const loadingElement = document.createElement("div");
//         loadingElement.classList.add("message", "bot-message", "loading-indicator");
//         loadingElement.id = "loading-indicator";
//         loadingElement.innerHTML = `<span></span><span></span><span></span>`;
//         chatBox.appendChild(loadingElement);
//         scrollToBottom();
//     }

//     function hideLoadingIndicator() {
//         const loadingElement = document.getElementById("loading-indicator");
//         if (loadingElement) {
//             loadingElement.remove();
//         }
//     }

//     function scrollToBottom() {
//         chatBox.scrollTop = chatBox.scrollHeight;
//     }
// });

// document.addEventListener("DOMContentLoaded", () => {
//     const chatForm = document.getElementById("chat-form");
//     const userInput = document.getElementById("user-input");
//     const chatBox = document.getElementById("chat-box");
//     const sendButton = document.getElementById("send-button");

//     // Define the absolute base URL for your FastAPI backend
//     const API_BASE_URL = "http://127.0.0.1:8000"; // Or whatever port your backend is running on

//     const modelName = "rag-chat-model";
//     let isRequestInProgress = false;

//     const handleSendMessage = async () => {
//         if (isRequestInProgress) return;
//         const userMessage = userInput.value.trim();

//         if (userMessage) {
//             isRequestInProgress = true;
//             sendButton.disabled = true;
//             appendMessage(userMessage, "user-message");
//             const currentInput = userInput.value;
//             userInput.value = "";
//             showLoadingIndicator();

//             try {
//                 const payload = {
//                     model: modelName,
//                     messages: [{ role: "user", content: userMessage }]
//                 };
//                 const response = await fetch(`${API_BASE_URL}/v1/chat/completions`, {
//                     method: "POST",
//                     headers: { "Content-Type": "application/json" },
//                     body: JSON.stringify(payload),
//                 });

//                 if (!response.ok) {
//                     const errorBody = await response.text();
//                     console.error("Response Error Body:", errorBody);
//                     throw new Error(`HTTP error! status: ${response.status}`);
//                 }

//                 const data = await response.json();
//                 const botResponse = data.choices[0].message.content;
//                 appendMessage(botResponse, "bot-message");

//             } catch (error) {
//                 console.error("Error:", error);
//                 userInput.value = currentInput;
//                 appendMessage("Sorry, something went wrong. Please check the console for details.", "bot-message");
//             } finally {
//                 // This block is crucial and will now always be reached.
//                 hideLoadingIndicator();
//                 isRequestInProgress = false;
//                 sendButton.disabled = false;
//                 userInput.focus();
//             }
//         }
//     };

//     chatForm.addEventListener("submit", (e) => {
//         e.preventDefault();
//         handleSendMessage();
//     });

//     /**
//      * Appends a message to the chat box, formatting it if it's a special response.
//      * @param {string} content - The message content.
//      * @param {string} className - The CSS class for the message.
//      */
//     function appendMessage(content, className) {
//         const messageElement = document.createElement("div");
//         messageElement.classList.add("message", className);

//         // For bot messages, always try to format them.
//         if (className === 'bot-message') {
//             const formattedContent = formatBotResponse(content);
//             messageElement.appendChild(formattedContent);
//         } else {
//             // User messages are always plain text.
//             const p = document.createElement("p");
//             p.textContent = content;
//             messageElement.appendChild(p);
//         }
        
//         chatBox.appendChild(messageElement);
//         scrollToBottom();
//     }

//     /**
//      * Parses different markdown formats from the bot into beautiful HTML.
//      * This function is now wrapped in a try/catch to prevent it from
//      * breaking the main application flow.
//      * @param {string} markdownText - The raw markdown string from the bot.
//      * @returns {HTMLElement} - A div element containing the formatted HTML.
//      */
//     function formatBotResponse(markdownText) {
//         try {
//             const container = document.createElement('div');
//             container.className = 'formatted-response';
//             const text = markdownText.replace(/\*\*/g, ''); // Clean bold markers

//             // --- Rule 1: Detect and format the detailed address layout ---
//             if (text.includes('Assistant Response:') && /\n\d+\.\s*\w+:/.test(text)) {
//                 const sections = text.split('Assistant Response:');
                
//                 // Part 1: Key Takeaways (if it exists)
//                 if (sections[0] && sections[0].includes('Key Takeaways:')) {
//                     const takeawaysContent = sections[0].replace('Key Takeaways:', '').trim();
//                     const takeawaysHeader = document.createElement('h3');
//                     takeawaysHeader.textContent = 'Key Takeaways';
//                     container.appendChild(takeawaysHeader);
//                     const takeawaysList = document.createElement('ul');
//                     takeawaysContent.split('\n').forEach(line => {
//                         const trimmedLine = line.trim();
//                         if (trimmedLine.startsWith('*') || trimmedLine.startsWith('+')) {
//                             const listItem = document.createElement('li');
//                             listItem.innerHTML = trimmedLine.substring(1).trim().replace(/(\(.*?\))/g, '<em>$1</em>');
//                             takeawaysList.appendChild(listItem);
//                         }
//                     });
//                     container.appendChild(takeawaysList);
//                 }

//                 // Part 2: Addresses
//                 if (sections[1]) {
//                     const introText = sections[1].trim().split('\n')[0];
//                     const introP = document.createElement('p');
//                     introP.textContent = introText;
//                     container.appendChild(introP);

//                     const addressListContainer = document.createElement('div');
//                     addressListContainer.className = 'address-list';
//                     const cityBlocks = sections[1].trim().split(/\n\d+\.\s*/).filter(block => block.trim() && !block.includes(introText));
                    
//                     cityBlocks.forEach(block => {
//                         const lines = block.trim().split('\n').map(l => l.trim());
//                         const cityNameWithBranches = lines.shift().replace(':', '');
//                         const locationCard = document.createElement('div');
//                         locationCard.className = 'location-card';
//                         const cityHeader = document.createElement('h4');
//                         cityHeader.textContent = cityNameWithBranches;
//                         locationCard.appendChild(cityHeader);
//                         const branchList = document.createElement('ul');
//                         lines.forEach(addressLine => {
//                             const cleanedLine = addressLine.replace(/^[\*\+]\s*/, '').trim();
//                             if (!cleanedLine) return;
//                             const branchItem = document.createElement('li');
//                             branchItem.textContent = cleanedLine;
//                             branchList.appendChild(branchItem);
//                         });
//                         if (branchList.children.length > 0) locationCard.appendChild(branchList);
//                         addressListContainer.appendChild(locationCard);
//                     });
//                     container.appendChild(addressListContainer);
//                 }
//                 return container;
//             }

//             // --- Rule 2: Handle generic lists and paragraphs with proper indentation ---
//             const lines = text.split('\n');
//             let currentList = null;

//             lines.forEach(line => {
//                 const trimmedLine = line.trim();
//                 if (!trimmedLine) {
//                     if (currentList) { container.appendChild(currentList); currentList = null; }
//                     return;
//                 }
//                 const olMatch = trimmedLine.match(/^(\d+)\.\s+(.*)/);
//                 const ulMatch = trimmedLine.match(/^[\*\+]\s+(.*)/);
//                 if (olMatch) {
//                     if (!currentList || currentList.tagName !== 'OL') { if (currentList) container.appendChild(currentList); currentList = document.createElement('ol'); container.appendChild(currentList); }
//                     const listItem = document.createElement('li'); listItem.textContent = olMatch[2]; currentList.appendChild(listItem);
//                 } else if (ulMatch) {
//                     if (!currentList || currentList.tagName !== 'UL') { if (currentList) container.appendChild(currentList); currentList = document.createElement('ul'); container.appendChild(currentList); }
//                     const listItem = document.createElement('li'); listItem.textContent = ulMatch[1]; currentList.appendChild(listItem);
//                 } else {
//                     if (currentList) { container.appendChild(currentList); currentList = null; }
//                     const p = document.createElement('p'); p.textContent = trimmedLine; container.appendChild(p);
//                 }
//             });
//             if (currentList) container.appendChild(currentList);
//             return container;

//         } catch (formatError) {
//             // --- FIX: Graceful fallback for any formatting error ---
//             console.error("Failed to format bot response:", formatError);
//             const fallbackContainer = document.createElement('div');
//             const p = document.createElement('p');
//             p.textContent = markdownText; // Show the raw text if formatting fails
//             fallbackContainer.appendChild(p);
//             return fallbackContainer;
//         }
//     }

//     function showLoadingIndicator() {
//         const loadingElement = document.createElement("div");
//         loadingElement.classList.add("message", "bot-message", "loading-indicator");
//         loadingElement.id = "loading-indicator";
//         loadingElement.innerHTML = `<span></span><span></span><span></span>`;
//         chatBox.appendChild(loadingElement);
//         scrollToBottom();
//     }

//     function hideLoadingIndicator() {
//         const loadingElement = document.getElementById("loading-indicator");
//         if (loadingElement) {
//             chatBox.removeChild(loadingElement);
//         }
//     }

//     function scrollToBottom() {
//         chatBox.scrollTop = chatBox.scrollHeight;
//     }
// });