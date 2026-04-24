document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const sendButton = document.getElementById("send-button");

    let isRequestInProgress = false;

    // --- 1. Connect to the Socket.IO server ---
    const socket = io("http://178.63.15.33:8099")

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
            // Render Citations if they exist in the payload
            if (data.citation && data.citation.length > 0) {
                renderCitations(data.citation, currentBotMessageElement);
            }
            
            // Show Controls (Copy/Export)
            const controls = currentBotMessageElement.querySelector(".bot-controls");
            if (controls) controls.style.display = "flex";
        }
        currentBotMessageElement = null;
        resetInputState();
    });
    function renderCitations(citations, parentWrapper) {
        const panel = document.createElement("div");
        panel.style.cssText = "margin-top: 16px; padding: 12px 16px; background: #f8faf9; border-radius: 8px; border-left: 4px solid #009688; font-size: 0.85rem;";

        const title = document.createElement("p");
        title.innerHTML = "📄 <strong>Sources</strong>";
        title.style.margin = "0 0 8px 0";
        panel.appendChild(title);

        const list = document.createElement("ul");
        list.style.cssText = "margin: 0; padding: 0; list-style: none;";

        citations.forEach((c) => {
            const item = document.createElement("li");
            item.style.marginBottom = "8px";

            // Fallback to source or N/A if display_name is missing
            let locationText = c.display_name || c.source || "Unknown Source";
            if (c.page != null) locationText += ` — Page ${c.page}`;
            if (c.row != null) locationText += ` — Row ${c.row}`;

            const canPreview = (c.file_path || c.source) && c.page != null;
            const sourcePath = c.file_path || c.source;

            item.innerHTML = `
                <span style="font-weight: 600; color: #009688; margin-right: 4px;">[${c.id}]</span>
                <span class="${canPreview ? 'citation-link' : ''}" style="${canPreview ? 'color: #0056b3; cursor: pointer; text-decoration: underline;' : ''}">
                    ${locationText}
                </span>: <i style="color: #666;">${c.topic}</i>
            `;

            if (canPreview) {
                item.querySelector(".citation-link").addEventListener("click", () => {
                    openDocumentModal(sourcePath, c.page, locationText);
                });
            }
            list.appendChild(item);
        });

        panel.appendChild(list);
        
        // Append directly underneath the bot message bubble text
        const msgDiv = parentWrapper.querySelector(".message");
        if (msgDiv) msgDiv.appendChild(panel);
    }

    // --- Document Preview Modal Logic ---
    function openDocumentModal(source, page, label) {
        const modal = document.getElementById("pdfModal");
        const previewBody = document.getElementById("pdfPreviewBody");
        const titleNode = document.getElementById("pdfModalTitle");
        const downloadBtn = document.getElementById("downloadPdfBtn");

        titleNode.textContent = label;
        downloadBtn.style.display = "none"; // Hide PDF print button for images

        previewBody.innerHTML = `
            <div style="display:flex; justify-content:center; align-items:center; height:100%;">
                <div class="loading-spinner" style="color: #666;">Loading Document Page...</div>
                <img id="doc-modal-img" style="display:none; max-width:100%; border:1px solid #ddd; box-shadow:0 4px 8px rgba(0,0,0,0.1);" alt="Document page" />
            </div>
        `;

        const img = document.getElementById("doc-modal-img");
        const loading = previewBody.querySelector(".loading-spinner");

        img.onload = () => { loading.style.display = "none"; img.style.display = "block"; };
        img.onerror = () => { loading.innerHTML = "Could not load this page."; };

        // Hit your FastAPI endpoint
        img.src = `/v1/document-page?source=${encodeURIComponent(source)}&page=${page}`;
        modal.style.display = "flex";
    }

    // Modal Close Triggers
    document.getElementById("closeModal").onclick = () => document.getElementById("pdfModal").style.display = "none";
    document.getElementById("cancelModal").onclick = () => document.getElementById("pdfModal").style.display = "none";
    window.addEventListener("click", (e) => {
        if (e.target == document.getElementById("pdfModal")) document.getElementById("pdfModal").style.display = "none";
    });
        
    // --- Event listener for potential server errors ---
    socket.on("error", (data) => {
        console.error("Server Error:", data.message);
        appendMessage(`Sorry, an error occurred: ${data.message}`, "bot-message");
        resetInputState();
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

    function resetInputState() {
        hideLoadingIndicator();
        isRequestInProgress = false;
        sendButton.disabled = false;
        userInput.focus();
    }

    function scrollToBottom() {
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});

