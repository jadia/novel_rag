/**
 * =============================================================================
 * Novel RAG — Front-End Application Logic
 * =============================================================================
 * This file controls ALL client-side behavior for the Novel RAG web app.
 *
 * ARCHITECTURE OVERVIEW:
 * The app follows a simple "State → Render" pattern:
 *   1. STATE:    A single JavaScript object holds all app data (selected novel,
 *                messages, sessions, settings, theme).
 *   2. EVENTS:   Button clicks, WebSocket messages, and keyboard input modify
 *                the state.
 *   3. RENDER:   After state changes, we update the DOM to reflect the new state.
 *
 * WHY NOT REACT/VUE?
 * For a small app like this, vanilla JS is simpler to understand and debug.
 * There are no build tools, no virtual DOM, no component lifecycle — just
 * direct DOM manipulation. This makes it perfect for learning how web apps
 * actually work under the hood.
 *
 * FILE STRUCTURE:
 *   1. State Management
 *   2. API Service Layer (fetch wrappers + WebSocket)
 *   3. DOM Rendering Functions
 *   4. Event Handlers
 *   5. Utilities (markdown, theme, auto-resize)
 *   6. Initialization
 * =============================================================================
 */

// ==========================================================================
// 1. STATE MANAGEMENT
// ==========================================================================
// The entire app state lives in this one object. Every function reads from
// or writes to this object. This makes it trivial to debug — just inspect
// `window.state` in the browser console to see everything.
// ==========================================================================

const state = {
    /** @type {string|null}  Currently selected novel name */
    selectedNovel: null,

    /** @type {string|null}  Currently active chat session ID */
    activeSessionId: null,

    /** @type {Array<{role: string, content: string}>}  Messages in current chat */
    messages: [],

    /** @type {Array<{id: string, title: string, created_at: string}>}  All sessions for selected novel */
    sessions: [],

    /** @type {object}  Current settings from server */
    settings: {},

    /** @type {string}  Current theme: 'light' or 'dark' */
    theme: localStorage.getItem('novel-rag-theme') || 'light',

    /** @type {boolean}  Whether the AI is currently generating a response */
    isStreaming: false,

    /** @type {WebSocket|null}  Active chat WebSocket connection */
    chatWs: null,

    /** @type {string}  Accumulated streaming response text */
    streamBuffer: '',
};

// Make state accessible in the browser console for debugging
window.state = state;

// ==========================================================================
// 2. API SERVICE LAYER
// ==========================================================================
// These functions wrap all communication with the server. By centralizing
// API calls here, the rest of the code never needs to know about URLs,
// HTTP methods, or JSON parsing.
//
// PATTERN: Each function returns a Promise that resolves to the parsed
// JSON response. If the request fails, we log the error and return null.
// ==========================================================================

const API = {
    /**
     * Generic fetch wrapper with error handling.
     * All API calls go through this function.
     *
     * @param {string} url    - The API endpoint path (e.g., '/api/novels')
     * @param {object} [opts] - fetch() options (method, body, headers)
     * @returns {Promise<object|null>} Parsed JSON response or null on error
     */
    async request(url, opts = {}) {
        try {
            // Default to JSON content type for non-GET requests
            if (opts.body && !opts.headers) {
                opts.headers = { 'Content-Type': 'application/json' };
            }
            const response = await fetch(url, opts);
            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                console.error(`API error ${response.status}:`, error);
                return null;
            }
            return await response.json();
        } catch (err) {
            console.error(`API request failed for ${url}:`, err);
            return null;
        }
    },

    /** Fetch the list of available novels with their embedding status. */
    async getNovels() {
        return this.request('/api/novels');
    },

    /** Check a specific novel's embedding status and chunk count. */
    async getNovelStatus(novelName) {
        return this.request(`/api/novels/${encodeURIComponent(novelName)}/status`);
    },

    /** Create a new chat session for a novel. */
    async createSession(novelName) {
        return this.request('/api/sessions', {
            method: 'POST',
            body: JSON.stringify({ novel_name: novelName }),
        });
    },

    /** List all chat sessions for a specific novel. */
    async getSessions(novelName) {
        return this.request(`/api/sessions/${encodeURIComponent(novelName)}`);
    },

    /** Get all messages for a specific session. */
    async getSessionMessages(sessionId) {
        return this.request(`/api/sessions/${sessionId}/messages`);
    },

    /** Delete a chat session. */
    async deleteSession(sessionId) {
        return this.request(`/api/sessions/${sessionId}`, { method: 'DELETE' });
    },

    /** Fetch current settings (API keys are masked). */
    async getSettings() {
        return this.request('/api/settings');
    },

    /** Update runtime settings. */
    async updateSettings(settings) {
        return this.request('/api/settings', {
            method: 'PUT',
            body: JSON.stringify(settings),
        });
    },
};


// ==========================================================================
// 3. WEBSOCKET MANAGEMENT
// ==========================================================================
// WebSockets maintain a persistent, bidirectional connection between the
// browser and server. Unlike HTTP (request → response), WebSockets let
// the server push data to the client at any time — perfect for streaming
// LLM tokens and ingestion progress.
//
// We manage two types of WebSocket connections:
//   1. Chat WS: stays open while a novel is selected, handles Q&A streaming
//   2. Ingest WS: temporary, opened only during re-embedding, auto-closes
// ==========================================================================

const WS = {
    /**
     * Open a WebSocket connection for streaming chat responses.
     *
     * CONNECTION LIFECYCLE:
     *   1. Browser opens connection to ws://host/ws/chat
     *   2. Connection stays open — client sends questions, server streams answers
     *   3. Connection closes when the user switches novels or navigates away
     *
     * MESSAGE PROTOCOL (server → client):
     *   { type: "token",  content: "partial text" }   — streaming token
     *   { type: "done",   content: "full answer" }     — stream complete
     *   { type: "error",  content: "error message" }   — something went wrong
     */
    connectChat() {
        // Close any existing connection first
        if (state.chatWs) {
            state.chatWs.close();
            state.chatWs = null;
        }

        // Determine WebSocket URL (ws:// for http://, wss:// for https://)
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/chat`;

        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log('Chat WebSocket connected');
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleChatMessage(data);
        };

        ws.onerror = (error) => {
            console.error('Chat WebSocket error:', error);
        };

        ws.onclose = () => {
            console.log('Chat WebSocket closed');
            state.chatWs = null;
        };

        state.chatWs = ws;
    },

    /**
     * Send a question through the chat WebSocket.
     *
     * @param {string} question   - The user's question
     * @param {string} novelName  - Which novel to query
     * @param {string} sessionId  - The session to store the message in
     */
    sendQuestion(question, novelName, sessionId) {
        if (!state.chatWs || state.chatWs.readyState !== WebSocket.OPEN) {
            console.error('Chat WebSocket not connected');
            // Try to reconnect
            this.connectChat();
            // Wait a bit for connection, then retry
            setTimeout(() => {
                if (state.chatWs && state.chatWs.readyState === WebSocket.OPEN) {
                    state.chatWs.send(JSON.stringify({
                        question,
                        novel_name: novelName,
                        session_id: sessionId,
                    }));
                }
            }, 500);
            return;
        }

        state.chatWs.send(JSON.stringify({
            question,
            novel_name: novelName,
            session_id: sessionId,
        }));
    },

    /**
     * Open a temporary WebSocket for ingestion progress.
     *
     * Unlike the chat WS, this connection is short-lived:
     *   1. Opens connection → server immediately starts embedding
     *   2. Server sends progress updates as { type: "progress", current, total }
     *   3. Server sends { type: "done" } → we close the connection
     *
     * @param {string} novelName - The novel to re-embed
     */
    startIngest(novelName) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/ingest/${encodeURIComponent(novelName)}`;

        const ws = new WebSocket(wsUrl);

        // Show the progress modal
        showIngestModal();

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleIngestMessage(data);
        };

        ws.onerror = (error) => {
            console.error('Ingest WebSocket error:', error);
            hideIngestModal();
        };

        ws.onclose = () => {
            console.log('Ingest WebSocket closed');
        };
    },
};


// ==========================================================================
// 4. DOM RENDERING FUNCTIONS
// ==========================================================================
// These functions take data from the state and update the DOM accordingly.
// They never make API calls — they just paint pixels on the screen.
// ==========================================================================

/**
 * Populate the novel selector dropdown with available novels.
 *
 * @param {Array<{name: string, has_embeddings: boolean}>} novels
 */
function renderNovels(novels) {
    const select = document.getElementById('novel-select');

    // Clear existing options (except the placeholder)
    select.innerHTML = '<option value="">— Choose a novel —</option>';

    novels.forEach(novel => {
        const option = document.createElement('option');
        option.value = novel.name;
        // Show a check mark if embeddings exist, warning if not
        option.textContent = `${novel.has_embeddings ? '✅' : '⚠️'} ${novel.name}`;
        select.appendChild(option);
    });
}

/**
 * Update the novel embedding status indicator.
 *
 * @param {boolean} hasEmbeddings - Whether the novel has been ingested
 * @param {number}  chunkCount    - Number of chunks in the database
 */
function renderNovelStatus(hasEmbeddings, chunkCount) {
    const statusDiv = document.getElementById('novel-status');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');

    statusDiv.classList.remove('hidden');
    statusDot.className = 'status-dot ' + (hasEmbeddings ? 'ready' : 'not-ready');
    statusText.textContent = hasEmbeddings
        ? `Ready — ${chunkCount} chunks indexed`
        : 'Not embedded — click Re-Embed';
}

/**
 * Render the session list in the sidebar.
 *
 * @param {Array<{id: string, title: string, updated_at: string}>} sessions
 */
function renderSessions(sessions) {
    const list = document.getElementById('session-list');

    if (sessions.length === 0) {
        list.innerHTML = '<p class="empty-state">No chats yet. Start a new one!</p>';
        return;
    }

    list.innerHTML = '';
    sessions.forEach(session => {
        const item = document.createElement('div');
        item.className = 'session-item' +
            (session.id === state.activeSessionId ? ' active' : '');
        item.dataset.sessionId = session.id;

        const title = document.createElement('span');
        title.className = 'session-title';
        title.textContent = session.title;
        title.title = session.title; // Show full title on hover

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'session-delete';
        deleteBtn.textContent = '🗑️';
        deleteBtn.title = 'Delete this chat';
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // Don't trigger session selection
            handleDeleteSession(session.id);
        });

        item.appendChild(title);
        item.appendChild(deleteBtn);

        // Click to load this session's messages
        item.addEventListener('click', () => handleSelectSession(session.id));

        list.appendChild(item);
    });
}

/**
 * Render all messages in the chat view.
 *
 * @param {Array<{role: string, content: string}>} messages
 */
function renderMessages(messages) {
    const container = document.getElementById('message-container');
    container.innerHTML = '';

    messages.forEach(msg => {
        appendMessage(msg.role, msg.content);
    });

    scrollToBottom();
}

/**
 * Append a single message bubble to the chat.
 * This is used both for loading history and for new messages.
 *
 * @param {string} role    - 'user' or 'assistant'
 * @param {string} content - The message text (markdown for assistant)
 */
function appendMessage(role, content) {
    const container = document.getElementById('message-container');

    const msgDiv = document.createElement('div');
    msgDiv.className = `message message-${role}`;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    if (role === 'assistant') {
        // Parse basic markdown for assistant responses
        bubble.innerHTML = renderMarkdown(content);
    } else {
        // User messages are plain text (escape HTML to prevent XSS)
        bubble.textContent = content;
    }

    msgDiv.appendChild(bubble);
    container.appendChild(msgDiv);

    scrollToBottom();
}

/**
 * Get the streaming message bubble element, creating it if it doesn't exist.
 * During streaming, we progressively update this single bubble rather than
 * creating a new element for each token.
 *
 * @returns {HTMLElement} The streaming assistant bubble element.
 */
function getOrCreateStreamBubble() {
    let existing = document.getElementById('streaming-bubble');
    if (existing) return existing;

    const container = document.getElementById('message-container');

    const msgDiv = document.createElement('div');
    msgDiv.className = 'message message-assistant';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.id = 'streaming-bubble';

    msgDiv.appendChild(bubble);
    container.appendChild(msgDiv);

    return bubble;
}

/**
 * Switch between the welcome screen and the chat view.
 *
 * @param {boolean} showChat - true to show chat, false for welcome
 */
function toggleChatView(showChat) {
    document.getElementById('welcome-screen').classList.toggle('hidden', showChat);
    document.getElementById('chat-view').classList.toggle('hidden', !showChat);
}

/** Show the ingestion progress modal. */
function showIngestModal() {
    document.getElementById('ingest-modal').classList.remove('hidden');
    document.getElementById('progress-bar').style.width = '0%';
    document.getElementById('ingest-percent').textContent = '0%';
    document.getElementById('ingest-message').textContent = 'Preparing...';
}

/** Hide the ingestion progress modal. */
function hideIngestModal() {
    document.getElementById('ingest-modal').classList.add('hidden');
}

/** Show or hide the typing indicator dots. */
function toggleTypingIndicator(show) {
    document.getElementById('typing-indicator').classList.toggle('hidden', !show);
    if (show) scrollToBottom();
}

/** Scroll the message container to the bottom. */
function scrollToBottom() {
    const container = document.getElementById('message-container');
    // requestAnimationFrame ensures the DOM has updated before scrolling
    requestAnimationFrame(() => {
        container.scrollTop = container.scrollHeight;
    });
}


// ==========================================================================
// 5. EVENT HANDLERS
// ==========================================================================
// These functions respond to user interactions (clicks, key presses,
// WebSocket messages) and coordinate between the API layer and the
// rendering functions.
// ==========================================================================

/**
 * Handle novel selection change.
 * When the user picks a novel from the dropdown, we:
 *   1. Update the state
 *   2. Check embedding status
 *   3. Load sessions for this novel
 *   4. Open a chat WebSocket connection
 */
async function handleNovelSelect(novelName) {
    if (!novelName) {
        state.selectedNovel = null;
        state.sessions = [];
        state.activeSessionId = null;
        document.getElementById('novel-status').classList.add('hidden');
        document.getElementById('btn-new-chat').disabled = true;
        document.getElementById('btn-reingest').disabled = true;
        renderSessions([]);
        toggleChatView(false);
        return;
    }

    state.selectedNovel = novelName;
    state.activeSessionId = null;

    // Enable action buttons
    document.getElementById('btn-new-chat').disabled = false;
    document.getElementById('btn-reingest').disabled = false;

    // Check embedding status
    const statusData = await API.getNovelStatus(novelName);
    if (statusData) {
        renderNovelStatus(statusData.has_embeddings, statusData.chunk_count);
    }

    // Load sessions
    const sessionsData = await API.getSessions(novelName);
    if (sessionsData) {
        state.sessions = sessionsData.sessions;
        renderSessions(state.sessions);
    }

    // Connect WebSocket for chat
    WS.connectChat();

    // Show welcome screen (no chat selected yet)
    toggleChatView(false);
}

/**
 * Handle "New Chat" button click.
 * Creates a new session on the server and switches to the chat view.
 */
async function handleNewChat() {
    if (!state.selectedNovel) return;

    const data = await API.createSession(state.selectedNovel);
    if (!data) return;

    state.activeSessionId = data.session_id;
    state.messages = [];

    // Refresh the session list
    const sessionsData = await API.getSessions(state.selectedNovel);
    if (sessionsData) {
        state.sessions = sessionsData.sessions;
        renderSessions(state.sessions);
    }

    // Switch to chat view
    toggleChatView(true);
    renderMessages([]);

    // Focus the input
    document.getElementById('question-input').focus();
}

/**
 * Handle clicking on a session in the sidebar.
 * Loads the session's messages and displays them.
 */
async function handleSelectSession(sessionId) {
    state.activeSessionId = sessionId;

    // Load messages
    const data = await API.getSessionMessages(sessionId);
    if (data) {
        state.messages = data.messages;
        renderMessages(state.messages);

        // Update the novel from the session (in case it differs)
        if (data.session && data.session.novel_name) {
            state.selectedNovel = data.session.novel_name;
        }
    }

    // Update the active session highlight in the sidebar
    renderSessions(state.sessions);

    // Switch to chat view
    toggleChatView(true);

    // Ensure WebSocket is connected
    if (!state.chatWs || state.chatWs.readyState !== WebSocket.OPEN) {
        WS.connectChat();
    }
}

/**
 * Handle deleting a session.
 * Shows a confirmation before deleting.
 */
async function handleDeleteSession(sessionId) {
    if (!confirm('Delete this chat and all its messages?')) return;

    await API.deleteSession(sessionId);

    // If we deleted the active session, go back to welcome
    if (state.activeSessionId === sessionId) {
        state.activeSessionId = null;
        state.messages = [];
        toggleChatView(false);
    }

    // Refresh session list
    const sessionsData = await API.getSessions(state.selectedNovel);
    if (sessionsData) {
        state.sessions = sessionsData.sessions;
        renderSessions(state.sessions);
    }
}

/**
 * Handle sending a message.
 * This is triggered by the send button or Enter key.
 */
async function handleSendMessage() {
    const input = document.getElementById('question-input');
    const question = input.value.trim();
    if (!question || state.isStreaming) return;

    // If no session exists, create one first
    if (!state.activeSessionId) {
        await handleNewChat();
    }

    // Add the user's message to the UI immediately (optimistic update)
    state.messages.push({ role: 'user', content: question });
    appendMessage('user', question);

    // Clear the input and reset height
    input.value = '';
    input.style.height = 'auto';
    updateSendButton();

    // Show typing indicator while waiting for the response
    state.isStreaming = true;
    state.streamBuffer = '';
    toggleTypingIndicator(true);

    // Send the question via WebSocket
    WS.sendQuestion(question, state.selectedNovel, state.activeSessionId);
}

/**
 * Handle incoming WebSocket chat messages.
 * This is called for each message received from the server.
 *
 * @param {object} data - The parsed JSON message from the server
 */
function handleChatMessage(data) {
    switch (data.type) {
        case 'token':
            // A new token has arrived — append it to the streaming bubble
            toggleTypingIndicator(false);
            state.streamBuffer += data.content;

            const bubble = getOrCreateStreamBubble();
            bubble.innerHTML = renderMarkdown(state.streamBuffer);
            scrollToBottom();
            break;

        case 'done':
            // Streaming is complete — finalize the message
            state.isStreaming = false;
            toggleTypingIndicator(false);

            // Remove the streaming bubble ID so it becomes permanent
            const streamBubble = document.getElementById('streaming-bubble');
            if (streamBubble) {
                streamBubble.removeAttribute('id');
                streamBubble.innerHTML = renderMarkdown(data.content);
            }

            // Add to state
            state.messages.push({ role: 'assistant', content: data.content });

            // Update session ID if the server assigned one
            if (data.session_id) {
                state.activeSessionId = data.session_id;
            }

            // Refresh sessions (title may have auto-updated)
            refreshSessions();
            break;

        case 'error':
            // Something went wrong — show error in chat
            state.isStreaming = false;
            toggleTypingIndicator(false);

            // Remove streaming bubble if it was created
            const errorStreamBubble = document.getElementById('streaming-bubble');
            if (errorStreamBubble) {
                errorStreamBubble.parentElement.remove();
            }

            appendMessage('assistant', `⚠️ Error: ${data.content}`);
            break;
    }
}

/**
 * Handle incoming WebSocket ingest progress messages.
 * Updates the progress bar modal.
 *
 * @param {object} data - The parsed JSON progress message
 */
function handleIngestMessage(data) {
    switch (data.type) {
        case 'progress': {
            // Update the progress bar
            const percent = data.total > 0
                ? Math.round((data.current / data.total) * 100)
                : 0;
            document.getElementById('progress-bar').style.width = `${percent}%`;
            document.getElementById('ingest-percent').textContent = `${percent}%`;
            document.getElementById('ingest-message').textContent = data.message;
            break;
        }
        case 'done':
            // Ingestion complete!
            document.getElementById('progress-bar').style.width = '100%';
            document.getElementById('ingest-percent').textContent = '100%';
            document.getElementById('ingest-message').textContent =
                data.message || 'Complete!';

            // Refresh novel status to show "Ready"
            setTimeout(async () => {
                hideIngestModal();
                if (state.selectedNovel) {
                    const statusData = await API.getNovelStatus(state.selectedNovel);
                    if (statusData) {
                        renderNovelStatus(statusData.has_embeddings, statusData.chunk_count);
                    }
                }
                // Also refresh the novels list
                const novelsData = await API.getNovels();
                if (novelsData) renderNovels(novelsData.novels);
            }, 1500); // Brief pause so user sees 100%
            break;

        case 'error':
            document.getElementById('ingest-message').textContent =
                `Error: ${data.content}`;
            document.getElementById('progress-bar').style.width = '0%';
            setTimeout(hideIngestModal, 3000);
            break;
    }
}

/** Refresh the sessions list without changing the selected session. */
async function refreshSessions() {
    if (!state.selectedNovel) return;
    const data = await API.getSessions(state.selectedNovel);
    if (data) {
        state.sessions = data.sessions;
        renderSessions(state.sessions);
    }
}

/**
 * Handle opening the settings modal.
 * Fetches current settings and populates the form.
 */
async function handleOpenSettings() {
    const data = await API.getSettings();
    if (!data) return;

    state.settings = data.settings;

    // Populate form fields with current values
    document.getElementById('setting-api-key').value = data.settings.gemini_api_key || '';
    document.getElementById('setting-gen-model').value = data.settings.generation_model || '';
    document.getElementById('setting-use-local').value = String(data.settings.use_local_embeddings || false);
    document.getElementById('setting-api-model').value = data.settings.api_model_name || '';
    document.getElementById('setting-local-model').value = data.settings.local_model_name || '';

    // Show the modal
    document.getElementById('settings-modal').classList.remove('hidden');
}

/**
 * Handle saving settings.
 * Sends updated values to the server and closes the modal.
 */
async function handleSaveSettings() {
    const settings = {
        gemini_api_key: document.getElementById('setting-api-key').value,
        generation_model: document.getElementById('setting-gen-model').value,
        use_local_embeddings: document.getElementById('setting-use-local').value === 'true',
        api_model_name: document.getElementById('setting-api-model').value,
        local_model_name: document.getElementById('setting-local-model').value,
    };

    const result = await API.updateSettings(settings);
    if (result && result.saved) {
        // Close the modal
        document.getElementById('settings-modal').classList.add('hidden');
        console.log('Settings saved successfully');
    }
}

/**
 * Handle theme toggle (light ↔ dark).
 *
 * HOW CSS THEME SWITCHING WORKS:
 * We set a data-theme attribute on the <html> element. Our CSS defines
 * different variable values for [data-theme="dark"]. When we flip the
 * attribute, EVERY element using those CSS variables updates automatically.
 * We also persist the choice in localStorage so it survives page reloads.
 */
function handleThemeToggle() {
    state.theme = state.theme === 'light' ? 'dark' : 'light';

    // Set the CSS attribute that triggers the theme variables
    document.documentElement.setAttribute('data-theme', state.theme);

    // Update the toggle icon
    document.getElementById('theme-icon').textContent =
        state.theme === 'dark' ? '☀️' : '🌙';

    // Persist the preference in localStorage (survives browser restart)
    localStorage.setItem('novel-rag-theme', state.theme);
}

/** Update the send button's disabled state based on input content. */
function updateSendButton() {
    const input = document.getElementById('question-input');
    const btn = document.getElementById('btn-send');
    btn.disabled = !input.value.trim() || state.isStreaming;
}


// ==========================================================================
// 6. UTILITIES
// ==========================================================================

/**
 * Simple markdown-to-HTML renderer.
 *
 * WHY NOT USE A LIBRARY?
 * For this app, we only need basic formatting (bold, italic, headers,
 * lists, code blocks). A full markdown library like marked.js adds ~30KB.
 * This regex-based approach covers 90% of what the LLM outputs in ~30 lines.
 *
 * SECURITY NOTE:
 * We first escape HTML entities in the input to prevent XSS attacks.
 * Then we apply markdown patterns that generate safe HTML.
 *
 * @param {string} text - Raw markdown text
 * @returns {string} HTML string
 */
function renderMarkdown(text) {
    if (!text) return '';

    // First, escape HTML to prevent XSS (must happen BEFORE adding our HTML)
    let html = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Code blocks (```...```) — must be processed before inline patterns
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');

    // Inline code (`...`)
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Headers (### → h3, ## → h2, # → h1)
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Bold (**text**)
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic (*text*)
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Unordered lists (- item or * item)
    html = html.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

    // Ordered lists (1. item)
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Paragraphs (double newlines create paragraph breaks)
    html = html.replace(/\n\n/g, '</p><p>');
    html = `<p>${html}</p>`;

    // Single newlines become <br> (within paragraphs)
    html = html.replace(/\n/g, '<br>');

    // Clean up empty paragraphs
    html = html.replace(/<p><\/p>/g, '');

    return html;
}

/**
 * Auto-resize the textarea as the user types.
 * The textarea starts at 1 row and expands up to max-height (120px in CSS).
 *
 * HOW IT WORKS:
 * 1. Reset height to 'auto' so scrollHeight reflects actual content
 * 2. Set height to scrollHeight to fit all the text
 * The max-height in CSS prevents unlimited expansion.
 */
function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}


// ==========================================================================
// 7. INITIALIZATION
// ==========================================================================
// This section runs when the page loads. It:
//   1. Applies the saved theme
//   2. Fetches the list of novels
//   3. Wires up all event listeners
// ==========================================================================

document.addEventListener('DOMContentLoaded', async () => {
    // ---- Apply saved theme ----
    document.documentElement.setAttribute('data-theme', state.theme);
    document.getElementById('theme-icon').textContent =
        state.theme === 'dark' ? '☀️' : '🌙';

    // ---- Load novels ----
    const data = await API.getNovels();
    if (data) {
        renderNovels(data.novels);
    }

    // ---- Wire up event listeners ----

    // Novel selector
    document.getElementById('novel-select').addEventListener('change', (e) => {
        handleNovelSelect(e.target.value);
    });

    // New Chat button
    document.getElementById('btn-new-chat').addEventListener('click', handleNewChat);

    // Re-Embed button
    document.getElementById('btn-reingest').addEventListener('click', () => {
        if (state.selectedNovel) {
            WS.startIngest(state.selectedNovel);
        }
    });

    // Send button
    document.getElementById('btn-send').addEventListener('click', handleSendMessage);

    // Question input — handle Enter key and auto-resize
    const input = document.getElementById('question-input');
    input.addEventListener('keydown', (e) => {
        // Enter sends the message (Shift+Enter inserts a newline)
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });
    input.addEventListener('input', () => {
        autoResizeTextarea(input);
        updateSendButton();
    });

    // Theme toggle
    document.getElementById('btn-theme-toggle').addEventListener('click', handleThemeToggle);

    // Settings modal
    document.getElementById('btn-settings').addEventListener('click', handleOpenSettings);
    document.getElementById('btn-close-settings').addEventListener('click', () => {
        document.getElementById('settings-modal').classList.add('hidden');
    });
    document.getElementById('btn-save-settings').addEventListener('click', handleSaveSettings);

    // Close modals when clicking outside
    document.getElementById('settings-modal').addEventListener('click', (e) => {
        if (e.target.classList.contains('modal-overlay')) {
            document.getElementById('settings-modal').classList.add('hidden');
        }
    });
});
