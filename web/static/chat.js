// Easy Agent WebSocket Chat Client

class ChatClient {
    constructor() {
        this.ws = null;
        this.sessionId = null;
        this.connected = false;
        this.pendingToolApproval = null;
        this.heartbeatInterval = null;
        this.heartbeatTimeout = 30000;  // 30 seconds

        // Reconnection properties
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000;  // Initial 2 seconds
        this.isReconnecting = false;
        this.reconnectTimer = null;

        // DOM elements
        this.chatContainer = document.getElementById('chat-container');
        this.messageInput = document.getElementById('message-input');
        this.sendButton = document.getElementById('send-button');
        this.statusDot = document.getElementById('status-dot');
        this.statusText = document.getElementById('status-text');
        this.sessionDisplay = document.getElementById('session-display');

        // Bind event handlers
        this.sendMessage = this.sendMessage.bind(this);
        this.handleKeyPress = this.handleKeyPress.bind(this);

        // Initialize
        this.setupEventListeners();
        this.connect();
    }

    setupEventListeners() {
        this.sendButton.addEventListener('click', this.sendMessage);
        this.messageInput.addEventListener('keydown', this.handleKeyPress);
        this.messageInput.addEventListener('input', () => this.autoResize());
    }

    autoResize() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 200) + 'px';
    }

    handleKeyPress(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage();
        }
    }

    startHeartbeat() {
        this.stopHeartbeat();
        this.heartbeatInterval = setInterval(() => {
            if (this.connected && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, this.heartbeatTimeout);
    }

    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }

    attemptReconnect() {
        this.isReconnecting = true;
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

        this.updateStatus(false, `重新连接中... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

        this.reconnectTimer = setTimeout(() => {
            this.connect();
        }, delay);
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        this.updateStatus(false, 'Connecting...');
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            this.connected = true;
            this.updateStatus(true, 'Connected');
            this.startHeartbeat();
            this.reconnectAttempts = 0;
            this.isReconnecting = false;

            // Check for old session_id and send resume request
            const oldSessionId = localStorage.getItem('easy_agent_session_id');
            if (oldSessionId) {
                this.ws.send(JSON.stringify({
                    type: 'resume',
                    session_id: oldSessionId
                }));
            }
        };

        this.ws.onclose = () => {
            this.connected = false;
            this.stopHeartbeat();

            // Clear reconnect timer
            if (this.reconnectTimer) {
                clearTimeout(this.reconnectTimer);
                this.reconnectTimer = null;
            }

            this.updateStatus(false, '已断开连接');

            if (!this.isReconnecting && this.reconnectAttempts < this.maxReconnectAttempts) {
                this.attemptReconnect();
            } else if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                this.addSystemMessage('连接失败，请刷新页面。');
            }
        };

        this.ws.onerror = (error) => {
            this.connected = false;
            this.updateStatus(false, 'Error');
            this.addSystemMessage('Connection error occurred.');
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
    }

    handleMessage(data) {
        switch (data.type) {
            case 'session':
                this.sessionId = data.session_id;
                // Store session_id in localStorage for reconnection
                localStorage.setItem('easy_agent_session_id', data.session_id);
                this.sessionDisplay.textContent = `Session: ${data.session_id}`;
                // Show appropriate message based on resumed flag
                if (data.resumed) {
                    this.addSystemMessage(`已恢复会话 ${data.session_id} (${data.provider})`);
                } else {
                    this.addSystemMessage(`Connected to ${data.provider} (Session: ${data.session_id})`);
                }
                break;

            case 'pong':
                // Heartbeat response, no action needed
                break;

            case 'progress':
                this.updateProgressIndicator(data.content);
                break;

            case 'response':
                this.removeTypingIndicator();
                this.addAgentMessage(data.content);
                break;

            case 'error':
                this.removeTypingIndicator();
                this.addErrorMessage(data.content);
                break;

            case 'tool_request':
                this.showToolApprovalDialog(data.tool, data.args);
                break;

            default:
                console.warn('Unknown message type:', data.type);
        }
    }

    sendMessage() {
        const content = this.messageInput.value.trim();
        if (!content || !this.connected) return;

        // Check for pending tool approval
        if (this.pendingToolApproval) {
            return;
        }

        // Add user message to UI
        this.addUserMessage(content);

        // Clear input
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';

        // Send to server
        this.ws.send(JSON.stringify({
            type: 'message',
            content: content
        }));

        // Show typing indicator
        this.showTypingIndicator();
    }

    addUserMessage(content) {
        const msg = document.createElement('div');
        msg.className = 'message user';
        msg.innerHTML = this.renderMarkdown(content);
        this.chatContainer.appendChild(msg);
        this.scrollToBottom();
    }

    addAgentMessage(content) {
        const msg = document.createElement('div');
        msg.className = 'message agent';
        msg.innerHTML = this.renderMarkdown(content);
        this.chatContainer.appendChild(msg);
        this.scrollToBottom();
    }

    addErrorMessage(content) {
        const msg = document.createElement('div');
        msg.className = 'message error';
        msg.textContent = `Error: ${content}`;
        this.chatContainer.appendChild(msg);
        this.scrollToBottom();
    }

    addSystemMessage(content) {
        const msg = document.createElement('div');
        msg.className = 'message system';
        msg.textContent = content;
        this.chatContainer.appendChild(msg);
        this.scrollToBottom();
    }

    showTypingIndicator() {
        this.removeTypingIndicator();
        const indicator = document.createElement('div');
        indicator.className = 'message agent typing';
        indicator.innerHTML = `
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
                <span class="typing-label">处理中...</span>
            </div>
        `;
        this.chatContainer.appendChild(indicator);
        this.scrollToBottom();
    }

    updateProgressIndicator(label) {
        let indicator = this.chatContainer.querySelector('.typing');
        if (!indicator) {
            this.showTypingIndicator();
            indicator = this.chatContainer.querySelector('.typing');
        }
        const labelEl = indicator.querySelector('.typing-label');
        if (labelEl) {
            labelEl.textContent = label;
        }
        this.scrollToBottom();
    }

    removeTypingIndicator() {
        const typing = this.chatContainer.querySelector('.typing');
        if (typing) {
            typing.remove();
        }
    }

    showToolApprovalDialog(tool, args) {
        this.pendingToolApproval = { tool, args };

        const msg = document.createElement('div');
        msg.className = 'message agent';
        msg.innerHTML = `
            <div class="tool-dialog">
                <h4>Dangerous Operation Requested</h4>
                <p>Tool: <strong>${this.escapeHtml(tool)}</strong></p>
                <pre>${this.escapeHtml(JSON.stringify(args, null, 2))}</pre>
                <div class="buttons">
                    <button class="approve" onclick="chatClient.approveTool()">Approve</button>
                    <button class="deny" onclick="chatClient.denyTool()">Deny</button>
                </div>
            </div>
        `;
        this.chatContainer.appendChild(msg);
        this.scrollToBottom();
    }

    approveTool() {
        if (!this.pendingToolApproval) return;
        this.ws.send(JSON.stringify({
            type: 'tool_response',
            approved: true
        }));
        this.pendingToolApproval = null;
        this.showTypingIndicator();
    }

    denyTool() {
        if (!this.pendingToolApproval) return;
        this.ws.send(JSON.stringify({
            type: 'tool_response',
            approved: false
        }));
        this.pendingToolApproval = null;
        this.showTypingIndicator();
    }

    updateStatus(connected, text) {
        this.statusDot.classList.toggle('disconnected', !connected);
        this.statusText.textContent = text;
        this.sendButton.disabled = !connected;
    }

    scrollToBottom() {
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }

    renderMarkdown(text) {
        // Simple markdown rendering
        let html = this.escapeHtml(text);

        // Code blocks
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');

        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Headings
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

        // Bold
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Italic
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

        // Unordered list items
        html = html.replace(/^[-*] (.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

        // Line breaks (skip lines that are block elements)
        html = html.replace(/\n(?!<[hul])/g, '<br>');

        return html;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize chat client
let chatClient;
document.addEventListener('DOMContentLoaded', () => {
    chatClient = new ChatClient();
});
