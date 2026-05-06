// Easy Agent WebSocket Chat Client

class ChatClient {
    constructor() {
        this.ws = null;
        this.sessionId = null;
        this.connected = false;
        this.pendingToolApproval = null;

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

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        this.updateStatus(false, 'Connecting...');
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            this.connected = true;
            this.updateStatus(true, 'Connected');
        };

        this.ws.onclose = () => {
            this.connected = false;
            this.updateStatus(false, 'Disconnected');
            this.addSystemMessage('Connection closed. Refresh to reconnect.');
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
                this.sessionDisplay.textContent = `Session: ${data.session_id}`;
                this.addSystemMessage(`Connected to ${data.provider} (Session: ${data.session_id})`);
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
        const indicator = document.createElement('div');
        indicator.className = 'message agent typing';
        indicator.innerHTML = `
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        this.chatContainer.appendChild(indicator);
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

        // Bold
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Italic
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

        // Line breaks
        html = html.replace(/\n/g, '<br>');

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
