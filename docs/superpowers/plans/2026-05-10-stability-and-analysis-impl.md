# Session 稳定性 + AI 分析深度优化 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 WebSocket 心跳保活、断线自动重连、会话恢复、以及 AI 结构化分析框架

**Architecture:**
- 前端：心跳机制（setInterval）、指数退避重连、localStorage 存储 session_id、resume 握手
- 后端：ping/pong 处理、session_last_active 时间戳管理、resume 逻辑复用 thread_id
- AI：在 agent_node 的 SYSTEM_PROMPT 中注入场景化分析框架指令

**Tech Stack:** FastAPI WebSocket、JavaScript、LangGraph、localStorage

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `web/static/chat.js` | 心跳发送、自动重连、resume 握手、localStorage 管理 |
| `web/app.py` | ping/pong 处理、resume 逻辑、session_last_active 维护 |
| `start.sh` | uvicorn 启动参数增加 ws-ping-interval/timeout |
| `src/nodes/agent_node.py` | SYSTEM_PROMPT 增加分析框架、动态拼接逻辑 |

---

## Task 1: 前端心跳机制

**Files:**
- Modify: `web/static/chat.js`

- [ ] **Step 1: 在 ChatClient 构造函数中添加心跳相关属性**

```javascript
constructor() {
    this.ws = null;
    this.sessionId = null;
    this.connected = false;
    this.pendingToolApproval = null;
    this.heartbeatInterval = null;  // 新增
    this.heartbeatTimeout = 30000;  // 30秒

    // DOM elements
    this.chatContainer = document.getElementById('chat-container');
    // ... 其余不变
}
```

- [ ] **Step 2: 在 connect 成功后启动心跳**

在 `this.ws.onopen` 回调中增加启动心跳：

```javascript
this.ws.onopen = () => {
    this.connected = true;
    this.updateStatus(true, 'Connected');
    this.startHeartbeat();  // 新增
};
```

- [ ] **Step 3: 实现 startHeartbeat 方法**

在 `handleKeyPress` 方法后添加：

```javascript
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
```

- [ ] **Step 4: 在 ws.onclose 中停止心跳**

```javascript
this.ws.onclose = () => {
    this.connected = false;
    this.updateStatus(false, 'Disconnected');
    this.stopHeartbeat();  // 新增
    this.addSystemMessage('Connection closed. Refresh to reconnect.');
};
```

- [ ] **Step 5: 在 handleMessage 中处理 pong 消息**

在 `handleMessage` 的 switch 中增加 `pong` case：

```javascript
case 'pong':
    // 心跳响应，无需处理
    break;

case 'response':
    this.removeTypingIndicator();
    this.addAgentMessage(data.content);
    break;
```

- [ ] **Step 6: 提交前端心跳功能**

```bash
git add web/static/chat.js
git commit -m "feat: add WebSocket heartbeat to keep connection alive"
```

---

## Task 2: 后端 ping/pong 处理

**Files:**
- Modify: `web/app.py`

- [ ] **Step 1: 在 WebSocket 消息循环中增加 ping 处理**

在 `websocket_endpoint` 函数的 `while True` 循环中，`msg_type == "message"` 分支前增加：

```python
if msg_type == "ping":
    await websocket.send_json({"type": "pong"})
    continue
```

- [ ] **Step 2: 提交后端 ping/pong 处理**

```bash
git add web/app.py
git commit -m "feat: handle ping/pong for WebSocket heartbeat"
```

---

## Task 3: uvicorn WebSocket 保活配置

**Files:**
- Modify: `start.sh`

- [ ] **Step 1: 在 web 服务启动命令中增加 ws-ping 参数**

找到 `start_process "web"` 那一行，修改为：

```bash
start_process "web" "python web_app.py --host 0.0.0.0 --port 8080 --ws-ping-interval 30 --ws-ping-timeout 300"
```

- [ ] **Step 2: 提交启动脚本修改**

```bash
git add start.sh
git commit -m "feat: add uvicorn ws-ping-interval and timeout for WebSocket keepalive"
```

---

## Task 4: 前端自动重连（指数退避）

**Files:**
- Modify: `web/static/chat.js`

- [ ] **Step 1: 在构造函数中添加重连相关属性**

```javascript
constructor() {
    // ... 现有属性
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 2000;  // 初始 2 秒
    this.isReconnecting = false;
}
```

- [ ] **Step 2: 修改 ws.onclose 触发自动重连**

```javascript
this.ws.onclose = () => {
    this.connected = false;
    this.stopHeartbeat();
    this.updateStatus(false, 'Disconnected');

    if (!this.isReconnecting && this.reconnectAttempts < this.maxReconnectAttempts) {
        this.attemptReconnect();
    } else if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        this.addSystemMessage('连接失败，请刷新页面。');
    } else {
        this.addSystemMessage('Connection closed. Refresh to reconnect.');
    }
};
```

- [ ] **Step 3: 实现 attemptReconnect 方法**

在 `stopHeartbeat` 方法后添加：

```javascript
attemptReconnect() {
    this.isReconnecting = true;
    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    this.updateStatus(false, `重新连接中... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    setTimeout(() => {
        this.connect();
        this.isReconnecting = false;
    }, delay);
}
```

- [ ] **Step 4: 在 connect 成功后重置重连计数**

在 `ws.onopen` 中增加：

```javascript
this.ws.onopen = () => {
    this.connected = true;
    this.updateStatus(true, 'Connected');
    this.startHeartbeat();
    this.reconnectAttempts = 0;  // 新增
    this.reconnectDelay = 2000;  // 新增
};
```

- [ ] **Step 5: 提交自动重连功能**

```bash
git add web/static/chat.js
git commit -m "feat: add exponential backoff auto-reconnect for WebSocket"
```

---

## Task 5: 前端 localStorage 存储 session_id

**Files:**
- Modify: `web/static/chat.js`

- [ ] **Step 1: 在 handleMessage 的 session case 中存储 session_id**

```javascript
case 'session':
    this.sessionId = data.session_id;
    this.sessionDisplay.textContent = `Session: ${data.session_id}`;
    localStorage.setItem('easy_agent_session_id', data.session_id);  // 新增

    if (data.resumed) {
        this.addSystemMessage(`已恢复会话 (Session: ${data.session_id})`);
    } else {
        this.addSystemMessage(`Connected to ${data.provider} (Session: ${data.session_id})`);
    }
    break;
```

- [ ] **Step 2: 提交 localStorage 存储**

```bash
git add web/static/chat.js
git commit -m "feat: store session_id in localStorage for reconnection"
```

---

## Task 6: 前端 resume 握手

**Files:**
- Modify: `web/static/chat.js`

- [ ] **Step 1: 在 connect 方法中检查是否有旧 session_id**

在 `this.ws.onopen` 回调中，发送 session 消息前增加 resume 逻辑：

```javascript
this.ws.onopen = () => {
    this.connected = true;
    this.updateStatus(true, 'Connected');
    this.startHeartbeat();
    this.reconnectAttempts = 0;
    this.reconnectDelay = 2000;

    // 尝试恢复旧 session
    const oldSessionId = localStorage.getItem('easy_agent_session_id');
    if (oldSessionId) {
        this.ws.send(JSON.stringify({
            type: 'resume',
            session_id: oldSessionId
        }));
    }
};
```

- [ ] **Step 2: 提交 resume 握手逻辑**

```bash
git add web/static/chat.js
git commit -m "feat: send resume request with old session_id on reconnect"
```

---

## Task 7: 后端 session_last_active 时间戳管理

**Files:**
- Modify: `web/app.py`

- [ ] **Step 1: 在 create_app 中初始化 session_last_active 字典**

在 `sessions: dict = {}` 后添加：

```python
# Store active sessions
sessions: dict = {}
session_last_active: dict = {}  # 新增
```

- [ ] **Step 2: 在每次收到消息时更新时间戳**

在 `while True` 循环开始处，`data = await websocket.receive_json()` 后添加：

```python
while True:
    # Receive message
    data = await websocket.receive_json()
    msg_type = data.get("type", "message")

    # 更新 session 活跃时间
    if session_id in sessions:
        import time
        session_last_active[session_id] = time.time()
```

- [ ] **Step 3: 提交时间戳管理**

```bash
git add web/app.py
git commit -m "feat: track session last active timestamp"
```

---

## Task 8: 后端 resume 逻辑实现

**Files:**
- Modify: `web/app.py`

- [ ] **Step 1: 在消息循环中增加 resume 类型处理**

在 `if msg_type == "message"` 分支前增加：

```python
if msg_type == "resume":
    old_session_id = data.get("session_id")
    import time

    # 检查旧 session 是否存在且未过期（2小时内）
    if (old_session_id and
        old_session_id in sessions and
        old_session_id in session_last_active and
        time.time() - session_last_active[old_session_id] < 7200):

        # 恢复旧 session
        session_id = old_session_id
        graph_config = {"configurable": {"thread_id": session_id}}
        state = sessions[session_id]

        await websocket.send_json({
            "type": "session",
            "session_id": session_id,
            "provider": config.active_provider,
            "resumed": True,
        })
        continue
    else:
        # 创建新 session
        session_id = str(uuid.uuid4())[:8]
        graph_config = {"configurable": {"thread_id": session_id}}
        state = {
            "messages": [],
            "tools": [],
            "context_window": {
                "used_tokens": 0,
                "max_tokens": 128000,
                "threshold": config.agent["context_compression_threshold"],
            },
            "memory_context": "",
            "user_profile": "",
            "agent_notes": "",
            "pending_human_input": False,
            "iteration_count": 0,
            "max_iterations": config.agent["max_iterations"],
            "nudge_counter": 0,
            "provider_name": config.active_provider,
            "session_id": session_id,
        }
        sessions[session_id] = state
        session_last_active[session_id] = time.time()

        await websocket.send_json({
            "type": "session",
            "session_id": session_id,
            "provider": config.active_provider,
            "resumed": False,
        })
        continue
```

- [ ] **Step 2: 提交 resume 逻辑**

```bash
git add web/app.py
git commit -m "feat: implement session resume with 2-hour expiry check"
```

---

## Task 9: AI 分析框架 - 接待前准备场景

**Files:**
- Modify: `src/nodes/agent_node.py`

- [ ] **Step 1: 在 SYSTEM_PROMPT 中增加分析框架章节**

在 `## Guidelines` 章节后添加：

```python
## 分析框架

根据场景选择对应的分析结构输出，禁止泛泛而谈。

### 场景一：接待前准备
触发条件：有客户档案数据（mcp_results 非空），用户询问客户情况

输出结构：
【客户快照】姓名、年龄、会员等级、累计消费
【上次到店】时间、项目、满意度
【本次预判】根据间隔和偏好推测可能需求
【接待建议】1-3 条具体提示（如：注意过敏史、可推荐XX项目）

### 场景二：项目/产品咨询
触发条件：知识库有命中结果

输出结构：
【核心功效】2-3 句简洁描述
【适合人群】具体描述（肤质、年龄、问题类型）
【注意事项】禁忌或特殊要求
【推荐理由】如有客户数据，结合客户肤质/需求说明为何推荐

### 场景三：消费潜力分析
触发条件：有消费/档案数据，用户询问消费/潜力/推荐

输出结构：
【消费行为摘要】频次、客单价、偏好品类
【规律洞察】消费周期、季节偏好、升单趋势
【潜在需求】基于数据推断的未满足需求
【行动建议】推荐套餐或下一步跟进动作

### 场景四：回店/流失预警
触发条件：有行为数据，用户询问回店/流失/跟进

输出结构：
【最后到店】时间和项目
【到店间隔趋势】是否在拉长
【流失风险】高 / 中 / 低（附判断依据）
【建议话术】1-2 句具体的跟进话术示例

**禁止语：** 禁止输出"可能会回店"、"建议关注"等无实质内容的泛泛表述。
```

- [ ] **Step 2: 提交分析框架 prompt**

```bash
git add src/nodes/agent_node.py
git commit -m "feat: add structured analysis framework to agent system prompt"
```

---

## Task 10: 测试心跳和重连

**Files:**
- 无需创建文件（手动测试）

- [ ] **Step 1: 启动服务**

```bash
./stop.sh && ./start.sh
```

- [ ] **Step 2: 打开浏览器控制台，观察心跳**

打开 http://localhost:8080，打开开发者工具 Network 标签，筛选 WS，观察每 30 秒是否有 ping/pong 消息。

- [ ] **Step 3: 测试自动重连**

在控制台执行 `chatClient.ws.close()`，观察是否自动重连，状态栏是否显示"重新连接中..."，重连成功后是否能继续对话。

- [ ] **Step 4: 测试 session 恢复**

发送一条消息，然后刷新页面，观察是否能恢复之前的 session_id（localStorage 中应有值），对话历史是否保留。

---

## Task 11: 测试 AI 分析深度

**Files:**
- 无需创建文件（手动测试）

- [ ] **Step 1: 测试接待前准备场景**

输入："帮我查一下手机号 18102481137 的客户资料，准备接待"

预期输出包含：
- 【客户快照】
- 【上次到店】
- 【本次预判】
- 【接待建议】

- [ ] **Step 2: 测试消费潜力分析场景**

输入："分析一下用户 64924 的消费潜力"

预期输出包含：
- 【消费行为摘要】
- 【规律洞察】
- 【潜在需求】
- 【行动建议】

- [ ] **Step 3: 测试项目咨询场景**

输入："嫩小白项目的功效是什么？"

预期输出包含：
- 【核心功效】
- 【适合人群】
- 【注意事项】

---

## Task 12: 最终提交和文档更新

**Files:**
- 无

- [ ] **Step 1: 确认所有改动已提交**

```bash
git status
```

- [ ] **Step 2: 推送到远程仓库（如有）**

```bash
git push
```

---

## 完成标准

- [ ] WebSocket 每 30 秒发送心跳，长时间无操作不断连
- [ ] 断开后自动重连，指数退避，最多 5 次
- [ ] 重连后恢复旧 session，对话历史保留
- [ ] AI 输出结构化分析，包含固定段落
- [ ] 所有代码已提交
