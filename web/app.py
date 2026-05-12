"""FastAPI application with WebSocket chat endpoint."""

import json
import uuid
import asyncio
import time
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from src.config import load_config
from src.graph import build_graph
from src.training_graph import build_training_graph
from src.memory.conversation_store import ConversationStore


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Create FastAPI application.

    Args:
        config_path: Optional path to config.yaml

    Returns:
        FastAPI app instance
    """
    app = FastAPI(title="Easy Agent", version="0.1.0")

    # Load configuration
    config = load_config(config_path)

    # Setup static files and templates
    web_dir = Path(__file__).parent
    app.mount("/static", StaticFiles(directory=web_dir / "static"), name="static")
    templates = Jinja2Templates(directory=web_dir / "templates")

    # Store active sessions
    sessions: dict = {}
    # Track session last active timestamp for expiry check
    session_last_active: dict = {}

    # Build graphs with checkpointer (one per graph-type agent)
    checkpointer = MemorySaver()
    graphs = {
        "beauty": build_graph(checkpointer=checkpointer),
        "training": build_training_graph(checkpointer=checkpointer),
    }

    # Conversation store
    import os
    data_dir = os.path.expanduser(config.memory.get("data_dir", "~/.easy_agent"))
    conv_store = ConversationStore(os.path.join(data_dir, "conversations.db"))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Render chat page."""
        # Read HTML file directly to avoid Jinja2 cache issues
        html_path = web_dir / "templates" / "index.html"
        return HTMLResponse(content=html_path.read_text())

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "version": "0.1.0",
                "provider": config.active_provider,
            }
        )

    @app.get("/api/agents")
    async def get_agents():
        """Return list of available agents."""
        agents_list = []
        for aid, acfg in config.agents.items():
            agents_list.append({
                "id": aid,
                "name": acfg.name,
                "icon": acfg.icon,
                "badge": acfg.badge,
                "description": acfg.description,
                "type": acfg.type,
            })
        return JSONResponse(content={"agents": agents_list})

    @app.get("/api/conversations")
    async def list_conversations(agent_id: str | None = None):
        """List conversations, optionally filtered by agent."""
        convs = conv_store.get_conversations(agent_id=agent_id)
        # Group by time
        now = time.time()
        today_start = now - (now % 86400)
        yesterday_start = today_start - 86400
        grouped = {"today": [], "yesterday": [], "earlier": []}
        for c in convs:
            title = c["title"] or "新对话"
            item = {
                "id": c["id"],
                "agent_id": c["agent_id"],
                "title": title,
                "time": time.strftime("%H:%M", time.localtime(c["updated_at"])),
                "date": time.strftime("%Y/%m/%d", time.localtime(c["updated_at"])),
            }
            if c["updated_at"] >= today_start:
                grouped["today"].append(item)
            elif c["updated_at"] >= yesterday_start:
                grouped["yesterday"].append(item)
            else:
                grouped["earlier"].append(item)
        return JSONResponse(content={"conversations": grouped})

    @app.delete("/api/conversations/{conv_id}")
    async def delete_conversation(conv_id: str):
        """Delete a conversation and its messages."""
        ok = conv_store.delete_conversation(conv_id)
        return JSONResponse(content={"deleted": ok})

    @app.get("/api/conversations/{conv_id}/messages")
    async def get_messages(conv_id: str):
        """Get all messages in a conversation."""
        msgs = conv_store.get_messages(conv_id)
        return JSONResponse(content={"messages": msgs})

    @app.post("/api/conversations/new")
    async def new_conversation(request: Request):
        """Create a new conversation."""
        body = await request.json()
        agent_id = body.get("agent_id", "beauty")
        conv_id = conv_store.create_conversation(agent_id=agent_id)
        return JSONResponse(content={"id": conv_id, "agent_id": agent_id})

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        nonlocal graphs
        """WebSocket endpoint for real-time chat.

        Message format (JSON):
        - Incoming: {"type": "message", "content": "user text"}
        - Outgoing: {"type": "response", "content": "agent text"}
        - Outgoing: {"type": "error", "content": "error message"}
        - Outgoing: {"type": "tool_request", "tool": "name", "args": {...}}
        - Incoming: {"type": "tool_response", "approved": true/false}
        """
        await websocket.accept()

        # Session expiry time: 2 hours
        SESSION_EXPIRY_SECONDS = 2 * 60 * 60

        # Generate session ID
        session_id = str(uuid.uuid4())[:8]
        graph_config = {"configurable": {"thread_id": session_id}}
        resumed = False

        # Wait for first message - check for resume request
        first_data = await websocket.receive_json()
        first_msg_type = first_data.get("type", "message")

        # Track current conversation and agent
        current_conv_id = None
        current_agent_id = first_data.get("agent_id", "beauty")

        if first_msg_type == "resume":
            old_session_id = first_data.get("session_id")
            if old_session_id and old_session_id in sessions:
                # Check if session is not expired
                last_active = session_last_active.get(old_session_id, 0)
                if time.time() - last_active < SESSION_EXPIRY_SECONDS:
                    # Resume old session
                    session_id = old_session_id
                    graph_config = {"configurable": {"thread_id": session_id}}
                    state = sessions[session_id]
                    resumed = True
                    current_conv_id = state.get("conversation_id")
                else:
                    # Session expired, clean up
                    if old_session_id in sessions:
                        del sessions[old_session_id]
                    if old_session_id in session_last_active:
                        del session_last_active[old_session_id]

        # Initialize state if not resumed
        if not resumed:
            state: dict = {
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
                # Training fields
                "training_phase": "",
                "training_scenario": "",
                "training_context": {},
                "training_score": {},
                "training_history": [],
            }
            sessions[session_id] = state

        try:
            # Send session info
            await websocket.send_json({
                "type": "session",
                "session_id": session_id,
                "provider": config.active_provider,
                "resumed": resumed,
            })

            # Update session last active timestamp
            session_last_active[session_id] = time.time()

            # Process first message if it's not a resume request
            data = first_data
            msg_type = first_msg_type
            first_message = True

            while True:
                # Only receive new message if we've already processed the first one
                if first_message:
                    first_message = False
                else:
                    # Receive message
                    data = await websocket.receive_json()
                    msg_type = data.get("type", "message")

                # Update session last active timestamp
                session_last_active[session_id] = time.time()

                # Handle ping/pong heartbeat
                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                # Handle resume request (only valid as first message)
                if msg_type == "resume":
                    # Ignore resume requests after first message
                    continue

                if msg_type == "message":
                    user_input = data.get("content", "").strip()
                    if not user_input:
                        continue

                    # Update agent and conversation context from message
                    msg_agent_id = data.get("agent_id", current_agent_id)
                    if msg_agent_id and msg_agent_id != current_agent_id:
                        current_agent_id = msg_agent_id
                        current_conv_id = None
                    if data.get("conversation_id"):
                        current_conv_id = data["conversation_id"]

                    # Handle slash commands
                    if user_input.startswith("/"):
                        cmd_response = await handle_command(user_input, state)
                        await websocket.send_json({
                            "type": "response",
                            "content": cmd_response,
                        })
                        continue

                    # Check if this is a workflow agent (external API)
                    agent_cfg = config.agents.get(current_agent_id)
                    if agent_cfg and agent_cfg.type == "workflow":
                        await handle_workflow_agent(websocket, agent_cfg, user_input, current_conv_id)
                        continue

                    # Select the correct graph for this agent
                    active_graph = graphs.get(current_agent_id) or graphs.get("beauty")

                    # Add user message
                    state["messages"].append(HumanMessage(content=user_input))
                    state["iteration_count"] = 0

                    # Create conversation on first message, or use existing one
                    if not current_conv_id:
                        current_conv_id = conv_store.create_conversation(
                            agent_id=current_agent_id,
                            title=user_input[:30],
                        )
                        state["conversation_id"] = current_conv_id
                    # Save user message to conversation store
                    conv_store.add_message(current_conv_id, "user", user_input)

                    # Node name to display label mapping
                    NODE_LABELS = {
                        "intent_classify": "🔍 分析意图...",
                        "knowledge_retrieve": "📚 检索知识库...",
                        "memory_retrieve": "🧠 检索记忆...",
                        "agent": "🤔 生成回复...",
                        "tool_executor": "🔧 执行工具...",
                        "observe": "👁 观察结果...",
                        "human_gate": "🚦 检查审批...",
                        "memory_learn": "💾 更新记忆...",
                        "nudge_check": "📊 检查迭代...",
                        "memory_save": "💾 保存记忆...",
                        # Training nodes
                        "welcome": " 欢迎...",
                        "wait_input": "⏳ 等待输入...",
                        "router": "🔀 路由中...",
                        "setup": "⚙️ 对练设置...",
                        "roleplay": " 对练对话中...",
                        "evaluate": "📊 评价中...",
                    }

                    async def send_progress(node_name: str):
                        # Check for MCP service nodes
                        label = NODE_LABELS.get(node_name)
                        if label is None and node_name.startswith("mcp_"):
                            svc = node_name[4:]
                            label = f"🔌 查询 {svc} 服务..."
                        if label:
                            await websocket.send_json({
                                "type": "progress",
                                "content": label,
                            })

                    try:
                        # Stream graph execution for progress tracking
                        result = None
                        async for chunk in active_graph.astream(state, graph_config, stream_mode="updates"):
                            for node_name, node_output in chunk.items():
                                await send_progress(node_name)
                                result = node_output

                        # Re-fetch final state if needed
                        final_state = active_graph.get_state(graph_config)
                        if final_state and final_state.values:
                            result = final_state.values

                        # Handle interrupts (tool approval)
                        while True:
                            gs = active_graph.get_state(graph_config)
                            if not gs.next:
                                break
                            tasks = gs.tasks
                            if not tasks or not tasks[0].interrupts:
                                break

                            interrupt_val = tasks[0].interrupts[0].value
                            itype = interrupt_val.get("type", "")

                            if itype == "tool_approval":
                                # Send tool approval request to client
                                await websocket.send_json({
                                    "type": "tool_request",
                                    "tool": interrupt_val["tool"],
                                    "args": interrupt_val["args"],
                                })

                                # Wait for client response
                                response = await websocket.receive_json()
                                if response.get("type") == "tool_response":
                                    approved = response.get("approved", False)
                                    from langgraph.types import Command
                                    async for chunk in active_graph.astream(
                                        Command(resume={"approved": approved}),
                                        graph_config,
                                        stream_mode="updates",
                                    ):
                                        for node_name in chunk:
                                            await send_progress(node_name)
                                    final_state = active_graph.get_state(graph_config)
                                    if final_state and final_state.values:
                                        result = final_state.values
                            else:
                                from langgraph.types import Command
                                async for chunk in active_graph.astream(Command(resume={}), graph_config, stream_mode="updates"):
                                    for node_name in chunk:
                                        await send_progress(node_name)
                                final_state = active_graph.get_state(graph_config)
                                if final_state and final_state.values:
                                    result = final_state.values

                        # Find and send AI response
                        for m in reversed(result["messages"]):
                            if isinstance(m, AIMessage) and m.content:
                                # content 可能是字符串或 content blocks 列表
                                content = m.content
                                if isinstance(content, list):
                                    # 提取 text 块，跳过 thinking 等
                                    text_parts = [
                                        block["text"] for block in content
                                        if isinstance(block, dict) and block.get("type") == "text"
                                    ]
                                    content = "\n".join(text_parts) if text_parts else str(content)
                                await websocket.send_json({
                                    "type": "response",
                                    "content": content,
                                })
                                # Save assistant message to conversation store
                                if current_conv_id:
                                    conv_store.add_message(current_conv_id, "assistant", content)
                                break

                        # Update state
                        state = result
                        state["messages"] = [
                            m for m in state["messages"]
                            if not (isinstance(m, AIMessage) and m.tool_calls)
                        ]
                        state["nudge_counter"] = state.get("nudge_counter", 0) + 1
                        sessions[session_id] = state

                    except Exception as e:
                        import traceback
                        print(f"[ERROR] {type(e).__name__}: {e}")
                        traceback.print_exc()
                        # Try fallback provider
                        if hasattr(config, 'fallback_provider') and config.fallback_provider != config.active_provider:
                            await websocket.send_json({
                                "type": "response",
                                "content": f"Error with {config.active_provider}, switching to {config.fallback_provider}...",
                            })
                            config.active_provider = config.fallback_provider
                            graphs["beauty"] = build_graph(checkpointer=checkpointer)
                            graphs["training"] = build_training_graph(checkpointer=checkpointer)
                        else:
                            await websocket.send_json({
                                "type": "error",
                                "content": str(e),
                            })

                elif msg_type == "tool_response":
                    # This should be handled in the tool approval loop above
                    pass

        except WebSocketDisconnect:
            # Don't delete session on disconnect - allow resume within expiry window
            # Session will be cleaned up when expired (SESSION_EXPIRY_SECONDS = 2 hours)
            pass

        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "content": str(e),
            })

    async def handle_workflow_agent(
        websocket: WebSocket,
        agent_cfg,
        user_input: str,
        conv_id: str | None,
    ):
        """Handle a workflow agent request (external API call)."""
        # Create or use conversation
        nonlocal conv_store
        if not conv_id:
            conv_id = conv_store.create_conversation(
                agent_id=agent_cfg.name,
                title=user_input[:30],
            )
        conv_store.add_message(conv_id, "user", user_input)

        # Send progress
        await websocket.send_json({
            "type": "progress",
            "content": f"调用 {agent_cfg.name} 工作流...",
        })

        try:
            import httpx
            headers = {}
            if agent_cfg.api_key:
                headers["Authorization"] = f"Bearer {agent_cfg.api_key}"
            # Coze API uses {"topic": "..."} as input
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    agent_cfg.endpoint,
                    json={"topic": user_input},
                    headers=headers,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    # Coze XHS workflow returns: {note_content: "...", image_url: "..."}
                    note = result.get("note_content", "")
                    img = result.get("image_url", "")
                    reply = note
                    if img:
                        reply += f"\n\n图片: {img}"
                else:
                    reply = f"工作流调用失败 (HTTP {resp.status_code}): {resp.text}"

            await websocket.send_json({
                "type": "response",
                "content": reply,
            })
            conv_store.add_message(conv_id, "assistant", reply)

        except Exception as e:
            error_msg = f"工作流调用异常: {str(e)}"
            await websocket.send_json({
                "type": "error",
                "content": error_msg,
            })

    async def handle_command(cmd: str, state: dict) -> str:
        """Handle slash commands."""
        cmd_lower = cmd.strip().lower()

        if cmd_lower in ("/quit", "/exit", "/q"):
            return "Use the close button to exit the chat."
        elif cmd_lower in ("/help", "/h"):
            return """
**Commands:**
- `/help`, `/h` — Show this help
- `/memory` — Show current MEMORY.md
- `/profile` — Show current USER.md
- `/session` — Show session ID
- `/clear` — Start a new session
"""
        elif cmd_lower == "/memory":
            from src.memory.file_memory import FileMemory
            fm = FileMemory(config.memory["data_dir"])
            content = fm.read_memory() or "(empty)"
            return f"**MEMORY.md:**\n{content}"
        elif cmd_lower == "/profile":
            from src.memory.file_memory import FileMemory
            fm = FileMemory(config.memory["data_dir"])
            content = fm.read_user() or "(empty)"
            return f"**USER.md:**\n{content}"
        elif cmd_lower == "/session":
            return f"**Session ID:** {state.get('session_id', 'unknown')}"
        elif cmd_lower == "/clear":
            state["messages"] = []
            state["iteration_count"] = 0
            return "Session cleared. Starting fresh."
        else:
            return f"Unknown command: {cmd}"

    return app