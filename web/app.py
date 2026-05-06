"""FastAPI application with WebSocket chat endpoint."""

import json
import uuid
import asyncio
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

    # Build graph with checkpointer
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)

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

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time chat.

        Message format (JSON):
        - Incoming: {"type": "message", "content": "user text"}
        - Outgoing: {"type": "response", "content": "agent text"}
        - Outgoing: {"type": "error", "content": "error message"}
        - Outgoing: {"type": "tool_request", "tool": "name", "args": {...}}
        - Incoming: {"type": "tool_response", "approved": true/false}
        """
        await websocket.accept()

        # Generate session ID
        session_id = str(uuid.uuid4())[:8]
        graph_config = {"configurable": {"thread_id": session_id}}

        # Initialize state
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
        }

        sessions[session_id] = state

        try:
            # Send session info
            await websocket.send_json({
                "type": "session",
                "session_id": session_id,
                "provider": config.active_provider,
            })

            while True:
                # Receive message
                data = await websocket.receive_json()
                msg_type = data.get("type", "message")

                if msg_type == "message":
                    user_input = data.get("content", "").strip()
                    if not user_input:
                        continue

                    # Handle slash commands
                    if user_input.startswith("/"):
                        cmd_response = await handle_command(user_input, state)
                        await websocket.send_json({
                            "type": "response",
                            "content": cmd_response,
                        })
                        continue

                    # Add user message
                    state["messages"].append(HumanMessage(content=user_input))
                    state["iteration_count"] = 0

                    try:
                        # Invoke graph
                        result = graph.invoke(state, graph_config)

                        # Handle interrupts (tool approval)
                        while True:
                            gs = graph.get_state(graph_config)
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
                                    result = graph.invoke(
                                        Command(resume={"approved": approved}),
                                        graph_config,
                                    )
                            else:
                                from langgraph.types import Command
                                result = graph.invoke(Command(resume={}), graph_config)

                        # Find and send AI response
                        for m in reversed(result["messages"]):
                            if isinstance(m, AIMessage) and m.content:
                                await websocket.send_json({
                                    "type": "response",
                                    "content": m.content,
                                })
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
                        # Try fallback provider
                        if hasattr(config, 'fallback_provider') and config.fallback_provider != config.active_provider:
                            await websocket.send_json({
                                "type": "response",
                                "content": f"Error with {config.active_provider}, switching to {config.fallback_provider}...",
                            })
                            config.active_provider = config.fallback_provider
                            graph = build_graph(checkpointer=checkpointer)
                        else:
                            await websocket.send_json({
                                "type": "error",
                                "content": str(e),
                            })

                elif msg_type == "tool_response":
                    # This should be handled in the tool approval loop above
                    pass

        except WebSocketDisconnect:
            # Clean up session
            if session_id in sessions:
                del sessions[session_id]

        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "content": str(e),
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