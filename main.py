#!/usr/bin/env python3
"""Easy Agent — A Hermes-style intelligent agent with LangGraph."""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from src.config import load_config
from src.graph import build_graph


def parse_args():
    parser = argparse.ArgumentParser(description="Easy Agent — Hermes-style AI agent")
    parser.add_argument("--provider", "-p", help="LLM provider (openai, zhipu, anthropic)")
    parser.add_argument("--model", "-m", help="Model name override")
    parser.add_argument("--config", "-c", help="Path to config.yaml")
    parser.add_argument("--session", "-s", help="Session ID for resuming", default=None)
    return parser.parse_args()


def print_banner():
    print(r"""
╔══════════════════════════════════════════╗
║         🤖 Easy Agent v0.1.0             ║
║    Hermes-style LangGraph Agent          ║
║    Type /help for commands, /quit to exit║
╚══════════════════════════════════════════╝
""")


def handle_command(user_input: str, state: dict) -> tuple[bool, str, bool]:
    """Handle slash commands. Returns (should_quit, response, is_command)."""
    # Only process commands that start with /
    if not user_input.startswith("/"):
        return False, "", False

    cmd = user_input.strip().lower()

    if cmd in ("/quit", "/exit", "/q"):
        return True, "Goodbye!", True
    elif cmd in ("/help", "/h"):
        return False, """
Commands:
  /help, /h      — Show this help
  /quit, /q      — Exit the agent
  /memory        — Show current MEMORY.md
  /profile       — Show current USER.md
  /session       — Show session ID
  /clear         — Start a new session
""", True
    elif cmd == "/memory":
        from src.memory.file_memory import FileMemory
        config = load_config()
        fm = FileMemory(config.memory["data_dir"])
        return False, f"MEMORY.md:\n{fm.read_memory() or '(empty)'}", True
    elif cmd == "/profile":
        from src.memory.file_memory import FileMemory
        config = load_config()
        fm = FileMemory(config.memory["data_dir"])
        return False, f"USER.md:\n{fm.read_user() or '(empty)'}", True
    elif cmd == "/session":
        import uuid
        return False, f"Session ID: {state.get('session_id', 'unknown')}", True
    elif cmd == "/clear":
        state["messages"] = []
        state["iteration_count"] = 0
        return False, "Session cleared. Starting fresh.", True
    else:
        return False, f"Unknown command: {cmd}", True


def run_agent(args):
    config = load_config(args.config)

    # Override provider if specified
    if args.provider:
        config.active_provider = args.provider

    # Build graph with checkpointer
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)

    import uuid
    session_id = args.session or str(uuid.uuid4())[:8]
    graph_config = {"configurable": {"thread_id": session_id}}

    print_banner()
    print(f"Provider: {config.active_provider} | Session: {session_id}")
    print()

    # Initial state
    state: dict = {
        "messages": [],
        "tools": [],
        "context_window": {"used_tokens": 0, "max_tokens": 128000, "threshold": config.agent["context_compression_threshold"]},
        "memory_context": "",
        "user_profile": "",
        "agent_notes": "",
        "pending_human_input": False,
        "iteration_count": 0,
        "max_iterations": config.agent["max_iterations"],
        "nudge_counter": 0,
        "provider_name": config.active_provider,
    }

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if not user_input:
                continue
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        # Handle commands
        should_quit, response, is_command = handle_command(user_input, state)
        if should_quit:
            print(response)
            break
        if is_command:
            print(f"\n{response}")
            continue

        # Add user message and invoke graph
        state["messages"].append(HumanMessage(content=user_input))
        state["iteration_count"] = 0  # reset for new turn

        try:
            result = graph.invoke(state, graph_config)

            # Handle interrupts (e.g. human-gate tool approval)
            while True:
                gs = graph.get_state(graph_config)
                if not gs.next:
                    break
                # Check for pending interrupts
                tasks = gs.tasks
                if not tasks or not tasks[0].interrupts:
                    break
                interrupt_val = tasks[0].interrupts[0].value
                itype = interrupt_val.get("type", "")

                if itype == "tool_approval":
                    print(f"\n⚠️  Dangerous operation:")
                    print(f"  Tool: {interrupt_val['tool']}")
                    print(f"  Args: {interrupt_val['args']}")
                    approval = input("  Approve? [y/n]: ").strip().lower()
                    if approval == "y":
                        from langgraph.types import Command
                        result = graph.invoke(Command(resume={"approved": True}), graph_config)
                    else:
                        result = graph.invoke(Command(resume={"approved": False}), graph_config)
                else:
                    # Unknown interrupt type — resume without action
                    from langgraph.types import Command
                    result = graph.invoke(Command(resume={}), graph_config)

            # Display agent response
            for m in reversed(result["messages"]):
                if isinstance(m, AIMessage) and m.content:
                    print(f"\n🤖 Agent: {m.content}")
                    break

            # Update state for next turn
            state = result

            # Prevent accumulation of tool_call messages across turns
            state["messages"] = [m for m in state["messages"] if not (isinstance(m, AIMessage) and m.tool_calls)]

            # Increment nudge counter
            state["nudge_counter"] = state.get("nudge_counter", 0) + 1

        except Exception as e:
            print(f"\n❌ Error: {e}")
            # Try fallback provider
            if hasattr(config, 'fallback_provider') and config.fallback_provider != config.active_provider:
                print(f"🔄 Switching to fallback provider: {config.fallback_provider}")
                config.active_provider = config.fallback_provider
                graph = build_graph(checkpointer=checkpointer)


if __name__ == "__main__":
    args = parse_args()
    run_agent(args)