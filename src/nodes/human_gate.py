from langchain_core.messages import AIMessage
from src.state import AgentState

# Default risk levels for built-in tools
TOOL_RISK_LEVELS = {
    "read_file": "safe",
    "list_dir": "safe",
    "get_time": "safe",
    "memory_search": "safe",
    "write_file": "write",
    "todo_write": "write",
    "web_search": "dangerous",
    "web_fetch": "dangerous",
    "run_shell": "dangerous",
}


def human_gate(state: AgentState) -> dict:
    messages = state["messages"]
    last_ai = None
    for m in reversed(messages):
        if isinstance(m, AIMessage) and hasattr(m, "tool_calls") and m.tool_calls:
            last_ai = m
            break

    if last_ai is None or not last_ai.tool_calls:
        # Check for [HUMAN_INPUT: ...] in last AI message text
        last_text = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                last_text = str(m.content)
                break
        if "[HUMAN_INPUT:" in last_text:
            return {"pending_human_input": True}
        return {"pending_human_input": False}

    for tc in last_ai.tool_calls:
        risk = TOOL_RISK_LEVELS.get(tc.get("name", ""), "safe")
        if risk == "dangerous":
            return {"pending_human_input": True}

    return {"pending_human_input": False}