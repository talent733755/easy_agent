from langchain_core.messages import HumanMessage
from src.state import AgentState


def observe_node(state: AgentState) -> dict:
    """Format tool results as observations for the next agent iteration."""
    messages = state["messages"]
    # Find the most recent tool messages
    recent = messages[-4:] if len(messages) >= 4 else messages
    tool_results = [m for m in recent if hasattr(m, "tool_call_id")]

    if tool_results:
        summary = "\n".join(
            f"[{m.tool_call_id[:8]}]: {str(m.content)[:200]}"
            for m in tool_results
        )
        observation = HumanMessage(
            content=f"[Observation]\n{summary}\n\nContinue with your next action or respond to the user."
        )
        return {"messages": [observation]}
    return {}