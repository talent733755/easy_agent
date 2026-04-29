from langgraph.graph import StateGraph, MessagesState
from src.state import AgentState


def test_agent_state_keys():
    """Verify all required keys exist in AgentState."""
    required_keys = {
        "messages", "tools", "context_window", "memory_context",
        "user_profile", "agent_notes", "pending_human_input",
        "iteration_count", "max_iterations", "nudge_counter", "provider_name",
    }
    # AgentState inherits from MessagesState which provides 'messages'
    annotations = AgentState.__annotations__
    for key in required_keys:
        assert key in annotations, f"Missing key: {key}"


def test_agent_state_defaults():
    """Verify AgentState can be constructed with defaults."""
    state = AgentState(
        messages=[],
        tools=[],
        context_window={"used_tokens": 0, "max_tokens": 128000, "threshold": 0.7},
        memory_context="",
        user_profile="",
        agent_notes="",
        pending_human_input=False,
        iteration_count=0,
        max_iterations=15,
        nudge_counter=0,
        provider_name="zhipu",
    )
    assert state["iteration_count"] == 0
    assert state["max_iterations"] == 15
    assert state["nudge_counter"] == 0
    assert state["pending_human_input"] is False
