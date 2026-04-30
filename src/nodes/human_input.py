from langchain_core.messages import HumanMessage
from src.state import AgentState


def human_input_node(state: AgentState, user_response: str = None) -> dict:
    if user_response is None:
        user_response = input("\n👤 Your input: ")

    return {
        "messages": [HumanMessage(content=f"[Human Response]: {user_response}")],
        "pending_human_input": False,
    }