from langchain_core.tools import tool


@tool
def todo_write(tasks: str) -> str:
    """Manage a task list. Provide tasks as a JSON list of {id, title, status}."""
    return f"[todo_write] Tasks recorded: {tasks}"


@tool
def memory_search(query: str) -> str:
    """Search the agent's memory/history for relevant past interactions."""
    return f"[memory_search] Searching memory for: '{query}' — connected to FTS5/vector store at runtime."
