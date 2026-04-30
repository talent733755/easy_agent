from src.state import AgentState
from src.memory.file_memory import FileMemory
from src.memory.fts5_store import FTS5Store
from src.memory.vector_store import VectorStore


def memory_retrieve_node(state: AgentState, data_dir: str = "~/.easy_agent") -> dict:
    fm = FileMemory(data_dir)
    fts5 = FTS5Store(f"{data_dir}/history.db")
    vs = VectorStore(f"{data_dir}/vectors")

    user_profile = fm.read_user()
    agent_notes = fm.read_memory()

    # Get last user message
    last_user_msg = ""
    for m in reversed(state["messages"]):
        if hasattr(m, "content") and not hasattr(m, "tool_call_id"):
            last_user_msg = str(m.content)
            break

    # Search history
    fts5_results = fts5.search(last_user_msg, limit=3) if last_user_msg else []
    vs_results = vs.search(last_user_msg, top_k=3) if last_user_msg else []

    # Build memory context
    parts = []
    if fts5_results:
        parts.append("## Recent Related History\n")
        for r in fts5_results:
            parts.append(f"- [{r['role']}]: {r['content'][:200]}")
    if vs_results:
        parts.append("\n## Semantically Similar\n")
        for r in vs_results:
            meta = r.get("metadata", {})
            text = meta.get("_text", str(meta)[:200])
            parts.append(f"- {text}")

    return {
        "memory_context": "\n".join(parts),
        "user_profile": user_profile,
        "agent_notes": agent_notes,
    }