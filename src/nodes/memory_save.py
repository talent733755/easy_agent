from src.state import AgentState
from src.memory.fts5_store import FTS5Store
from src.memory.vector_store import VectorStore


def memory_save_node(state: AgentState, data_dir: str = "~/.easy_agent") -> dict:
    fts5 = FTS5Store(f"{data_dir}/history.db")
    vs = VectorStore(f"{data_dir}/vectors")

    # Save last exchange to FTS5 and vector store
    for m in state["messages"][-4:]:  # last 4 messages
        if hasattr(m, "content") and str(m.content):
            role = getattr(m, "type", "unknown")
            content = str(m.content)
            if len(content) > 20:  # skip very short messages
                fts5.insert(role, content)
                vs.add(content, {"_text": content, "role": role})

    return {}