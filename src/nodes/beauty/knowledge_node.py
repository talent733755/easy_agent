"""Knowledge retrieval node for beauty agent."""
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings

from src.state import AgentState
from src.tools.beauty.knowledge_rag import KnowledgeRAG


def _has_documents(knowledge_dir: Path) -> bool:
    """Check if knowledge directory has any markdown files."""
    return bool(list(knowledge_dir.rglob("*.md")))


def _build_index_safe(rag: KnowledgeRAG, knowledge_dir: Path) -> bool:
    """Build index safely, returning False if no documents exist."""
    if not _has_documents(knowledge_dir):
        return False
    rag.build_index()
    return True


def knowledge_retrieve_node(
    state: AgentState,
    knowledge_dir: str,
    index_path: Optional[str] = None,
    top_k: int = 3,
) -> dict:
    """Retrieve knowledge from RAG system based on user query.

    This node:
    1. Checks intent from state (only processes "knowledge_query" or "mixed")
    2. Gets query from last user message
    3. Initializes KnowledgeRAG with knowledge_dir
    4. Tries to load existing index, otherwise builds new index
    5. Calls rag.search(query, top_k=3)
    6. Returns {"knowledge_results": results}

    Args:
        state: Current agent state containing messages and intent
        knowledge_dir: Directory containing knowledge markdown files
        index_path: Optional path to pre-built FAISS index
        top_k: Number of results to return (default: 3)

    Returns:
        dict with knowledge_results key containing list of search results
    """
    # Step 1: Check intent - only process knowledge_query or mixed
    intent = state.get("intent", "")
    if intent not in ("knowledge_query", "mixed"):
        return {"knowledge_results": []}

    # Step 2: Get query from last user message
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    if not last_user_msg:
        return {"knowledge_results": []}

    # Step 3: Initialize KnowledgeRAG
    kb_path = Path(knowledge_dir)
    if not kb_path.exists():
        return {"knowledge_results": []}

    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        openai_api_base="https://api.siliconflow.cn/v1",
        openai_api_key="sk-pzjxmvkbjgscxidtcuqrgaxngfktlfctlhfwjpmwyfkdztgy",
    )
    rag = KnowledgeRAG(str(kb_path), embeddings=embeddings)

    # Step 4: Try to load existing index or build new one
    # Determine which index path to use
    actual_index_path = index_path if index_path else str(kb_path / ".faiss_index")
    index_file = Path(actual_index_path)

    # Try to load existing index
    if index_file.exists():
        try:
            rag.load_index(str(index_file))
        except Exception:
            # Loading failed, need to build new index
            if not _build_index_safe(rag, kb_path):
                return {"knowledge_results": []}
    else:
        # No existing index, build new one
        if not _build_index_safe(rag, kb_path):
            return {"knowledge_results": []}

    # Step 5: Search for relevant knowledge
    try:
        results = rag.search(last_user_msg, top_k=top_k)
    except Exception:
        results = []

    # Step 6: Return results
    return {"knowledge_results": results}
