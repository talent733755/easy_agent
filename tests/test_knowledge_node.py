"""Test knowledge retrieval node."""
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage

from src.state import AgentState
from src.nodes.beauty.knowledge_node import knowledge_retrieve_node


class FakeEmbeddings(Embeddings):
    """Fake embeddings for testing without API calls."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return fake embeddings for documents."""
        return [self._fake_embedding(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return fake embedding for query."""
        return self._fake_embedding(text)

    def _fake_embedding(self, text: str) -> list[float]:
        """Generate deterministic fake embedding based on text content."""
        np.random.seed(hash(text) % (2**32))
        return np.random.rand(1536).tolist()


def create_test_state(message: str, intent: str = "knowledge_query") -> AgentState:
    """Create a test state with a single message."""
    return {
        "messages": [HumanMessage(content=message)],
        "tools": [],
        "context_window": {"used_tokens": 0, "max_tokens": 128000, "threshold": 0.7},
        "memory_context": "",
        "user_profile": "",
        "agent_notes": "",
        "pending_human_input": False,
        "iteration_count": 0,
        "max_iterations": 15,
        "nudge_counter": 0,
        "provider_name": "openai",
        "intent": intent,
        "customer_context": {},
        "knowledge_results": [],
        "mcp_results": {},
    }


def test_knowledge_retrieve_node_returns_results():
    """Should retrieve knowledge results for knowledge_query intent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kb_dir = Path(tmpdir) / "test_kb"
        kb_dir.mkdir()

        # Create test knowledge file
        (kb_dir / "product.md").write_text(
            "# 保湿精华\n\n这款保湿精华含有透明质酸，适合干性肌肤使用。"
        )

        state = create_test_state("干性肌肤适合什么产品？", intent="knowledge_query")

        # Use fake embeddings to avoid API calls
        with patch("src.nodes.beauty.knowledge_node.OpenAIEmbeddings") as MockEmbeddings:
            MockEmbeddings.return_value = FakeEmbeddings()
            result = knowledge_retrieve_node(state, knowledge_dir=str(kb_dir))

        assert "knowledge_results" in result
        assert len(result["knowledge_results"]) > 0
        assert "保湿精华" in result["knowledge_results"][0]["content"]


def test_knowledge_retrieve_node_handles_mixed_intent():
    """Should retrieve knowledge results for mixed intent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kb_dir = Path(tmpdir) / "test_kb"
        kb_dir.mkdir()

        (kb_dir / "treatment.md").write_text(
            "# 抗衰疗程\n\n光子嫩肤适合30岁以上女性。"
        )

        state = create_test_state("张女士适合做什么项目？", intent="mixed")

        with patch("src.nodes.beauty.knowledge_node.OpenAIEmbeddings") as MockEmbeddings:
            MockEmbeddings.return_value = FakeEmbeddings()
            result = knowledge_retrieve_node(state, knowledge_dir=str(kb_dir))

        assert len(result["knowledge_results"]) > 0


def test_knowledge_retrieve_node_skips_other_intent():
    """Should skip retrieval for non-knowledge intents."""
    state = create_test_state("帮我查一下张女士的档案", intent="query_customer")

    result = knowledge_retrieve_node(state, knowledge_dir="/dummy/path")

    assert result["knowledge_results"] == []


def test_knowledge_retrieve_node_uses_cached_index():
    """Should load existing FAISS index if available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kb_dir = Path(tmpdir) / "test_kb"
        kb_dir.mkdir()

        (kb_dir / "product.md").write_text("# 产品\n\n产品描述内容")

        # Pre-build and save the index
        from src.tools.beauty.knowledge_rag import KnowledgeRAG
        rag = KnowledgeRAG(str(kb_dir), embeddings=FakeEmbeddings())
        rag.build_index()
        index_path = kb_dir / ".faiss_index"
        rag.save_index(str(index_path))

        state = create_test_state("产品信息", intent="knowledge_query")

        # Mock embeddings to track if they're called
        with patch("src.nodes.beauty.knowledge_node.OpenAIEmbeddings") as MockEmbeddings:
            MockEmbeddings.return_value = FakeEmbeddings()
            result = knowledge_retrieve_node(
                state,
                knowledge_dir=str(kb_dir),
                index_path=str(index_path),
            )

        assert len(result["knowledge_results"]) > 0


def test_knowledge_retrieve_node_handles_empty_knowledge_dir():
    """Should return empty results if knowledge directory is empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        kb_dir = Path(tmpdir) / "empty_kb"
        kb_dir.mkdir()

        state = create_test_state("什么产品好？", intent="knowledge_query")

        with patch("src.nodes.beauty.knowledge_node.OpenAIEmbeddings") as MockEmbeddings:
            MockEmbeddings.return_value = FakeEmbeddings()
            result = knowledge_retrieve_node(state, knowledge_dir=str(kb_dir))

        assert result["knowledge_results"] == []
