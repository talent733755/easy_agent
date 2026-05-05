"""Test knowledge RAG functionality."""
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from langchain_core.embeddings import Embeddings


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
        # Simple hash-based embedding for reproducibility
        np.random.seed(hash(text) % (2**32))
        return np.random.rand(1536).tolist()


def test_knowledge_rag_index_and_search():
    """Should index markdown files and search them."""
    # Create temp knowledge base
    with tempfile.TemporaryDirectory() as tmpdir:
        kb_dir = Path(tmpdir) / "test_kb"
        kb_dir.mkdir()

        # Create test files
        (kb_dir / "product1.md").write_text("# 产品A\n\n这是一款保湿精华，适合干性肌肤。")
        (kb_dir / "product2.md").write_text("# 产品B\n\n这是一款控油爽肤水，适合油性肌肤。")

        # Initialize RAG with fake embeddings
        from src.tools.beauty.knowledge_rag import KnowledgeRAG
        rag = KnowledgeRAG(str(kb_dir), embeddings=FakeEmbeddings())
        rag.build_index()

        # Search
        results = rag.search("干性肌肤适合什么", top_k=1)

        assert len(results) == 1
        assert "保湿精华" in results[0]["content"]
