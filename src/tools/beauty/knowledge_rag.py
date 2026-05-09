"""Knowledge RAG tool for beauty domain."""
from pathlib import Path
from typing import Any, Optional

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class KnowledgeRAG:
    """Manages knowledge base indexing and retrieval."""

    def __init__(
        self,
        knowledge_dir: str,
        index_name: str = "beauty_kb",
        embeddings: Optional[Embeddings] = None,
    ):
        self.knowledge_dir = Path(knowledge_dir)
        self.index_name = index_name
        self.embeddings = embeddings or OpenAIEmbeddings(
            model="BAAI/bge-m3",
            openai_api_base="https://api.siliconflow.cn/v1",
            openai_api_key="sk-pzjxmvkbjgscxidtcuqrgaxngfktlfctlhfwjpmwyfkdztgy",
        )
        self.vectorstore: FAISS | None = None

    def build_index(self) -> None:
        """Build FAISS index from all markdown files in knowledge_dir."""
        documents = []

        # Load all .md files
        for md_file in self.knowledge_dir.rglob("*.md"):
            loader = TextLoader(str(md_file), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = str(md_file.relative_to(self.knowledge_dir))
            documents.extend(docs)

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        splits = text_splitter.split_documents(documents)

        # Build FAISS index
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Search knowledge base and return top-k results."""
        if self.vectorstore is None:
            raise ValueError("Index not built. Call build_index() first.")

        results = self.vectorstore.similarity_search(query, k=top_k)

        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
            }
            for doc in results
        ]

    def save_index(self, path: str) -> None:
        """Save FAISS index to disk."""
        if self.vectorstore is None:
            raise ValueError("Index not built.")

        self.vectorstore.save_local(path)

    def load_index(self, path: str) -> None:
        """Load FAISS index from disk."""
        self.vectorstore = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
