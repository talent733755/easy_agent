import pickle
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, data_dir: str, max_entries: int = 500, model_name: str = "all-MiniLM-L6-v2"):
        self.dir = Path(data_dir).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.index_path = self.dir / "faiss.index"
        self.meta_path = self.dir / "meta.pkl"
        self._model = None  # lazy load
        self._model_name = model_name
        self._index = None
        self._metadata: list[dict] = []
        self._load()

    @property
    def model(self):
        if self._model is None:
            # Use local_files_only to avoid network requests when model is cached
            self._model = SentenceTransformer(
                self._model_name,
                local_files_only=True
            )
        return self._model

    def _load(self):
        if self.index_path.exists() and self.meta_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "rb") as f:
                self._metadata = pickle.load(f)
        else:
            self._index = faiss.IndexFlatL2(384)  # all-MiniLM-L6-v2 = 384 dims
            self._metadata = []

    def _save(self):
        faiss.write_index(self._index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self._metadata, f)

    def add(self, text: str, metadata: dict = None):
        embedding = self.model.encode([text], normalize_embeddings=True)
        self._index.add(np.array(embedding, dtype=np.float32))
        meta = metadata.copy() if metadata else {}
        meta["_text"] = text  # Store text for potential re-indexing
        self._metadata.append(meta)
        # FIFO eviction
        if len(self._metadata) > self.max_entries:
            self._metadata = self._metadata[-self.max_entries:]
            # Rebuild index for remaining entries
            all_texts = [m.get("_text", "") for m in self._metadata]
            embeddings = self.model.encode(all_texts, normalize_embeddings=True)
            # Create new index and swap
            new_index = faiss.IndexFlatL2(384)
            new_index.add(np.array(embeddings, dtype=np.float32))
            self._index = new_index
        self._save()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if len(self._metadata) == 0:
            return []
        embedding = self.model.encode([query], normalize_embeddings=True)
        distances, indices = self._index.search(np.array(embedding, dtype=np.float32), min(top_k, len(self._metadata)))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self._metadata):
                results.append({"distance": float(dist), "metadata": self._metadata[idx]})
        return results

    def count(self) -> int:
        return len(self._metadata)

    def delete(self, filter_fn) -> int:
        """Delete entries where filter_fn(metadata) returns True.

        Args:
            filter_fn: A function that takes metadata dict and returns True to delete.

        Returns:
            Number of entries deleted.
        """
        if len(self._metadata) == 0:
            return 0

        # Find indices to keep
        indices_to_keep = [i for i, meta in enumerate(self._metadata) if not filter_fn(meta)]
        deleted_count = len(self._metadata) - len(indices_to_keep)

        if deleted_count == 0:
            return 0

        # Rebuild metadata
        self._metadata = [self._metadata[i] for i in indices_to_keep]

        # Rebuild index
        if self._metadata:
            all_texts = [m.get("_text", "") for m in self._metadata]
            embeddings = self.model.encode(all_texts, normalize_embeddings=True)
            new_index = faiss.IndexFlatL2(384)
            new_index.add(np.array(embeddings, dtype=np.float32))
            self._index = new_index
        else:
            self._index = faiss.IndexFlatL2(384)

        self._save()
        return deleted_count

    def clear(self):
        """Clear all entries."""
        self._metadata = []
        self._index = faiss.IndexFlatL2(384)
        self._save()
