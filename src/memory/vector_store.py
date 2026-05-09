import pickle
from pathlib import Path
import numpy as np
import faiss
import openai


SILICONFLOW_BASE = "https://api.siliconflow.cn/v1"
SILICONFLOW_KEY = "sk-pzjxmvkbjgscxidtcuqrgaxngfktlfctlhfwjpmwyfkdztgy"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024  # BAAI/bge-m3 输出维度


def _embed(texts: list[str]) -> np.ndarray:
    """调用硅基流动 API 获取 embeddings。"""
    client = openai.OpenAI(base_url=SILICONFLOW_BASE, api_key=SILICONFLOW_KEY)
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = [item.embedding for item in resp.data]
    return np.array(vectors, dtype=np.float32)


class VectorStore:
    def __init__(self, data_dir: str, max_entries: int = 500):
        self.dir = Path(data_dir).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.index_path = self.dir / "faiss.index"
        self.meta_path = self.dir / "meta.pkl"
        self._index = None
        self._metadata: list[dict] = []
        self._load()

    def _load(self):
        if self.index_path.exists() and self.meta_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "rb") as f:
                self._metadata = pickle.load(f)
        else:
            self._index = faiss.IndexFlatL2(EMBEDDING_DIM)
            self._metadata = []

    def _save(self):
        faiss.write_index(self._index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self._metadata, f)

    def add(self, text: str, metadata: dict = None):
        embedding = _embed([text])
        self._index.add(embedding)
        meta = metadata.copy() if metadata else {}
        meta["_text"] = text
        self._metadata.append(meta)
        # FIFO eviction
        if len(self._metadata) > self.max_entries:
            self._metadata = self._metadata[-self.max_entries:]
            all_texts = [m.get("_text", "") for m in self._metadata]
            embeddings = _embed(all_texts)
            new_index = faiss.IndexFlatL2(EMBEDDING_DIM)
            new_index.add(embeddings)
            self._index = new_index
        self._save()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if len(self._metadata) == 0:
            return []
        embedding = _embed([query])
        distances, indices = self._index.search(embedding, min(top_k, len(self._metadata)))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self._metadata):
                results.append({"distance": float(dist), "metadata": self._metadata[idx]})
        return results

    def count(self) -> int:
        return len(self._metadata)

    def delete(self, filter_fn) -> int:
        """Delete entries where filter_fn(metadata) returns True."""
        if len(self._metadata) == 0:
            return 0

        indices_to_keep = [i for i, meta in enumerate(self._metadata) if not filter_fn(meta)]
        deleted_count = len(self._metadata) - len(indices_to_keep)

        if deleted_count == 0:
            return 0

        self._metadata = [self._metadata[i] for i in indices_to_keep]

        if self._metadata:
            all_texts = [m.get("_text", "") for m in self._metadata]
            embeddings = _embed(all_texts)
            new_index = faiss.IndexFlatL2(EMBEDDING_DIM)
            new_index.add(embeddings)
            self._index = new_index
        else:
            self._index = faiss.IndexFlatL2(EMBEDDING_DIM)

        self._save()
        return deleted_count

    def clear(self):
        """Clear all entries."""
        self._metadata = []
        self._index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self._save()
