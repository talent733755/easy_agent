import re
from pathlib import Path


INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"curl\s+.*\$",
    r"<script",
    r"\{\{.*\}\}",
]


class FileMemory:
    def __init__(self, data_dir: str, memory_cap: int = 2000, user_cap: int = 1500):
        self.dir = Path(data_dir).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.dir / "MEMORY.md"
        self.user_path = self.dir / "USER.md"
        self.memory_cap = memory_cap
        self.user_cap = user_cap

    def read_memory(self) -> str:
        if self.memory_path.exists():
            return self.memory_path.read_text(encoding="utf-8")
        return ""

    def read_user(self) -> str:
        if self.user_path.exists():
            return self.user_path.read_text(encoding="utf-8")
        return ""

    def write_memory(self, content: str) -> bool:
        if self._is_malicious(content):
            return False
        truncated = content[:self.memory_cap]
        self.memory_path.write_text(truncated, encoding="utf-8")
        return True

    def write_user(self, content: str) -> bool:
        if self._is_malicious(content):
            return False
        truncated = content[:self.user_cap]
        self.user_path.write_text(truncated, encoding="utf-8")
        return True

    def append_memory(self, content: str) -> bool:
        existing = self.read_memory()
        merged = f"{existing}\n{content}".strip()
        return self.write_memory(merged)

    def append_user(self, content: str) -> bool:
        existing = self.read_user()
        merged = f"{existing}\n{content}".strip()
        return self.write_user(merged)

    def _is_malicious(self, content: str) -> bool:
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False