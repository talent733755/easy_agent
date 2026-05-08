from dataclasses import dataclass
from datetime import datetime


@dataclass
class MemoryEntry:
    category: str = "correction"     # "correction" | "instruction"
    scene: str = ""                  # 场景描述
    error: str = ""                  # 错误做法（仅 correction 类型）
    correct: str = ""                # 正确做法
    instruction: str = ""            # 指令内容（仅 instruction 类型）
    source: str = ""                 # 来源: "2026-05-08 用户纠错"
    importance: str = "medium"       # "low" | "medium" | "high"
    correction_count: int = 1        # 纠错次数
    created: str = ""                # 创建日期 YYYY-MM-DD


def serialize_entries(entries: list[MemoryEntry]) -> str:
    """将条目列表序列化为 MEMORY.md 格式。"""
    now = datetime.now().strftime("%Y-%m-%d")
    lines = [f"---", f"last_updated: {now}", f"---\n"]

    corrections = [e for e in entries if e.category == "correction"]
    instructions = [e for e in entries if e.category == "instruction"]

    if corrections:
        lines.append("## 错误经验\n")
        for e in corrections:
            lines.append(f"- [场景] {e.scene}")
            if e.error:
                lines.append(f"  [错误] {e.error}")
            lines.append(f"  [正确] {e.correct}")
            lines.append(f"  [来源] {e.source}")
            lines.append(f"  [重要度] {e.importance}")
            lines.append(f"  [纠错次数] {e.correction_count}")
            lines.append("")

    if instructions:
        lines.append("---\n")
        lines.append("## 用户指令\n")
        for e in instructions:
            lines.append(f"- [指令] {e.instruction}")
            lines.append(f"  [来源] {e.source}")
            lines.append(f"  [重要度] {e.importance}")
            lines.append("")

    return "\n".join(lines)


def parse_entries(content: str) -> list[MemoryEntry]:
    """从 MEMORY.md 文本中解析出条目列表。"""
    entries = []
    current_category = "correction"
    current: dict = {}

    for line in content.splitlines():
        stripped = line.strip()

        if stripped.startswith("## 错误经验"):
            current_category = "correction"
            continue
        elif stripped.startswith("## 用户指令"):
            current_category = "instruction"
            continue
        elif stripped.startswith("---") or stripped == "" or stripped.startswith("last_updated"):
            if current and (current.get("scene") or current.get("instruction")):
                entries.append(MemoryEntry(category=current_category, **current))
                current = {}
            continue

        if stripped.startswith("- [场景]"):
            if current and (current.get("scene") or current.get("instruction")):
                entries.append(MemoryEntry(category=current_category, **current))
            current = {"scene": stripped.replace("- [场景]", "").strip()}
        elif stripped.startswith("[错误]"):
            current["error"] = stripped.replace("[错误]", "").strip()
        elif stripped.startswith("[正确]"):
            current["correct"] = stripped.replace("[正确]", "").strip()
        elif stripped.startswith("[来源]"):
            current["source"] = stripped.replace("[来源]", "").strip()
        elif stripped.startswith("[重要度]"):
            current["importance"] = stripped.replace("[重要度]", "").strip()
        elif stripped.startswith("[纠错次数]"):
            try:
                current["correction_count"] = int(stripped.replace("[纠错次数]", "").strip())
            except ValueError:
                current["correction_count"] = 1
        elif stripped.startswith("- [指令]"):
            if current and (current.get("scene") or current.get("instruction")):
                entries.append(MemoryEntry(category=current_category, **current))
            current = {"instruction": stripped.replace("- [指令]", "").strip()}

    # Flush last entry
    if current and (current.get("scene") or current.get("instruction")):
        entries.append(MemoryEntry(category=current_category, **current))

    return entries
