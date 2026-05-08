"""自动记忆学习节点：检测触发 → 提取 → 去重 → 写入。"""

import re
from datetime import datetime
from src.memory.memory_entry import MemoryEntry, serialize_entries, parse_entries
from src.nodes.trigger_detector import detect_trigger


EXTRACT_PROMPT = """从以下对话中提取结构化记忆内容。

触发类型：{trigger_type}

最近对话：
{conversation}

根据触发类型，输出对应格式：

如果触发类型是 "correction"（纠错）：
[场景] <用户纠正了什么场景下的错误>
[错误] <Agent 之前的错误做法>
[正确] <用户指出的正确做法>

如果触发类型是 "instruction"（指令）：
[指令] <用户要求记住的内容，保留原意>

只输出标签和内容，不要多余文字。"""


DEDUP_PROMPT = """以下是已有的记忆条目和新的记忆条目。

请判断新条目与哪个已有条目语义最相似（相似度 > 0.8）。

已有条目：
{existing_entries}

新条目：
{new_entry}

输出规则：
- 如果有相似条目，输出：MERGE <index>（index 从 0 开始）
- 如果没有相似条目，输出：NEW

只输出一个词或"MERGE <index>"。"""


def _extract_messages(state: dict, max_messages: int = 8) -> list[dict]:
    """提取最近的对话消息。"""
    messages = state.get("messages", [])
    recent = messages[-max_messages:] if len(messages) > max_messages else messages
    result = []
    for m in recent:
        if not hasattr(m, "content") or not str(m.content):
            continue
        content = str(m.content)[:300]
        if hasattr(m, "type"):
            role = m.type
        elif hasattr(m, "role"):
            role = m.role
        else:
            role = "unknown"
        result.append({"role": role, "content": content})
    return result


def _parse_extracted_fields(trigger_type: str, extracted: str) -> dict:
    """从 LLM 提取结果中解析字段。"""
    fields = {}
    for line in extracted.splitlines():
        line = line.strip()
        if line.startswith("[场景]"):
            fields["scene"] = line.replace("[场景]", "").strip()
        elif line.startswith("[错误]"):
            fields["error"] = line.replace("[错误]", "").strip()
        elif line.startswith("[正确]"):
            fields["correct"] = line.replace("[正确]", "").strip()
        elif line.startswith("[指令]"):
            fields["instruction"] = line.replace("[指令]", "").strip()
    return fields


def _check_dedup(new_entry: MemoryEntry, existing: list[MemoryEntry], model) -> int | None:
    """与已有条目去重。返回相似条目在 full existing 列表中的 index，或 None。只与同 category 的条目比较。"""
    # Build same_cat as (original_index, entry) tuples so we can map back
    same_cat = [(idx, e) for idx, e in enumerate(existing) if e.category == new_entry.category]
    if not same_cat:
        return None

    existing_desc = []
    for filtered_i, (orig_i, e) in enumerate(same_cat):
        if e.category == "correction":
            existing_desc.append(f"[{filtered_i}] 场景:{e.scene} | 错误:{e.error} | 正确:{e.correct}")
        else:
            existing_desc.append(f"[{filtered_i}] 指令:{e.instruction}")

    if new_entry.category == "correction":
        new_desc = f"场景:{new_entry.scene} | 错误:{new_entry.error} | 正确:{new_entry.correct}"
    else:
        new_desc = f"指令:{new_entry.instruction}"

    prompt = DEDUP_PROMPT.format(
        existing_entries="\n".join(existing_desc),
        new_entry=new_desc,
    )
    try:
        result = model.invoke(prompt)
        content = str(result.content).strip()
        if "MERGE" in content:
            match = re.search(r"\d+", content)
            if match:
                filtered_idx = int(match.group())
                if 0 <= filtered_idx < len(same_cat):
                    return same_cat[filtered_idx][0]  # original index in full list
        return None
    except Exception:
        return None


def _merge_entries(existing: MemoryEntry, new_entry: MemoryEntry) -> MemoryEntry:
    """合并新旧条目。纠错次数+1，重要度升级。"""
    merged = MemoryEntry(
        category=existing.category,
        scene=new_entry.scene or existing.scene,
        error=new_entry.error or existing.error,
        correct=new_entry.correct or existing.correct,
        instruction=new_entry.instruction or existing.instruction,
        source=new_entry.source,
        importance=existing.importance,
        correction_count=existing.correction_count + 1,
        created=new_entry.created,
    )
    if merged.correction_count >= 2:
        merged.importance = "high"
    return merged


def memory_learn_node(state: dict, data_dir: str = "~/.easy_agent") -> dict:
    """自动记忆学习节点。"""
    from src.memory.file_memory import FileMemory
    from src.config import load_config
    from src.providers.factory import get_provider

    # 获取 LLM 模型
    model = None
    try:
        config = load_config()
        provider_config = config.providers.get(state.get("provider_name", config.active_provider))
        if provider_config:
            provider = get_provider(state["provider_name"], provider_config)
            model = provider.get_model()
    except Exception:
        pass

    if model is None:
        return {}

    # 1. 触发检测
    trigger_type = detect_trigger(state, model=model)
    if not trigger_type:
        return {}

    # 2. LLM 提取结构化内容
    conversation_msgs = _extract_messages(state)
    conversation = "\n".join(f"[{m['role']}]: {m['content']}" for m in conversation_msgs)

    extract_prompt = EXTRACT_PROMPT.format(
        trigger_type=trigger_type,
        conversation=conversation,
    )
    try:
        extract_result = model.invoke(extract_prompt)
        extracted = str(extract_result.content)
    except Exception:
        return {}

    fields = _parse_extracted_fields(trigger_type, extracted)

    # 3. 构建 MemoryEntry
    today = datetime.now().strftime("%Y-%m-%d")
    if trigger_type == "correction":
        if not fields.get("scene") or not fields.get("correct"):
            return {}  # 提取不完整，放弃
        new_entry = MemoryEntry(
            category="correction",
            scene=fields.get("scene", ""),
            error=fields.get("error", ""),
            correct=fields.get("correct", ""),
            source=f"{today} 用户纠错",
            importance="medium",
            correction_count=1,
            created=today,
        )
    else:  # instruction
        if not fields.get("instruction"):
            return {}
        new_entry = MemoryEntry(
            category="instruction",
            instruction=fields.get("instruction", ""),
            source=f"{today} 用户指令",
            importance="high",
            created=today,
        )

    # 4. 去重合并
    fm = FileMemory(data_dir)
    existing_content = fm.read_memory()
    existing_entries = parse_entries(existing_content)

    merge_index = _check_dedup(new_entry, existing_entries, model)
    if merge_index is not None:
        merged = _merge_entries(existing_entries[merge_index], new_entry)
        existing_entries[merge_index] = merged
    else:
        existing_entries.append(new_entry)

    # 5. 容量兜底
    if len(existing_entries) >= 20:
        existing_entries.sort(key=lambda e: (
            {"high": 2, "medium": 1, "low": 0}[e.importance],
            e.correction_count,
        ), reverse=True)
        existing_entries = existing_entries[:20]

    # 6. 写入 MEMORY.md
    content = serialize_entries(existing_entries)
    fm.write_memory(content)

    # 7. 同时写入 FTS5 + VectorStore（让普通纠错可通过检索唤醒）
    from src.memory.fts5_store import FTS5Store
    from src.memory.vector_store import VectorStore

    try:
        fts5 = FTS5Store(f"{data_dir}/history.db")
        vs = VectorStore(f"{data_dir}/vectors")
        if new_entry.category == "correction":
            text = f"【纠错经验】{new_entry.scene}: {new_entry.correct}"
        else:
            text = f"【用户指令】{new_entry.instruction}"
        fts5.insert("system", text)
        vs.add(text, {"_text": text, "role": "system", "type": "learned_memory"})
    except Exception:
        pass  # 检索层写入失败不阻塞

    return {}
