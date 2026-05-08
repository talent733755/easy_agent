import pytest
from src.memory.memory_entry import MemoryEntry, serialize_entries, parse_entries


class TestMemoryEntrySerialization:
    def test_serialize_correction(self):
        entry = MemoryEntry(
            category="correction",
            scene="产品推荐混淆嫩小白和焕颜精华",
            error="美白归于嫩小白",
            correct="嫩小白补水，焕颜精华美白",
            source="2026-05-08 用户纠错",
            importance="high",
            correction_count=2,
        )
        result = serialize_entries([entry])
        assert "## 错误经验" in result
        assert "[场景] 产品推荐混淆嫩小白和焕颜精华" in result
        assert "[错误] 美白归于嫩小白" in result
        assert "[纠错次数] 2" in result

    def test_serialize_instruction(self):
        entry = MemoryEntry(
            category="instruction",
            instruction="回答要简洁",
            source="2026-05-08 用户指令",
            importance="medium",
        )
        result = serialize_entries([entry])
        assert "## 用户指令" in result
        assert "[指令] 回答要简洁" in result

    def test_serialize_empty(self):
        result = serialize_entries([])
        assert "last_updated" in result

    def test_serialize_mixed(self):
        entries = [
            MemoryEntry(category="correction", scene="场景A", correct="对A", source="来源A"),
            MemoryEntry(category="instruction", instruction="指令B", source="来源B"),
        ]
        result = serialize_entries(entries)
        assert "## 错误经验" in result
        assert "## 用户指令" in result
        assert "场景A" in result
        assert "指令B" in result


class TestMemoryEntryParsing:
    def test_parse_correction(self):
        content = """---
last_updated: 2026-05-08
---

## 错误经验

- [场景] 产品推荐混淆
  [错误] 美白归于嫩小白
  [正确] 嫩小白补水保湿
  [来源] 2026-05-08 用户纠错
  [重要度] high
  [纠错次数] 2
"""
        entries = parse_entries(content)
        assert len(entries) == 1
        assert entries[0].category == "correction"
        assert entries[0].scene == "产品推荐混淆"
        assert entries[0].correction_count == 2
        assert entries[0].importance == "high"

    def test_parse_multiple_entries(self):
        content = """---
last_updated: 2026-05-08
---

## 错误经验

- [场景] 场景A
  [错误] 错A
  [正确] 对A
  [来源] 2026-05-08
  [重要度] low
  [纠错次数] 1

- [场景] 场景B
  [错误] 错B
  [正确] 对B
  [来源] 2026-05-08
  [重要度] high
  [纠错次数] 2
"""
        entries = parse_entries(content)
        assert len(entries) == 2
        assert entries[1].correction_count == 2

    def test_parse_instructions(self):
        content = """---
last_updated: 2026-05-08
---

## 用户指令

- [指令] 回答要简洁
  [来源] 2026-05-08 用户指令
  [重要度] high
"""
        entries = parse_entries(content)
        assert len(entries) == 1
        assert entries[0].category == "instruction"
        assert entries[0].instruction == "回答要简洁"

    def test_parse_empty(self):
        entries = parse_entries("")
        assert entries == []

    def test_roundtrip(self):
        original = MemoryEntry(
            category="correction",
            scene="测试场景",
            error="错误做法",
            correct="正确做法",
            source="2026-05-08 用户纠错",
            importance="high",
            correction_count=3,
        )
        serialized = serialize_entries([original])
        parsed = parse_entries(serialized)
        assert len(parsed) == 1
        assert parsed[0].scene == original.scene
        assert parsed[0].correction_count == original.correction_count

    def test_roundtrip_instruction(self):
        original = MemoryEntry(
            category="instruction",
            instruction="保持简洁回答",
            source="2026-05-08 用户指令",
            importance="high",
        )
        serialized = serialize_entries([original])
        parsed = parse_entries(serialized)
        assert len(parsed) == 1
        assert parsed[0].instruction == original.instruction
