import pytest
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from src.nodes.memory_learn_node import (
    _extract_messages,
    _parse_extracted_fields,
    _merge_entries,
    _check_dedup,
)
from src.memory.memory_entry import MemoryEntry


class TestExtractMessages:
    def test_extracts_user_and_ai(self):
        state = {
            "messages": [
                HumanMessage(content="你好"),
                AIMessage(content="你好！有什么可以帮忙的"),
                HumanMessage(content="不对，嫩小白是补水的"),
            ]
        }
        result = _extract_messages(state, max_messages=4)
        assert len(result) == 3
        assert result[0]["role"] == "human"
        assert result[1]["role"] == "ai"
        assert result[2]["role"] == "human"

    def test_truncates_long_content(self):
        state = {"messages": [HumanMessage(content="x" * 1000)]}
        result = _extract_messages(state)
        assert len(result[0]["content"]) == 300

    def test_limits_messages(self):
        state = {"messages": [HumanMessage(content=f"msg{i}") for i in range(20)]}
        result = _extract_messages(state, max_messages=5)
        assert len(result) == 5

    def test_skips_empty_content(self):
        state = {"messages": [HumanMessage(content="")] }
        result = _extract_messages(state)
        assert len(result) == 0


class TestParseExtractedFields:
    def test_correction_fields(self):
        extracted = """[场景] 产品推荐
[错误] 混淆功效
[正确] 嫩小白补水"""
        fields = _parse_extracted_fields("correction", extracted)
        assert fields["scene"] == "产品推荐"
        assert fields["error"] == "混淆功效"
        assert fields["correct"] == "嫩小白补水"

    def test_instruction_fields(self):
        extracted = "[指令] 回答要简洁"
        fields = _parse_extracted_fields("instruction", extracted)
        assert fields["instruction"] == "回答要简洁"

    def test_partial_fields(self):
        extracted = "[场景] 产品推荐\n[正确] 嫩小白补水"
        fields = _parse_extracted_fields("correction", extracted)
        assert "scene" in fields
        assert "correct" in fields
        assert "error" not in fields


class TestMergeEntries:
    def test_merge_increments_count(self):
        existing = MemoryEntry(
            category="correction", scene="场景", correction_count=1, importance="medium"
        )
        new = MemoryEntry(category="correction", scene="场景", correct="更新")
        merged = _merge_entries(existing, new)
        assert merged.correction_count == 2

    def test_merge_upgrades_importance_at_2(self):
        existing = MemoryEntry(
            category="correction", scene="场景", correction_count=1, importance="medium"
        )
        new = MemoryEntry(category="correction", scene="场景")
        merged = _merge_entries(existing, new)
        assert merged.importance == "high"

    def test_merge_preserves_existing_correct(self):
        existing = MemoryEntry(
            category="correction", scene="场景", correct="旧正确", correction_count=1
        )
        new = MemoryEntry(category="correction", scene="场景", correct="")
        merged = _merge_entries(existing, new)
        assert merged.correct == "旧正确"

    def test_merge_replaces_with_new_correct(self):
        existing = MemoryEntry(
            category="correction", scene="场景", correct="旧正确", correction_count=1
        )
        new = MemoryEntry(category="correction", scene="场景", correct="新正确")
        merged = _merge_entries(existing, new)
        assert merged.correct == "新正确"


class TestCheckDedup:
    def test_no_existing_entries(self):
        entry = MemoryEntry(category="correction", scene="场景")
        result = _check_dedup(entry, [], MagicMock())
        assert result is None

    def test_different_category(self):
        existing = [MemoryEntry(category="instruction", instruction="指令")]
        entry = MemoryEntry(category="correction", scene="场景")
        result = _check_dedup(entry, existing, MagicMock())
        assert result is None

    def test_llm_returns_merge(self):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="MERGE 0")
        existing = [MemoryEntry(category="correction", scene="场景A")]
        entry = MemoryEntry(category="correction", scene="场景A类似")
        result = _check_dedup(entry, existing, mock_model)
        assert result == 0

    def test_llm_returns_new(self):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="NEW")
        existing = [MemoryEntry(category="correction", scene="场景A")]
        entry = MemoryEntry(category="correction", scene="完全不同")
        result = _check_dedup(entry, existing, mock_model)
        assert result is None

    def test_llm_exception(self):
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("error")
        existing = [MemoryEntry(category="correction", scene="场景")]
        entry = MemoryEntry(category="correction", scene="新场景")
        result = _check_dedup(entry, existing, mock_model)
        assert result is None
