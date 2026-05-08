"""端到端集成测试：模拟真实对话，验证记忆学习行为。"""
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from src.memory.memory_entry import MemoryEntry, parse_entries
from src.nodes.trigger_detector import detect_trigger
from src.nodes.memory_learn_node import _extract_messages, _merge_entries, memory_learn_node
import src.config
import src.providers.factory


def _mock_heavy_modules():
    """Prevent native segfaults from torch/sentence_transformers by mocking heavy modules."""
    for mod_name in ["src.memory.vector_store", "src.memory.fts5_store"]:
        sys.modules[mod_name] = MagicMock()


class TestCorrectionDetection:
    """验收标准1: 纠错触发。"""

    def test_keyword_correction(self):
        state = {"messages": [HumanMessage(content="不对，嫩小白是补水的不是美白的")]}
        assert detect_trigger(state) == "correction"

    def test_keyword_instruction(self):
        state = {"messages": [HumanMessage(content="记住，以后回答要简洁")]}
        assert detect_trigger(state) == "instruction"

    def test_no_trigger_normal_conversation(self):
        state = {"messages": [HumanMessage(content="今天有什么活动")]}
        assert detect_trigger(state) is None


class TestEndToEndLearning:
    """验收标准2-4: 完整学习流程测试。"""

    def test_correction_writes_memory(self):
        """验收标准2: 用户纠错 → 自动写入 MEMORY.md。"""
        _mock_heavy_modules()
        with patch.object(src.config, "load_config") as mock_lc, \
             patch.object(src.providers.factory, "get_provider") as mock_gp:

            mock_model = MagicMock()
            mock_model.invoke.return_value = MagicMock(
                content="[场景] 产品推荐\n[错误] 混淆功效\n[正确] 嫩小白补水保湿"
            )
            mock_gp.return_value = MagicMock(get_model=MagicMock(return_value=mock_model))
            mock_lc.return_value = MagicMock(
                active_provider="mimo", providers={"mimo": MagicMock()}
            )

            state = {
                "messages": [
                    AIMessage(content="嫩小白美白淡斑效果很好"),
                    HumanMessage(content="不对，嫩小白是补水保湿的，美白是焕颜精华"),
                ],
                "provider_name": "mimo",
            }

            tmpdir = tempfile.mkdtemp()
            try:
                result = memory_learn_node(state, data_dir=tmpdir)
                memory_file = Path(tmpdir) / "MEMORY.md"
                assert memory_file.exists()
                content = memory_file.read_text()
                assert "嫩小白补水保湿" in content
            finally:
                shutil.rmtree(tmpdir)

    def test_instruction_writes_memory(self):
        """验收标准3: 显式指令 → 自动写入 MEMORY.md。"""
        _mock_heavy_modules()
        with patch.object(src.config, "load_config") as mock_lc, \
             patch.object(src.providers.factory, "get_provider") as mock_gp:

            mock_model = MagicMock()
            mock_model.invoke.return_value = MagicMock(content="[指令] 回答要简洁")
            mock_gp.return_value = MagicMock(get_model=MagicMock(return_value=mock_model))
            mock_lc.return_value = MagicMock(
                active_provider="mimo", providers={"mimo": MagicMock()}
            )

            state = {
                "messages": [HumanMessage(content="记住，以后回答要简洁，不要啰嗦")],
                "provider_name": "mimo",
            }

            tmpdir = tempfile.mkdtemp()
            try:
                result = memory_learn_node(state, data_dir=tmpdir)
                memory_file = Path(tmpdir) / "MEMORY.md"
                assert memory_file.exists()
                content = memory_file.read_text()
                assert "回答要简洁" in content
            finally:
                shutil.rmtree(tmpdir)

    def test_no_trigger_skips(self):
        """验收标准5: 普通对话不触发。"""
        state = {"messages": [HumanMessage(content="今天有什么活动")]}
        assert detect_trigger(state) is None


class TestRepeatedCorrectionMerge:
    """验收标准4: 重复纠错合并。"""

    def test_merge_increments_and_upgrades(self):
        existing = MemoryEntry(
            category="correction",
            scene="产品推荐",
            error="混淆功效",
            correct="嫩小白补水",
            correction_count=1,
            importance="medium",
        )
        new = MemoryEntry(
            category="correction",
            scene="产品推荐",
            correct="嫩小白补水保湿，焕颜精华美白淡斑",
        )
        merged = _merge_entries(existing, new)
        assert merged.correction_count == 2
        assert merged.importance == "high"


class TestSpecAcceptanceCriteria:
    """验收标准逐一验证。"""

    def test_correction_keyword_detection(self):
        """验收标准1: 纠错关键词触发。"""
        state = {"messages": [HumanMessage(content="不对，嫩小白是补水的不是美白的")]}
        assert detect_trigger(state) == "correction"

    def test_instruction_keyword_detection(self):
        """验收标准2: 指令关键词触发。"""
        state = {"messages": [HumanMessage(content="记住，以后回答要简洁")]}
        assert detect_trigger(state) == "instruction"

    def test_normal_no_trigger(self):
        """验收标准3: 普通对话不触发。"""
        state = {"messages": [HumanMessage(content="今天有什么活动")]}
        assert detect_trigger(state) is None

    def test_extraction_message_limit(self):
        """消息提取限制。"""
        state = {"messages": [HumanMessage(content=f"msg{i}") for i in range(20)]}
        result = _extract_messages(state, max_messages=5)
        assert len(result) == 5
