import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from src.nodes.trigger_detector import keyword_check, detect_trigger


class TestKeywordCheck:
    def test_correction_detected(self):
        assert keyword_check("不对，嫩小白是补水的") == "correction"

    def test_instruction_detected(self):
        assert keyword_check("记住，以后回答简洁") == "instruction"

    def test_no_trigger(self):
        assert keyword_check("你好，请问最近有什么活动") is None

    def test_correction_priority_over_instruction(self):
        assert keyword_check("记住，你错了") == "correction"

    def test_various_correction_keywords(self):
        assert keyword_check("你错了") == "correction"
        assert keyword_check("应该这样做") == "correction"
        assert keyword_check("正确的是焕颜精华") == "correction"

    def test_various_instruction_keywords(self):
        assert keyword_check("记下来") == "instruction"
        assert keyword_check("以后别这样回答") == "instruction"
        assert keyword_check("以后要记得查知识库") == "instruction"

    def test_partial_match(self):
        assert keyword_check("我觉得不对啊") == "correction"
        assert keyword_check("你记住以后要简洁") == "instruction"


class TestDetectTrigger:
    def test_keyword_triggers_fast(self):
        state = {"messages": [HumanMessage(content="不对，应该用焕颜精华")]}
        result = detect_trigger(state)
        assert result == "correction"

    def test_llm_fallback_correction(self):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="CORRECTION")
        state = {
            "messages": [
                HumanMessage(content="我更喜欢简洁的回答"),
                AIMessage(content="好的，我会详细说明每个产品功效..."),
                HumanMessage(content="你这个太长了，我刚才说了要简洁"),
            ]
        }
        result = detect_trigger(state, model=mock_model)
        assert result == "correction"

    def test_llm_fallback_instruction(self):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="INSTRUCTION")
        state = {"messages": [HumanMessage(content="我更喜欢用表格展示")]}
        result = detect_trigger(state, model=mock_model)
        assert result == "instruction"

    def test_llm_fallback_skip(self):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="SKIP")
        state = {"messages": [HumanMessage(content="今天天气真好")]}
        result = detect_trigger(state, model=mock_model)
        assert result is None

    def test_llm_fallback_lower_case(self):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="correction")
        state = {"messages": [HumanMessage(content="随便聊天")]}
        result = detect_trigger(state, model=mock_model)
        assert result == "correction"

    def test_llm_exception_returns_none(self):
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("API error")
        state = {"messages": [HumanMessage(content="随便说说")]}
        result = detect_trigger(state, model=mock_model)
        assert result is None

    def test_empty_messages(self):
        state = {"messages": []}
        assert detect_trigger(state) is None

    def test_no_model_returns_none(self):
        state = {"messages": [HumanMessage(content="普通消息没有关键词")]}
        assert detect_trigger(state) is None
