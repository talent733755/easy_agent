"""Test intent classification node."""
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage
from src.nodes.beauty.intent_node import intent_classify_node, _rule_based_classify
from src.state import AgentState


def create_test_state(message: str) -> AgentState:
    """Create a test state with a single message."""
    return {
        "messages": [HumanMessage(content=message)],
        "tools": [],
        "context_window": {"used_tokens": 0, "max_tokens": 128000, "threshold": 0.7},
        "memory_context": "",
        "user_profile": "",
        "agent_notes": "",
        "pending_human_input": False,
        "iteration_count": 0,
        "max_iterations": 15,
        "nudge_counter": 0,
        "provider_name": "openai",
        "intent": "",
        "customer_context": {},
        "knowledge_results": [],
        "mcp_results": {},
    }


def test_intent_classify_query_customer():
    """Should classify customer query intent using rule-based fallback."""
    # Test rule-based classification directly (no API needed)
    result = _rule_based_classify("帮我查一下张女士的档案")

    assert result["intent"] == "query_customer"
    assert "张女士" in result.get("customer_name", "")


def test_intent_classify_knowledge_query():
    """Should classify knowledge query intent."""
    result = _rule_based_classify("敏感肌适合什么产品")

    assert result["intent"] == "knowledge_query"


def test_intent_classify_mixed():
    """Should classify mixed intent."""
    result = _rule_based_classify("张女士适合做什么项目")

    assert result["intent"] == "mixed"
    assert "张女士" in result.get("customer_name", "")


def test_intent_classify_general():
    """Should classify general conversation intent."""
    result = _rule_based_classify("你好")

    assert result["intent"] == "general"


def test_intent_node_with_mocked_llm():
    """Test intent_node with mocked LLM response."""
    state = create_test_state("帮我查一下张女士的档案")

    # Mock the LLM response
    mock_response = MagicMock()
    mock_response.content = '{"intent": "query_customer", "customer_name": "张女士", "query_topic": ""}'

    with patch("src.nodes.beauty.intent_node.ChatOpenAI") as MockChatOpenAI:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm

        result = intent_classify_node(state)

        assert result["intent"] == "query_customer"
        assert result["customer_name"] == "张女士"


def test_intent_node_empty_message():
    """Should return general intent for empty messages."""
    state: AgentState = {
        "messages": [],
        "tools": [],
        "context_window": {"used_tokens": 0, "max_tokens": 128000, "threshold": 0.7},
        "memory_context": "",
        "user_profile": "",
        "agent_notes": "",
        "pending_human_input": False,
        "iteration_count": 0,
        "max_iterations": 15,
        "nudge_counter": 0,
        "provider_name": "openai",
        "intent": "",
        "customer_context": {},
        "knowledge_results": [],
        "mcp_results": {},
    }

    result = intent_classify_node(state)

    assert result["intent"] == "general"
