"""Test MCP client node for Customer MCP."""
from unittest.mock import patch, MagicMock
import httpx
from langchain_core.messages import HumanMessage

from src.state import AgentState
from src.nodes.beauty.mcp_client_node import mcp_customer_node


def create_test_state(message: str, intent: str = "query_customer", customer_name: str = "") -> AgentState:
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
        "intent": intent,
        "customer_context": {},
        "knowledge_results": [],
        "mcp_results": {},
        "customer_name": customer_name,
    }


def test_mcp_customer_node_returns_customer_data():
    """Should call Customer MCP and return customer data."""
    state = create_test_state(
        message="帮我查一下张女士的档案",
        intent="query_customer",
        customer_name="张女士",
    )

    # Mock httpx.post response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "张女士",
        "phone": "13800138000",
        "membership": "VIP",
        "last_visit": "2024-01-15",
        "total_spent": 15000,
    }

    with patch("src.nodes.beauty.mcp_client_node.httpx.post") as mock_post:
        mock_post.return_value = mock_response

        result = mcp_customer_node(state, mcp_url="http://localhost:3001")

    assert "customer_context" in result
    assert result["customer_context"]["name"] == "张女士"
    assert result["customer_context"]["membership"] == "VIP"
    assert "mcp_results" in result
    assert result["mcp_results"]["customer"]["name"] == "张女士"


def test_mcp_customer_node_handles_mixed_intent():
    """Should process customer query for mixed intent."""
    state = create_test_state(
        message="张女士适合做什么项目",
        intent="mixed",
        customer_name="张女士",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "张女士",
        "skin_type": "干性",
        "concerns": ["缺水", "细纹"],
    }

    with patch("src.nodes.beauty.mcp_client_node.httpx.post") as mock_post:
        mock_post.return_value = mock_response

        result = mcp_customer_node(state, mcp_url="http://localhost:3001")

    assert result["customer_context"]["name"] == "张女士"
    assert "skin_type" in result["customer_context"]


def test_mcp_customer_node_skips_other_intent():
    """Should skip for non-customer intents."""
    state = create_test_state(
        message="敏感肌适合什么产品",
        intent="knowledge_query",
        customer_name="",
    )

    result = mcp_customer_node(state, mcp_url="http://localhost:3001")

    assert result["customer_context"] == {}
    assert result["mcp_results"] == {}


def test_mcp_customer_node_handles_missing_customer_name():
    """Should return empty result if customer_name is missing."""
    state = create_test_state(
        message="帮我查一下客户档案",
        intent="query_customer",
        customer_name="",  # Empty customer name
    )

    result = mcp_customer_node(state, mcp_url="http://localhost:3001")

    assert result["customer_context"] == {}
    assert result["mcp_results"] == {}


def test_mcp_customer_node_handles_http_error():
    """Should handle HTTP errors gracefully."""
    state = create_test_state(
        message="帮我查一下张女士的档案",
        intent="query_customer",
        customer_name="张女士",
    )

    # Mock HTTP error
    with patch("src.nodes.beauty.mcp_client_node.httpx.post") as mock_post:
        mock_post.side_effect = httpx.HTTPError("Connection failed")

        result = mcp_customer_node(state, mcp_url="http://localhost:3001")

    # Should return empty context gracefully
    assert result["customer_context"] == {}
    assert result["mcp_results"]["customer"] == {}


def test_mcp_customer_node_handles_timeout():
    """Should handle timeout gracefully."""
    state = create_test_state(
        message="帮我查一下张女士的档案",
        intent="query_customer",
        customer_name="张女士",
    )

    # Mock timeout
    with patch("src.nodes.beauty.mcp_client_node.httpx.post") as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Request timed out")

        result = mcp_customer_node(state, mcp_url="http://localhost:3001", timeout=5)

    assert result["customer_context"] == {}
    assert result["mcp_results"]["customer"] == {}


def test_mcp_customer_node_handles_404_response():
    """Should handle 404 response gracefully."""
    state = create_test_state(
        message="帮我查一下不存在的客户",
        intent="query_customer",
        customer_name="不存在女士",
    )

    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch("src.nodes.beauty.mcp_client_node.httpx.post") as mock_post:
        mock_post.return_value = mock_response

        result = mcp_customer_node(state, mcp_url="http://localhost:3001")

    assert result["customer_context"] == {}
    assert result["mcp_results"]["customer"] == {}


def test_mcp_customer_node_handles_server_error():
    """Should handle 500 server error gracefully."""
    state = create_test_state(
        message="帮我查一下张女士的档案",
        intent="query_customer",
        customer_name="张女士",
    )

    mock_response = MagicMock()
    mock_response.status_code = 500

    with patch("src.nodes.beauty.mcp_client_node.httpx.post") as mock_post:
        mock_post.return_value = mock_response

        result = mcp_customer_node(state, mcp_url="http://localhost:3001")

    assert result["customer_context"] == {}
    assert result["mcp_results"]["customer"] == {}


def test_mcp_customer_node_uses_correct_endpoint():
    """Should call correct MCP endpoint with customer_name."""
    state = create_test_state(
        message="帮我查一下张女士的档案",
        intent="query_customer",
        customer_name="张女士",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"name": "张女士"}

    with patch("src.nodes.beauty.mcp_client_node.httpx.post") as mock_post:
        mock_post.return_value = mock_response

        mcp_customer_node(state, mcp_url="http://localhost:3001")

        # Verify correct endpoint was called
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:3001/tools/get_customer"
        assert call_args[1]["json"]["customer_name"] == "张女士"


def test_mcp_customer_node_uses_custom_timeout():
    """Should use custom timeout if provided."""
    state = create_test_state(
        message="帮我查一下张女士的档案",
        intent="query_customer",
        customer_name="张女士",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"name": "张女士"}

    with patch("src.nodes.beauty.mcp_client_node.httpx.post") as mock_post:
        mock_post.return_value = mock_response

        mcp_customer_node(state, mcp_url="http://localhost:3001", timeout=60)

        call_args = mock_post.call_args
        assert call_args[1]["timeout"] == 60