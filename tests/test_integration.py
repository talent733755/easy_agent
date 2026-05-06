"""Integration tests for Beauty Agent end-to-end flow.

Tests the complete flow from query to response including:
- Intent classification
- Knowledge retrieval
- MCP customer lookup
- Agent response generation
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from src.state import AgentState
from src.graph import build_graph
from src.config import load_config


@pytest.fixture
def beauty_config():
    """Load beauty config for testing."""
    config = load_config()
    return config


@pytest.fixture
def initial_state() -> AgentState:
    """Create initial agent state for testing."""
    return {
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
        "provider_name": "codingplan",
        # Beauty fields
        "intent": "general",
        "customer_name": "",
        "customer_context": {},
        "knowledge_results": [],
        "mcp_results": {},
    }


class TestIntentClassificationIntegration:
    """Integration tests for intent classification."""

    def test_classify_customer_query(self, initial_state):
        """Test classification of customer archive query."""
        from src.nodes.beauty.intent_node import _rule_based_classify

        # Use rule-based classification directly (no API key needed)
        result = _rule_based_classify("查一下张女士的档案")

        # Verify intent classification
        assert result["intent"] == "query_customer"
        assert result["customer_name"] == "张女士"

    def test_classify_knowledge_query(self, initial_state):
        """Test classification of knowledge query."""
        from src.nodes.beauty.intent_node import _rule_based_classify

        result = _rule_based_classify("推荐一个适合油性皮肤的护肤产品")

        assert result["intent"] == "knowledge_query"

    def test_classify_mixed_query(self, initial_state):
        """Test classification of mixed query."""
        from src.nodes.beauty.intent_node import _rule_based_classify

        result = _rule_based_classify("张女士上次买了什么产品，推荐适合她的护理流程")

        assert result["intent"] == "mixed"
        assert result["customer_name"] == "张女士"

    def test_classify_general_query(self, initial_state):
        """Test classification of general query."""
        from src.nodes.beauty.intent_node import _rule_based_classify

        result = _rule_based_classify("今天天气怎么样")

        assert result["intent"] == "general"

    def test_intent_node_with_mocked_llm(self, initial_state):
        """Test intent node with mocked LLM."""
        from src.nodes.beauty.intent_node import intent_classify_node

        state = initial_state.copy()
        state["messages"] = [HumanMessage(content="查一下张女士的档案")]

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


class TestMCPClientIntegration:
    """Integration tests for MCP client node."""

    def test_mcp_customer_lookup_success(self, initial_state):
        """Test successful customer lookup via MCP."""
        from src.nodes.beauty.mcp_client_node import mcp_customer_node

        state = initial_state.copy()
        state["intent"] = "query_customer"
        state["customer_name"] = "张女士"

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "张女士",
            "phone": "138****1234",
            "membership": "金卡会员",
            "last_visit": "2024-12-15",
            "total_spent": 28500,
        }

        with patch("httpx.post", return_value=mock_response):
            result = mcp_customer_node(state, mcp_url="http://localhost:3001")

        assert result["customer_context"]["name"] == "张女士"
        assert result["customer_context"]["membership"] == "金卡会员"
        assert "customer" in result["mcp_results"]

    def test_mcp_customer_lookup_not_found(self, initial_state):
        """Test customer lookup when customer not found."""
        from src.nodes.beauty.mcp_client_node import mcp_customer_node

        state = initial_state.copy()
        state["intent"] = "query_customer"
        state["customer_name"] = "不存在的客户"

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.post", return_value=mock_response):
            result = mcp_customer_node(state, mcp_url="http://localhost:3001")

        assert result["customer_context"] == {}
        assert result["mcp_results"] == {"customer": {}}

    def test_mcp_skip_for_knowledge_query(self, initial_state):
        """Test that MCP is skipped for knowledge queries."""
        from src.nodes.beauty.mcp_client_node import mcp_customer_node

        state = initial_state.copy()
        state["intent"] = "knowledge_query"

        result = mcp_customer_node(state, mcp_url="http://localhost:3001")

        # Should return empty without making HTTP call
        assert result["customer_context"] == {}
        assert result["mcp_results"] == {}


class TestKnowledgeRetrievalIntegration:
    """Integration tests for knowledge retrieval node."""

    def test_knowledge_retrieve_with_valid_dir(self, initial_state, tmp_path):
        """Test knowledge retrieval from valid directory."""
        from src.nodes.beauty.knowledge_node import knowledge_retrieve_node

        # Create test knowledge files
        kb_dir = tmp_path / "knowledge"
        kb_dir.mkdir()
        products_dir = kb_dir / "products"
        products_dir.mkdir()
        (products_dir / "skincare.txt").write_text("油性皮肤护理建议：使用控油洁面乳...")

        state = initial_state.copy()
        state["messages"] = [HumanMessage(content="推荐适合油性皮肤的产品")]

        result = knowledge_retrieve_node(state, knowledge_dir=str(kb_dir))

        assert "knowledge_results" in result

    def test_knowledge_retrieve_empty_dir(self, initial_state, tmp_path):
        """Test knowledge retrieval from empty directory."""
        from src.nodes.beauty.knowledge_node import knowledge_retrieve_node

        kb_dir = tmp_path / "empty_knowledge"
        kb_dir.mkdir()

        state = initial_state.copy()
        state["messages"] = [HumanMessage(content="推荐产品")]

        result = knowledge_retrieve_node(state, knowledge_dir=str(kb_dir))

        assert result["knowledge_results"] == []


class TestGraphEndToEnd:
    """End-to-end integration tests for the complete graph."""

    def test_graph_builds_with_beauty_config(self, beauty_config):
        """Test that graph builds successfully with beauty config."""
        graph = build_graph(checkpointer=MemorySaver())
        assert graph is not None

        # Verify beauty nodes are present
        node_names = list(graph.nodes.keys())
        assert "intent_classify" in node_names
        assert "knowledge_retrieve" in node_names
        assert "mcp_customer" in node_names

    @patch("httpx.post")
    @patch("src.nodes.beauty.intent_node.ChatOpenAI")
    def test_customer_query_flow(self, mock_chat_openai, mock_post, initial_state, beauty_config):
        """Test complete customer query flow: intent -> MCP -> agent."""
        # Mock LLM for intent classification
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"intent": "query_customer", "customer_name": "张女士", "query_topic": ""}'
        )
        mock_chat_openai.return_value = mock_llm

        # Mock MCP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "张女士",
            "phone": "138****1234",
            "membership": "金卡会员",
            "last_visit": "2024-12-15",
            "total_spent": 28500,
            "services": ["面部护理", "精油按摩"],
        }
        mock_post.return_value = mock_response

        # Mock the LLM to return a simple response
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_model.invoke.return_value = AIMessage(
            content="张女士是金卡会员，累计消费28500元，最近一次到店是2024年12月15日，"
                    "常做面部护理和精油按摩项目。"
        )

        with patch("src.graph._build_agent_model", return_value=mock_model):
            graph = build_graph(checkpointer=MemorySaver())

        # Set up state with customer query
        state = initial_state.copy()
        state["messages"] = [HumanMessage(content="查一下张女士的档案")]

        config = {"configurable": {"thread_id": "test-customer-query"}}

        # Invoke graph
        result = graph.invoke(state, config)

        # Verify flow completed
        assert len(result["messages"]) > 1

        # Check intent was classified
        assert result.get("intent") in ("query_customer", "general")

        # Check response was generated
        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)

    @patch("src.nodes.beauty.intent_node.ChatOpenAI")
    def test_knowledge_query_flow(self, mock_chat_openai, initial_state, beauty_config, tmp_path):
        """Test knowledge query flow: intent -> knowledge -> agent."""
        # Mock LLM for intent classification
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"intent": "knowledge_query", "customer_name": "", "query_topic": "护肤"}'
        )
        mock_chat_openai.return_value = mock_llm

        # Create test knowledge base
        kb_dir = tmp_path / "knowledge"
        kb_dir.mkdir()
        products_dir = kb_dir / "products"
        products_dir.mkdir()
        (products_dir / "skincare.txt").write_text(
            "油性皮肤护理：\n"
            "1. 使用控油洁面乳\n"
            "2. 定期做深层清洁\n"
            "3. 使用清爽型保湿产品"
        )

        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_model.invoke.return_value = AIMessage(
            content="对于油性皮肤，建议：\n"
                    "1. 使用控油洁面乳\n"
                    "2. 定期做深层清洁护理\n"
                    "3. 选择清爽型保湿产品"
        )

        # Mock knowledge_retrieve_node to return test data
        mock_knowledge_result = {
            "knowledge_results": [
                {"content": "油性皮肤护理建议...", "source": "skincare.txt"}
            ]
        }

        with patch("src.graph._build_agent_model", return_value=mock_model), \
             patch("src.nodes.beauty.knowledge_retrieve_node", return_value=mock_knowledge_result):
            graph = build_graph(checkpointer=MemorySaver())

        state = initial_state.copy()
        state["messages"] = [HumanMessage(content="推荐适合油性皮肤的护理流程")]

        config = {"configurable": {"thread_id": "test-knowledge-query"}}
        result = graph.invoke(state, config)

        assert len(result["messages"]) > 1

    @patch("src.nodes.beauty.intent_node.ChatOpenAI")
    def test_general_query_flow(self, mock_chat_openai, initial_state, beauty_config):
        """Test general query that doesn't require beauty nodes."""
        # Mock LLM for intent classification
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"intent": "general", "customer_name": "", "query_topic": ""}'
        )
        mock_chat_openai.return_value = mock_llm

        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_model.invoke.return_value = AIMessage(
            content="你好！我是美容顾问助手，有什么可以帮您的吗？"
        )

        with patch("src.graph._build_agent_model", return_value=mock_model):
            graph = build_graph(checkpointer=MemorySaver())

        state = initial_state.copy()
        state["messages"] = [HumanMessage(content="你好")]

        config = {"configurable": {"thread_id": "test-general-query"}}
        result = graph.invoke(state, config)

        assert len(result["messages"]) > 1
        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)


class TestWebIntegration:
    """Integration tests for web interface."""

    def test_health_endpoint(self):
        """Test health check endpoint."""
        from fastapi.testclient import TestClient
        from web.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"

    def test_index_page(self):
        """Test index page loads."""
        from fastapi.testclient import TestClient
        from web.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/")
        assert response.status_code == 200
        assert "Easy Agent" in response.text

    def test_websocket_connection(self):
        """Test WebSocket connection and session creation."""
        from fastapi.testclient import TestClient
        from web.app import create_app

        app = create_app()
        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Should receive session message
            data = websocket.receive_json()
            assert data["type"] == "session"
            assert "session_id" in data
            assert "provider" in data


class TestMCPServerStartup:
    """Tests for MCP server configuration."""

    def test_mcp_config_loaded(self, beauty_config):
        """Test that MCP config is properly loaded."""
        assert beauty_config.beauty is not None
        assert "customer" in beauty_config.beauty.mcp_servers

        customer_mcp = beauty_config.beauty.mcp_servers["customer"]
        assert customer_mcp.url == "http://localhost:3001"
        assert customer_mcp.timeout == 30

    def test_knowledge_base_config_loaded(self, beauty_config):
        """Test that knowledge base config is properly loaded."""
        assert beauty_config.beauty is not None
        assert beauty_config.beauty.knowledge_base.base_dir == "./knowledge/jiaolifu"

        indexes = beauty_config.beauty.knowledge_base.indexes
        assert "products" in indexes
        assert "procedures" in indexes
        assert "scripts" in indexes
        assert "troubleshooting" in indexes


class TestErrorHandling:
    """Tests for error handling in the integration."""

    def test_mcp_timeout_handling(self, initial_state):
        """Test that MCP timeout is handled gracefully."""
        from src.nodes.beauty.mcp_client_node import mcp_customer_node
        import httpx

        state = initial_state.copy()
        state["intent"] = "query_customer"
        state["customer_name"] = "张女士"

        # Mock timeout
        with patch("httpx.post", side_effect=httpx.TimeoutException("Timeout")):
            result = mcp_customer_node(state, mcp_url="http://localhost:3001", timeout=5)

        # Should return empty context instead of raising
        assert result["customer_context"] == {}
        assert result["mcp_results"] == {"customer": {}}

    def test_mcp_connection_error_handling(self, initial_state):
        """Test that MCP connection errors are handled gracefully."""
        from src.nodes.beauty.mcp_client_node import mcp_customer_node
        import httpx

        state = initial_state.copy()
        state["intent"] = "query_customer"
        state["customer_name"] = "张女士"

        # Mock connection error
        with patch("httpx.post", side_effect=httpx.ConnectError("Connection refused")):
            result = mcp_customer_node(state, mcp_url="http://localhost:3001")

        # Should return empty context instead of raising
        assert result["customer_context"] == {}
        assert result["mcp_results"] == {"customer": {}}
