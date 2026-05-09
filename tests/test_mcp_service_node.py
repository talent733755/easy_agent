import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from src.nodes.beauty.mcp_service_node import (
    create_mcp_service_node,
    _select_endpoint,
    _extract_last_user_message,
    _extract_params,
)


class TestExtractLastUserMessage:
    def test_extracts_last_human_message(self):
        state = {"messages": [HumanMessage(content="hello"), AIMessage(content="hi")]}
        assert _extract_last_user_message(state) == "hello"

    def test_empty_messages(self):
        assert _extract_last_user_message({"messages": []}) == ""

    def test_missing_messages_key(self):
        assert _extract_last_user_message({}) == ""


class TestSelectEndpoint:
    def test_single_endpoint_returns_immediately(self):
        endpoints = [{"name": "get_customer", "description": "查客户"}]
        result = _select_endpoint("随便说说", endpoints, MagicMock())
        assert result == "get_customer"

    def test_llm_selects_correct_endpoint(self):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="get_daily_performance")
        endpoints = [
            {"name": "get_customer", "description": "查客户"},
            {"name": "get_daily_performance", "description": "查当日业绩"},
        ]
        result = _select_endpoint("广州今天的业绩", endpoints, mock_model)
        assert result == "get_daily_performance"

    def test_llm_exception_fallback(self):
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("error")
        endpoints = [
            {"name": "get_customer", "description": "查客户"},
            {"name": "get_performance", "description": "查业绩"},
        ]
        result = _select_endpoint("业绩", endpoints, mock_model)
        assert result == "get_customer"  # fallback to first

    def test_empty_endpoints(self):
        result = _select_endpoint("随便说说", [], MagicMock())
        assert result == ""


class TestExtractParams:
    def test_llm_returns_valid_json(self):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content='{"city": "广州"}')
        result = _extract_params("广州的业绩", "get_performance", "查业绩", mock_model)
        assert result == {"city": "广州"}

    def test_llm_returns_invalid_json(self):
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="no json here")
        result = _extract_params("test", "ep", "desc", mock_model)
        assert result == {}

    def test_llm_exception(self):
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("error")
        result = _extract_params("test", "ep", "desc", mock_model)
        assert result == {}


class TestCreateMCPServiceNode:
    def test_node_skips_wrong_intent(self):
        config = {
            "url": "http://localhost:3001",
            "timeout": 30,
            "intent": "query_customer",
            "endpoints": [{"name": "get_customer", "description": "查客户"}],
            "name": "customer",
        }
        node = create_mcp_service_node(config)
        state = {
            "messages": [HumanMessage(content="天气怎么样")],
            "intent": "general",
            "customer_context": {},
            "mcp_results": {},
        }
        result = node(state)
        assert result == {}

    def test_node_handles_empty_messages(self):
        config = {
            "url": "http://localhost:3001",
            "timeout": 30,
            "intent": "query_customer",
            "endpoints": [{"name": "get_customer", "description": "查客户"}],
            "name": "customer",
        }
        node = create_mcp_service_node(config)
        state = {
            "messages": [],
            "intent": "query_customer",
            "customer_context": {},
            "mcp_results": {},
        }
        result = node(state)
        assert result == {}

    @patch("src.nodes.beauty.mcp_service_node.httpx")
    def test_node_calls_endpoint(self, mock_httpx):
        config = {
            "url": "http://localhost:3001",
            "timeout": 30,
            "intent": "query_customer",
            "endpoints": [{"name": "get_customer", "description": "查客户"}],
            "name": "customer",
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"name": "张女士", "id": "C001"}
        mock_httpx.post.return_value = mock_response

        node = create_mcp_service_node(config)
        state = {
            "messages": [HumanMessage(content="查询张女士的信息")],
            "intent": "query_customer",
            "customer_context": {},
            "mcp_results": {},
        }
        result = node(state)
        assert result["mcp_results"]["customer"]["name"] == "张女士"

    @patch("src.nodes.beauty.mcp_service_node.httpx")
    def test_node_handles_http_error(self, mock_httpx):
        config = {
            "url": "http://localhost:3001",
            "timeout": 30,
            "intent": "query_customer",
            "endpoints": [{"name": "get_customer", "description": "查客户"}],
            "name": "customer",
        }
        mock_httpx.post.side_effect = Exception("Connection refused")

        node = create_mcp_service_node(config)
        state = {
            "messages": [HumanMessage(content="查询张女士的信息")],
            "intent": "query_customer",
            "customer_context": {},
            "mcp_results": {},
        }
        result = node(state)
        assert "error" in result["mcp_results"]["customer"]

    def test_node_allows_mixed_intent(self):
        config = {
            "url": "http://localhost:3001",
            "timeout": 30,
            "intent": "query_customer",
            "endpoints": [{"name": "get_customer", "description": "查客户"}],
            "name": "customer",
        }
        node = create_mcp_service_node(config)
        state = {
            "messages": [HumanMessage(content="张女士的信息和产品推荐")],
            "intent": "mixed",
            "customer_context": {},
            "mcp_results": {},
        }
        with patch("src.nodes.beauty.mcp_service_node.httpx") as mock_httpx:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"name": "张女士"}
            mock_httpx.post.return_value = mock_response
            result = node(state)
            assert "customer" in result["mcp_results"]
