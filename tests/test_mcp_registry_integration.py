"""MCP 服务注册机制集成测试。"""
import pytest
from unittest.mock import MagicMock, patch
from src.config import MCPServerConfig
from src.nodes.beauty.mcp_service_node import create_mcp_service_node


class TestMCPRegistryIntegration:
    def test_multiple_mcp_configs(self):
        """验证多个 MCP 服务配置可以同时创建节点。"""
        configs = [
            {
                "url": "http://localhost:3001",
                "timeout": 30,
                "intent": "query_customer",
                "endpoints": [{"name": "get_customer", "description": "查客户"}],
                "name": "customer",
            },
            {
                "url": "http://localhost:3002",
                "timeout": 30,
                "intent": "query_performance",
                "endpoints": [{"name": "get_daily_performance", "description": "查当日业绩"}],
                "name": "performance",
            },
        ]
        for cfg in configs:
            node = create_mcp_service_node(cfg)
            assert callable(node)

    def test_config_dataclass_with_new_fields(self):
        """验证 MCPServerConfig 新字段正常工作。"""
        config = MCPServerConfig(
            url="http://localhost:3001",
            timeout=30,
            intent="query_customer",
            endpoints=[
                {"name": "get_customer", "description": "查客户"},
                {"name": "get_consumption", "description": "查消费"},
            ],
        )
        assert config.intent == "query_customer"
        assert len(config.endpoints) == 2

    def test_empty_mcp_config_no_crash(self):
        """验证空 MCP 配置不崩溃。"""
        config = {
            "url": "",
            "timeout": 30,
            "intent": "query_test",
            "endpoints": [],
            "name": "empty",
        }
        node = create_mcp_service_node(config)
        from langchain_core.messages import HumanMessage
        state = {
            "messages": [HumanMessage(content="test")],
            "intent": "query_test",
            "customer_context": {},
            "mcp_results": {},
        }
        result = node(state)
        assert result == {}

    def test_graph_dynamic_registration(self):
        """验证 graph 动态注册 MCP 节点。"""
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="test")

        with patch("src.graph._build_agent_model", return_value=mock_model):
            from src.graph import build_graph
            graph = build_graph(checkpointer=None)

        # Inspect builder nodes via compiled graph internals
        # The compiled graph stores nodes in its internal graph
        node_names = []
        if hasattr(graph, "builder"):
            node_names = list(graph.builder.nodes.keys())

        # Config has customer MCP → should register mcp_customer
        mcp_nodes = [n for n in node_names if n.startswith("mcp_")]
        assert len(mcp_nodes) > 0, f"No MCP nodes registered: {node_names}"

    def test_intent_routes_from_config(self):
        """验证意图路由从 config 动态构建。"""
        from src.graph import _build_mcp_intent_routes
        mock_configs = {
            "customer": MagicMock(intent="query_customer"),
            "performance": MagicMock(intent="query_performance"),
        }
        routes = _build_mcp_intent_routes(mock_configs)
        assert routes["query_customer"] == "mcp_customer"
        assert routes["query_performance"] == "mcp_performance"
