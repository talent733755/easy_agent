"""MCP Gateway integration tests."""
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


class TestGatewayIntegration:
    def test_all_services_mounted(self):
        """验证 gateway 自动挂载了所有 MCP 服务。"""
        from mcp_servers.gateway import app

        routes = [r.path for r in app.routes if hasattr(r, "methods")]
        customer_routes = [r for r in routes if r.startswith("/customer")]
        assert len(customer_routes) > 0, f"No customer routes: {routes}"
        user_routes = [r for r in routes if r.startswith("/user_profile")]
        assert len(user_routes) > 0, f"No user_profile routes: {routes}"

    def test_customer_endpoint_via_gateway(self):
        """通过 gateway 调用 customer 端点。"""
        from mcp_servers.gateway import app

        client = TestClient(app)
        resp = client.post(
            "/customer/tools/get_customer",
            json={"customer_name": "zhang"},
        )
        assert resp.status_code == 200
        assert resp.json()["customer_id"] == "C001"

    @patch("mcp_servers.user_profile.tools.httpx")
    def test_user_profile_endpoint_via_gateway(self, mock_httpx):
        """通过 gateway 调用 user_profile 端点。"""
        mock_resp = MagicMock()
        mock_resp.text = "用户档案数据"
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_httpx.get.return_value = mock_resp

        from mcp_servers.gateway import app

        client = TestClient(app)
        resp = client.post(
            "/user_profile/tools/get_user_info",
            json={"user_id": "231326"},
        )
        assert resp.status_code == 200
        assert "用户档案" in resp.json()["result"]

    def test_graph_uses_gateway_url(self):
        """验证 graph 中 MCP 节点使用 gateway URL。"""
        from src.config import load_config

        config = load_config()
        assert config.beauty.mcp_gateway.url == "http://localhost:3001"
        # 验证 graph 构建正常
        from src.graph import build_graph

        graph = build_graph(checkpointer=None)
        assert graph is not None
