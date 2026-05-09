"""Test user_profile MCP service."""
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from mcp_servers.user_profile.tools import get_user_info
from mcp_servers.user_profile.router import router


class TestUserInfoTools:
    @patch("mcp_servers.user_profile.tools.httpx")
    def test_get_user_info_success(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.text = "用户张三，会员等级金卡，最近消费2024-01-15"
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_httpx.get.return_value = mock_resp

        result = get_user_info(user_id="231326")
        assert result["success"] is True
        assert "张三" in result["result"]

    def test_get_user_info_no_params(self):
        result = get_user_info()
        assert result["success"] is False

    @patch("mcp_servers.user_profile.tools.httpx")
    def test_get_user_info_http_error(self, mock_httpx):
        mock_httpx.get.side_effect = Exception("Connection refused")
        result = get_user_info(user_id="123")
        assert result["success"] is False
        assert "error" in result


class TestUserInfoRouter:
    @patch("mcp_servers.user_profile.tools.httpx")
    def test_endpoint(self, mock_httpx):
        mock_resp = MagicMock()
        mock_resp.text = "用户信息"
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
        assert "result" in resp.json()
