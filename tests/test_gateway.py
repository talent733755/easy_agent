"""Test MCP gateway auto-discovery and mounting."""
from fastapi.testclient import TestClient
from mcp_servers.gateway import app, discover_and_mount


class TestGateway:
    def test_health_endpoint(self):
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_customer_service_mounted(self):
        client = TestClient(app)
        resp = client.post(
            "/customer/tools/get_customer",
            json={"customer_name": "zhang"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "result" in data or "success" in data or "customer_id" in data

    def test_discover_returns_service_list(self):
        from fastapi import FastAPI
        test_app = FastAPI()
        mounted = discover_and_mount(test_app)
        assert "customer" in mounted
