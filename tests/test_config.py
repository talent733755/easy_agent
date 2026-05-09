import os
import tempfile
from src.config import load_config, AppConfig, MCPServerConfig


def test_load_config_with_env_substitution():
    os.environ["TEST_KEY_123"] = "my-secret-value"
    yaml_content = """
active_provider: "openai"
providers:
  openai:
    type: openai
    api_key: "${TEST_KEY_123}"
    default_model: "gpt-4o"
agent:
  max_iterations: 10
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name

    try:
        config = load_config(tmp_path)
        assert config.active_provider == "openai"
        assert config.providers["openai"]["api_key"] == "my-secret-value"
        assert config.agent["max_iterations"] == 10
    finally:
        os.unlink(tmp_path)
        del os.environ["TEST_KEY_123"]


def test_load_config_fallback_defaults():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("active_provider: zhipu\nproviders:\n  zhipu:\n    type: openai_compatible\n    api_key: 'key'\n    default_model: 'glm-4-plus'\nagent:\n  max_iterations: 15\n")
        tmp_path = f.name

    try:
        config = load_config(tmp_path)
        assert config.agent["memory_nudge_interval"] == 10  # default
        assert config.agent["max_iterations"] == 15
    finally:
        os.unlink(tmp_path)


def test_load_beauty_config():
    """Config should load beauty section."""
    config = load_config()

    assert config.beauty is not None
    assert config.beauty.knowledge_base.base_dir == "./knowledge/jiaolifu"
    assert "products" in config.beauty.knowledge_base.indexes
    assert config.beauty.web.port == 8080

    # Verify mcp_gateway
    assert config.beauty.mcp_gateway.url == "http://localhost:3001"
    assert config.beauty.mcp_gateway.port == 3001

    # Verify customer (url is now empty, derived from gateway)
    customer = config.beauty.mcp_servers["customer"]
    assert customer.url == ""
    assert customer.intent == "query_customer"
    assert len(customer.endpoints) == 3
    assert customer.endpoints[0]["name"] == "get_customer"

    # Verify user_profile
    assert "user_profile" in config.beauty.mcp_servers
    user = config.beauty.mcp_servers["user_profile"]
    assert user.intent == "query_customer"
    assert len(user.endpoints) == 1


class TestMCPServerConfig:
    def test_init_with_endpoints(self):
        data = {
            "url": "http://localhost:3001",
            "timeout": 30,
            "intent": "query_customer",
            "endpoints": [
                {"name": "get_customer", "description": "查客户"},
            ],
        }
        config = MCPServerConfig(**data)
        assert config.intent == "query_customer"
        assert len(config.endpoints) == 1
        assert config.endpoints[0]["name"] == "get_customer"

    def test_init_defaults(self):
        config = MCPServerConfig()
        assert config.url == ""
        assert config.intent == ""
        assert config.endpoints == []


class TestMCPGatewayConfig:
    def test_gateway_defaults(self):
        from src.config import MCPGatewayConfig
        gw = MCPGatewayConfig()
        assert gw.url == "http://localhost:3001"
        assert gw.port == 3001

    def test_gateway_custom(self):
        from src.config import MCPGatewayConfig
        gw = MCPGatewayConfig(url="http://localhost:4000", port=4000)
        assert gw.url == "http://localhost:4000"
