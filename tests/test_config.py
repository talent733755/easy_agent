import os
import tempfile
from src.config import load_config, AppConfig


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
    assert config.beauty.mcp_servers["customer"].url == "http://localhost:3001"
    assert config.beauty.web.port == 8080
