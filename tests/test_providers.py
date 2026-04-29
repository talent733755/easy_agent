import pytest
from src.providers.factory import get_provider
from src.providers.base import BaseProvider


class TestBaseProvider:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseProvider()


def test_factory_returns_openai_provider():
    config = {
        "type": "openai",
        "api_key": "test-key",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
    }
    provider = get_provider("openai", config)
    assert provider is not None
    model = provider.get_model("gpt-4o")
    assert model.model_name == "gpt-4o"


def test_factory_returns_anthropic_provider():
    config = {
        "type": "anthropic",
        "api_key": "test-key",
        "default_model": "claude-sonnet-4-6",
    }
    provider = get_provider("anthropic", config)
    assert provider is not None
    model = provider.get_model("claude-sonnet-4-6")
    # ChatAnthropic uses 'model' attribute, not 'model_name'
    assert model.model == "claude-sonnet-4-6"


def test_factory_returns_zhipu_provider():
    config = {
        "type": "openai_compatible",
        "api_key": "test-key",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4-plus",
    }
    provider = get_provider("zhipu", config)
    assert provider is not None
    model = provider.get_model("glm-4-plus")
    assert model.model_name == "glm-4-plus"


def test_factory_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("unknown", {"type": "unknown"})
