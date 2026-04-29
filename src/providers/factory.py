from src.providers.openai_provider import OpenAIProvider
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import BaseProvider


def get_provider(name: str, config: dict) -> BaseProvider:
    """Factory function to get a provider instance.

    Args:
        name: Provider name (openai, anthropic, zhipu, etc.)
        config: Provider configuration dictionary.

    Returns:
        A BaseProvider instance.

    Raises:
        ValueError: If the provider type is unknown.
    """
    provider_type = config.get("type", "openai")
    if provider_type in ("openai", "openai_compatible"):
        return OpenAIProvider(config)
    elif provider_type == "anthropic":
        return AnthropicProvider(config)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
