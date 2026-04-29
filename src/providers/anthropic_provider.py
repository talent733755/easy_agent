from langchain_anthropic import ChatAnthropic
from src.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude APIs."""

    def get_model(self, model_name: str = None) -> ChatAnthropic:
        """Get a ChatAnthropic instance.

        Args:
            model_name: Optional model name override.

        Returns:
            ChatAnthropic instance configured with the provider settings.
        """
        return ChatAnthropic(
            model=model_name or self.config["default_model"],
            api_key=self.config["api_key"],
        )
