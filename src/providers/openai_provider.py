from langchain_openai import ChatOpenAI
from src.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI and OpenAI-compatible APIs."""

    def get_model(self, model_name: str = None) -> ChatOpenAI:
        """Get a ChatOpenAI instance.

        Args:
            model_name: Optional model name override.

        Returns:
            ChatOpenAI instance configured with the provider settings.
        """
        return ChatOpenAI(
            model=model_name or self.config["default_model"],
            base_url=self.config.get("base_url"),
            api_key=self.config["api_key"],
        )
