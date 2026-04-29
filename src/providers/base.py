from abc import ABC, abstractmethod
from langchain_core.language_models import BaseChatModel


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def get_model(self, model_name: str = None) -> BaseChatModel:
        """Get a chat model instance.

        Args:
            model_name: Optional model name override. Uses default from config if not provided.

        Returns:
            A LangChain chat model instance.
        """
        ...
