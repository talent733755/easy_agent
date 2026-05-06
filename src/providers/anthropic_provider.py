import os
import httpx
from langchain_anthropic import ChatAnthropic
from anthropic import Anthropic
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
        # Support custom base_url for Anthropic-compatible APIs
        base_url = self.config.get("base_url")
        if base_url:
            chat = ChatAnthropic(
                model=model_name or self.config["default_model"],
                api_key=self.config["api_key"],
                timeout=self.config.get("timeout", 60),
                anthropic_api_url=base_url,
            )
            # Override internal client to handle non-standard API paths/auth
            chat._client = self._build_custom_client(base_url)
            return chat
        return ChatAnthropic(
            model=model_name or self.config["default_model"],
            api_key=self.config["api_key"],
            timeout=self.config.get("timeout", 60),
        )

    def _build_custom_client(self, base_url: str) -> Anthropic:
        """Build an Anthropic client with URL rewriting for non-standard APIs.

        Some Anthropic-compatible APIs use different API versions (e.g., /v2/messages
        instead of /v1/messages). Since the Anthropic SDK always appends /v1/messages,
        we rewrite the URL to match the API's expected path.
        """
        # Remove interfering ANTHROPIC_* env vars (e.g. ANTHROPIC_AUTH_TOKEN
        # set by Claude Code itself would override our api_key)
        for k in list(os.environ.keys()):
            if k.startswith("ANTHROPIC_") and k != "ANTHROPIC_API_KEY":
                os.environ.pop(k, None)

        class _RewriteClient(httpx.Client):
            def send(self, request, **kwargs):
                url = str(request.url)
                # Replace duplicate version path when base_url already has version
                import re
                # e.g. /v2/v1/messages -> /v2/messages
                new_url = re.sub(r"/(v\d+)/v1/", r"/\1/", url)
                if new_url != url:
                    request.url = httpx.URL(new_url)
                return super().send(request, **kwargs)

        return Anthropic(
            api_key=self.config["api_key"],
            base_url=base_url,
            http_client=_RewriteClient(),
        )
