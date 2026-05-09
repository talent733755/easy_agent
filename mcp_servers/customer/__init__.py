"""Customer MCP Server - Mock customer data service."""

from .tools import get_customer, get_consumption, get_service_history
from .router import router

__all__ = [
    "get_customer",
    "get_consumption",
    "get_service_history",
    "router",
]