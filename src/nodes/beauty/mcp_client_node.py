"""MCP client node for Customer MCP integration."""
import httpx

from src.state import AgentState


def mcp_customer_node(
    state: AgentState,
    mcp_url: str,
    timeout: int = 30,
) -> dict:
    """Call Customer MCP to retrieve customer information.

    This node:
    1. Checks intent is "query_customer" or "mixed"
    2. Gets customer_name from state
    3. Calls Customer MCP via HTTP POST to {mcp_url}/tools/get_customer
    4. Returns customer data as {"customer_context": data, "mcp_results": {"customer": data}}
    5. Handles errors gracefully

    Args:
        state: Current agent state containing intent and customer_name
        mcp_url: URL of the Customer MCP server
        timeout: Request timeout in seconds (default: 30)

    Returns:
        dict with customer_context and mcp_results keys
    """
    # Step 1: Check intent - only process query_customer or mixed
    intent = state.get("intent", "")
    if intent not in ("query_customer", "mixed"):
        return {"customer_context": {}, "mcp_results": {}}

    # Step 2: Get customer_name from state
    customer_name = state.get("customer_name", "")
    if not customer_name:
        return {"customer_context": {}, "mcp_results": {}}

    # Step 3: Call Customer MCP via HTTP POST
    endpoint = f"{mcp_url}/tools/get_customer"
    payload = {"customer_name": customer_name}

    try:
        response = httpx.post(
            endpoint,
            json=payload,
            timeout=timeout,
        )

        # Step 4: Check response status
        if response.status_code != 200:
            return {
                "customer_context": {},
                "mcp_results": {"customer": {}},
            }

        # Parse customer data
        customer_data = response.json()

        # Step 5: Return customer data
        return {
            "customer_context": customer_data,
            "mcp_results": {"customer": customer_data},
        }

    except (httpx.HTTPError, httpx.TimeoutException):
        # Handle errors gracefully - return empty context
        return {
            "customer_context": {},
            "mcp_results": {"customer": {}},
        }