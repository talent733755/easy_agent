"""Customer MCP Server - FastAPI application.

Provides endpoints for customer data retrieval:
- POST /tools/get_customer
- POST /tools/get_consumption
- POST /tools/get_service_history
- GET /health
"""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .tools import get_customer, get_consumption, get_service_history

app = FastAPI(
    title="Customer MCP Server",
    description="Mock customer data service for beauty assistant",
    version="1.0.0",
)


class GetCustomerRequest(BaseModel):
    """Request model for get_customer endpoint."""

    customer_name: str


class GetConsumptionRequest(BaseModel):
    """Request model for get_consumption endpoint."""

    customer_id: str


class GetServiceHistoryRequest(BaseModel):
    """Request model for get_service_history endpoint."""

    customer_id: str


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "customer-mcp"}


@app.post("/tools/get_customer")
async def get_customer_endpoint(request: GetCustomerRequest) -> dict[str, Any]:
    """Get customer information by name.

    Args:
        request: Request with customer_name field

    Returns:
        Customer information including:
        - customer_id
        - name
        - phone
        - membership_level
        - preferences
        - skin_type
        - allergies
    """
    result = get_customer(request.customer_name)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["error"])
    return result["data"]


@app.post("/tools/get_consumption")
async def get_consumption_endpoint(request: GetConsumptionRequest) -> dict[str, Any]:
    """Get customer consumption history.

    Args:
        request: Request with customer_id field

    Returns:
        Consumption history including:
        - record_count
        - total_amount
        - records (list of purchase records)
    """
    result = get_consumption(request.customer_id)
    return result


@app.post("/tools/get_service_history")
async def get_service_history_endpoint(request: GetServiceHistoryRequest) -> dict[str, Any]:
    """Get customer service history.

    Args:
        request: Request with customer_id field

    Returns:
        Service history including:
        - record_count
        - average_satisfaction
        - records (list of service records)
    """
    result = get_service_history(request.customer_id)
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)