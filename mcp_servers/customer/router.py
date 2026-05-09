"""Customer MCP Service - APIRouter for gateway."""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from mcp_servers.customer.tools import get_customer, get_consumption, get_service_history

router = APIRouter()


class GetCustomerRequest(BaseModel):
    customer_name: str


class GetConsumptionRequest(BaseModel):
    customer_id: str


class GetServiceHistoryRequest(BaseModel):
    customer_id: str


@router.post("/tools/get_customer")
async def get_customer_endpoint(request: GetCustomerRequest) -> dict[str, Any]:
    result = get_customer(request.customer_name)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["error"])
    return result["data"]


@router.post("/tools/get_consumption")
async def get_consumption_endpoint(request: GetConsumptionRequest) -> dict[str, Any]:
    result = get_consumption(request.customer_id)
    return result


@router.post("/tools/get_service_history")
async def get_service_history_endpoint(request: GetServiceHistoryRequest) -> dict[str, Any]:
    result = get_service_history(request.customer_id)
    return result
