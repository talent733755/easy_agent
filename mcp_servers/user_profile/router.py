"""User Profile MCP Service - APIRouter for gateway."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from mcp_servers.user_profile.tools import get_user_info

router = APIRouter()


class GetUserInfoRequest(BaseModel):
    user_id: str = ""
    mobile: str = ""


@router.post("/tools/get_user_info")
async def get_user_info_endpoint(request: GetUserInfoRequest):
    result = get_user_info(user_id=request.user_id, mobile=request.mobile)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
