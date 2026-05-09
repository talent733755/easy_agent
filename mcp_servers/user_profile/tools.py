"""SAAS user profile API wrapper."""
import httpx
from httpx import HTTPStatusError

SAAS_BASE = "http://t.user.api.meididi88.com/index.php/v8.1.0/ChatUserapi"


def get_user_info(user_id: str = "", mobile: str = "") -> dict:
    """调用 SAAS 接口获取用户档案和行为记录。

    Args:
        user_id: 用户 ID
        mobile: 手机号

    Returns:
        包含 result 字段的字典
    """
    params = {}
    if user_id:
        params["user_id"] = user_id
    if mobile:
        params["mobile"] = mobile

    if not params:
        return {"success": False, "error": "需要提供 user_id 或 mobile"}

    try:
        resp = httpx.get(f"{SAAS_BASE}/getUserInfo", params=params, timeout=15)
        resp.raise_for_status()
        return {"success": True, "result": resp.text}
    except HTTPStatusError as e:
        return {"success": False, "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        return {"success": False, "error": f"请求失败: {e}"}
