"""通用 MCP 服务节点工厂：根据 config 创建标准 MCP 调用节点。"""

import json
import re
import httpx
from langchain_core.messages import HumanMessage
from src.state import AgentState


def _extract_last_user_message(state: AgentState) -> str:
    messages = state.get("messages", [])
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def _extract_identifiers(user_message: str) -> dict:
    """从用户消息中提取所有可用标识符。"""
    identifiers = {}

    # 手机号
    phone_match = re.search(r'1[3-9]\d{9}', user_message)
    if phone_match:
        identifiers["phone"] = phone_match.group()

    # 数字 ID（4-10位纯数字，排除手机号）
    # 先把手机号位置排除
    msg_no_phone = re.sub(r'1[3-9]\d{9}', '', user_message)
    id_match = re.search(r'(?<!\d)(\d{4,10})(?!\d)', msg_no_phone)
    if id_match:
        identifiers["user_id"] = id_match.group(1)

    # 客户姓名 (常见姓氏+女士/先生, 如 "张女士")
    name_match = re.search(
        r'([张王李赵刘陈杨黄周吴徐孙朱马胡郭林何高梁郑罗宋谢唐韩曹许邓萧冯曾程蔡彭潘于董余苏叶吕魏蒋田丁沈姜范江钟卢汪戴任廖方姚谭邹金陆孔白毛邱秦史侯孟万段雷钱汤尹黎易常武乔贺赖龚文])(女士|先生)',
        user_message,
    )
    if name_match:
        identifiers["name"] = name_match.group()

    return identifiers


def _build_endpoint_params(endpoint_desc: str, identifiers: dict) -> dict:
    """根据端点描述和可用标识符构建参数。"""
    desc_lower = endpoint_desc.lower()
    params = {}

    # 数字 ID → user_id / customer_id
    if "user_id" in identifiers:
        uid = identifiers["user_id"]
        if "user_id" in desc_lower or "用户id" in desc_lower:
            params["user_id"] = uid
        elif "customer_id" in desc_lower or "客户id" in desc_lower:
            params["customer_id"] = uid

    # 手机号 → phone / mobile
    if "phone" in identifiers:
        phone_val = identifiers["phone"]
        if "phone" in desc_lower:
            params["phone"] = phone_val
        elif "mobile" in desc_lower or "手机" in desc_lower:
            params["mobile"] = phone_val
        elif not params:  # 没有其他参数时用手机号兜底
            params["mobile"] = phone_val

    # 姓名 → customer_name
    if "name" in identifiers:
        if "customer_name" in desc_lower or "姓名" in desc_lower or "客户" in desc_lower:
            params["customer_name"] = identifiers["name"]

    return params


def _call_endpoint(url: str, endpoint_name: str, params: dict, timeout: int) -> dict:
    """调用 MCP 端点，返回结果或错误。"""
    try:
        response = httpx.post(
            f"{url}/tools/{endpoint_name}",
            json=params,
            timeout=timeout,
        )
        if response.status_code >= 400:
            return {"error": f"HTTP {response.status_code}: {response.text[:200]}"}
        return response.json()
    except Exception as e:
        return {"error": f"调用失败: {e}"}


def _is_success(data: dict) -> bool:
    """判断返回数据是否成功（非 error 且有实际内容）。"""
    if not isinstance(data, dict):
        return False
    if "error" in data:
        return False
    # 检查是否有非空结果
    if "result" in data and data["result"]:
        return True
    # 检查是否有其他有意义的字段
    return any(k not in ("success", "error") for k in data.keys())


def create_mcp_service_node(mcp_config: dict, model=None):
    """根据 config 声明创建通用 MCP 调用节点。

    策略：先提取用户消息中的标识符（手机号、姓名等），
    然后尝试所有可用端点，返回第一个成功的结果。

    Args:
        mcp_config: dict with keys: url, timeout, intent, endpoints, name
        model: LLM 模型实例（可选，未使用，保留接口兼容）

    Returns:
        标准的 LangGraph node 函数
    """
    url = mcp_config.get("url", "")
    timeout = mcp_config.get("timeout", 30)
    service_name = mcp_config.get("name", "mcp")
    intent = mcp_config.get("intent", "")
    endpoints = mcp_config.get("endpoints", [])

    def mcp_node(state: AgentState) -> dict:
        # 1. 检查意图是否匹配
        state_intent = state.get("intent", "")
        if intent and state_intent not in (intent, "mixed"):
            return {}

        # 2. 提取用户消息
        user_message = _extract_last_user_message(state)
        if not user_message:
            return {}

        # 3. 提取标识符
        identifiers = _extract_identifiers(user_message)

        if not url or not endpoints:
            return {}

        # 4. 尝试所有端点，返回第一个成功的结果
        last_error = None
        for ep in endpoints:
            ep_name = ep["name"]
            ep_desc = ep.get("description", "")
            params = _build_endpoint_params(ep_desc, identifiers)
            data = _call_endpoint(url, ep_name, params, timeout)

            if _is_success(data):
                return {"mcp_results": {service_name: data}}

            last_error = data.get("error", "unknown")

        # 所有端点都失败，返回最后一个错误
        return {
            "mcp_results": {service_name: {"error": last_error or "所有端点调用失败"}},
        }

    return mcp_node
