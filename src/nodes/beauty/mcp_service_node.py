"""通用 MCP 服务节点工厂：根据 config 创建标准 MCP 调用节点。"""

import json
import httpx
from langchain_core.messages import HumanMessage
from src.state import AgentState


ENDPOINT_SELECT_PROMPT = """根据用户消息，选择最合适的端点调用。

可用端点：
{endpoint_list}

用户消息：{user_message}

只输出端点名称，不要多余文字。"""


def _extract_last_user_message(state: AgentState) -> str:
    messages = state.get("messages", [])
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def _select_endpoint(user_message: str, endpoints: list[dict], model) -> str:
    """用 LLM 选择最合适的端点。"""
    if not endpoints:
        return ""
    if len(endpoints) == 1:
        return endpoints[0]["name"]

    endpoint_list = "\n".join(
        f"- {ep['name']}: {ep.get('description', '')}" for ep in endpoints
    )
    prompt = ENDPOINT_SELECT_PROMPT.format(
        endpoint_list=endpoint_list,
        user_message=user_message[:300],
    )
    try:
        result = model.invoke(prompt)
        content = str(result.content).strip()
        for ep in endpoints:
            if ep["name"] in content:
                return ep["name"]
        return endpoints[0]["name"]
    except Exception:
        return endpoints[0]["name"]


EXTRACT_PARAMS_PROMPT = """从用户消息中提取调用 {endpoint_name} 所需的参数。

端点描述：{endpoint_description}
用户消息：{user_message}

以 JSON 格式输出参数，例如：{{"city": "广州", "date": "2026-05-09"}}
如果没有明确参数，输出空 JSON：{{}}"""


def _extract_params(user_message: str, endpoint_name: str, endpoint_desc: str, model) -> dict:
    """用 LLM 从用户消息中提取调用参数。"""
    prompt = EXTRACT_PARAMS_PROMPT.format(
        endpoint_name=endpoint_name,
        endpoint_description=endpoint_desc,
        user_message=user_message[:300],
    )
    try:
        result = model.invoke(prompt)
        content = str(result.content).strip()
        if "{" in content and "}" in content:
            json_str = content[content.index("{"):content.rindex("}") + 1]
            return json.loads(json_str)
        return {}
    except Exception:
        return {}


def create_mcp_service_node(mcp_config: dict, model=None):
    """根据 config 声明创建通用 MCP 调用节点。

    Args:
        mcp_config: dict with keys: url, timeout, intent, endpoints, name
        model: LLM 模型实例（可选，用于端点选择和参数提取）

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

        # 3. 选择端点
        if model:
            endpoint_name = _select_endpoint(user_message, endpoints, model)
        else:
            endpoint_name = endpoints[0]["name"] if endpoints else ""

        # 4. 提取参数
        endpoint_desc = ""
        for ep in endpoints:
            if ep["name"] == endpoint_name:
                endpoint_desc = ep.get("description", "")
                break
        if model:
            params = _extract_params(user_message, endpoint_name, endpoint_desc, model)
        else:
            params = {}

        # 5. 调用 MCP 端点
        if not url or not endpoint_name:
            return {}

        try:
            response = httpx.post(
                f"{url}/tools/{endpoint_name}",
                json=params,
                timeout=timeout,
            )
            if response.status_code >= 400:
                data = {"error": f"HTTP {response.status_code}: {response.text[:200]}"}
            else:
                data = response.json()
        except Exception:
            data = {"error": f"调用 {service_name} 服务失败"}

        # 6. 返回结果（数据统一放 mcp_results，避免并行写 customer_context 冲突）
        return {
            "mcp_results": {service_name: data},
        }

    return mcp_node
