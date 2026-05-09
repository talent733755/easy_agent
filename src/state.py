import operator
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.tools import BaseTool


def _merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dicts, with b's values taking precedence."""
    return {**a, **b}


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tools: list[BaseTool]
    context_window: dict           # {"used_tokens": N, "max_tokens": N, "threshold": 0.7}
    memory_context: str            # 从记忆系统检索到的上下文
    user_profile: str              # USER.md 内容快照
    agent_notes: str               # MEMORY.md 内容快照
    pending_human_input: bool      # 是否等待人类输入
    iteration_count: int           # 当前循环计数
    max_iterations: int            # 最大循环轮数
    nudge_counter: int             # 自我进化触发计数
    provider_name: str             # 当前使用的 LLM 提供商

    # Phase 1: Beauty Agent 新增字段
    intent: str                    # 识别意图: query_customer | knowledge_query | mixed
    customer_context: dict         # 当前客户信息缓存
    knowledge_results: list        # RAG 检索结果
    mcp_results: Annotated[dict, _merge_dicts]  # MCP 调用结果（支持并行合并）