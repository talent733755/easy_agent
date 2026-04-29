from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.tools import BaseTool


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