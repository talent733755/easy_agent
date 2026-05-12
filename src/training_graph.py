"""独立的智能对练 Graph，不影响现有 beauty graph。"""

from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from src.state import AgentState
from src.config import load_config
from src.providers.factory import get_provider


def _build_training_model():
    """Build the LLM model for training agent."""
    config = load_config()
    provider_name = config.active_provider
    provider_config = config.providers.get(provider_name)
    if provider_config is None:
        raise ValueError(f"Provider '{provider_name}' not found in config")
    provider = get_provider(provider_name, provider_config)
    return provider.get_model()


# 结束触发关键词
END_TRIGGERS = ["结束对练", "结束训练", "评价", "打分", "评分", "总结"]
START_TRIGGERS = ["开始陪练", "开始对练", "开始训练", "陪练"]
RESTART_TRIGGERS = ["再来一轮", "再练一次", "重新开始", "继续对练"]


def _get_last_user_message(state: AgentState) -> str:
    from langchain_core.messages import HumanMessage
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.content.strip()
    return ""


def _route_training(state: AgentState) -> str:
    """训练流程路由器：根据当前阶段和用户输入决定下一步。

    Returns:
        "welcome"    → 首次进入，输出引导语
        "setup"      → 场景/客户设置
        "roleplay"   → 角色扮演对话
        "evaluate"   → 评分
    """
    phase = state.get("training_phase", "")

    # 首次进入（无阶段）
    if not phase:
        return "welcome"

    user_msg = _get_last_user_message(state)

    # welcome 阶段：等待 "开始陪练"
    if phase == "welcome":
        for trigger in START_TRIGGERS:
            if trigger in user_msg:
                return "setup"
        # 还没开始，重新输出欢迎提示
        return "welcome"

    # setup 阶段：可能需要多轮输入
    if phase == "setup":
        training_context = state.get("training_context", {})
        # 场景和客户都有了 → 进入 roleplay
        if training_context.get("scenario") and training_context.get("customer_name"):
            return "roleplay"
        # 还在设置中
        return "setup"

    # roleplay 阶段：检查结束关键词
    if phase == "roleplay":
        # 先检查是否想重新开始（优先级高于普通对话）
        for trigger in START_TRIGGERS + RESTART_TRIGGERS:
            if trigger in user_msg:
                return "evaluate"
        for trigger in END_TRIGGERS:
            if trigger in user_msg:
                return "evaluate"
        return "roleplay"

    # evaluate 阶段：检查是否重新开始
    if phase == "evaluate":
        for trigger in RESTART_TRIGGERS:
            if trigger in user_msg:
                return "setup"
        for trigger in START_TRIGGERS:
            if trigger in user_msg:
                return "setup"
        # 还没重新开始，输出提示
        return "welcome"

    return "welcome"


def build_training_graph(checkpointer: BaseCheckpointSaver = None) -> StateGraph:
    """构建智能对练 Graph。

    每次 WebSocket 收到用户消息 → 一次 graph.invoke → 输出回复 → END
    下次消息 → 再次 invoke，通过 checkpointer 恢复状态。

    流程：
        dispatch → 路由
            ├── "welcome"   → welcome_node → END
            ├── "setup"     → setup_node → END
            ├── "roleplay"  → roleplay_node → END
            └── "evaluate"  → evaluate_node → END
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    model = _build_training_model()

    from src.nodes.training.training_agent_node import create_training_agent_node
    from src.nodes.training.setup_node import create_setup_node
    from src.nodes.training.evaluate_node import evaluate_training_node

    agent_node_fn = create_training_agent_node(model)
    setup_node_fn = create_setup_node(model)

    builder = StateGraph(AgentState)

    # --- Nodes ---
    def welcome_node(state: AgentState) -> dict:
        from langchain_core.messages import AIMessage

        # 如果已有消息且不是首次，说明用户在 evaluate 后想重新开始
        messages = state.get("messages", [])
        if messages and state.get("training_phase") == "evaluate":
            # 重置训练状态
            prompt_path = Path(__file__).parent / "prompts" / "training" / "welcome.txt"
            welcome_text = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else "欢迎使用智能对练！"
            return {
                "messages": [AIMessage(content=welcome_text)],
                "training_phase": "welcome",
                "training_context": {},
                "training_score": {},
                "training_history": [],
            }
        elif not messages or not state.get("training_phase"):
            # 首次进入
            prompt_path = Path(__file__).parent / "prompts" / "training" / "welcome.txt"
            welcome_text = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else "欢迎使用智能对练！"
            return {
                "messages": [AIMessage(content=welcome_text)],
                "training_phase": "welcome",
                "training_context": {},
                "training_score": {},
                "training_history": [],
            }
        else:
            # 用户在 welcome 阶段说了无关内容
            from langchain_core.messages import AIMessage
            return {
                "messages": [AIMessage(content="请输入 **「开始陪练」** 来开始训练。")],
            }

    def setup_wrapper(state: AgentState) -> dict:
        return setup_node_fn(state)

    def roleplay_wrapper(state: AgentState) -> dict:
        return agent_node_fn(state)

    def evaluate_wrapper(state: AgentState) -> dict:
        return evaluate_training_node(state, model)

    builder.add_node("welcome", welcome_node)
    builder.add_node("setup", setup_wrapper)
    builder.add_node("roleplay", roleplay_wrapper)
    builder.add_node("evaluate", evaluate_wrapper)

    # --- Edges ---
    # welcome → END
    builder.add_edge("welcome", END)

    # setup → END
    builder.add_edge("setup", END)

    # roleplay → END
    builder.add_edge("roleplay", END)

    # evaluate → END
    builder.add_edge("evaluate", END)

    # 使用入口路由：第一次 invoke 直接进 welcome
    # 后续 invoke 通过 _route_training 路由
    def entry_router(state: AgentState) -> str:
        phase = state.get("training_phase", "")
        if not phase:
            return "welcome"
        return _route_training(state)

    # 重新设计：使用 conditional entry
    builder.set_entry_point("dispatch")

    def dispatch_node(state: AgentState) -> dict:
        return {}

    builder.add_node("dispatch", dispatch_node)

    builder.add_conditional_edges(
        "dispatch",
        entry_router,
        {
            "welcome": "welcome",
            "setup": "setup",
            "roleplay": "roleplay",
            "evaluate": "evaluate",
        },
    )

    return builder.compile(checkpointer=checkpointer)
