"""Training router: detects keywords and routes between training phases."""

import re
from src.state import AgentState


# 结束触发关键词
END_TRIGGERS = ["结束对练", "结束训练", "评价", "打分", "评分", "总结"]

# 开始触发关键词
START_TRIGGERS = ["开始陪练", "开始对练", "开始训练", "陪练"]

# 重新开始触发
RESTART_TRIGGERS = ["再来一轮", "再练一次", "重新开始", "继续对练"]


def _get_last_user_message(state: AgentState) -> str:
    """获取最后一条用户消息。"""
    from langchain_core.messages import HumanMessage
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.content.strip()
    return ""


def training_router(state: AgentState) -> str:
    """训练路由器：根据当前阶段和用户输入决定下一步。

    Returns:
        "setup"      → 场景/客户设置
        "roleplay"   → 角色扮演对话
        "evaluate"   → 评分
        "wait"       → 等待输入（welcome阶段或设置中间态）
    """
    phase = state.get("training_phase", "welcome")
    user_msg = _get_last_user_message(state)

    # welcome 阶段：等待 "开始陪练"
    if phase == "welcome":
        for trigger in START_TRIGGERS:
            if trigger in user_msg:
                return "setup"
        return "wait"

    # setup 阶段：可能需要多轮输入（选场景 → 选客户）
    if phase == "setup":
        training_context = state.get("training_context", {})
        # 场景已选择但客户未指定 → 继续 setup
        if training_context.get("scenario") and not training_context.get("customer_name"):
            return "setup"
        # 场景和客户都有了 → 进入 roleplay
        if training_context.get("scenario") and training_context.get("customer_name"):
            return "roleplay"
        # 场景未选择 → setup
        return "setup"

    # roleplay 阶段：检查结束关键词
    if phase == "roleplay":
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
        return "wait"

    return "wait"
