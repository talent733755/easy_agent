"""触发检测：关键词快速匹配 + LLM 兜底判断。"""

CORRECTION_KEYWORDS = ["不对", "错了", "不是这样", "应该", "正确的是", "实际上是", "错的"]
INSTRUCTION_KEYWORDS = ["记住", "记下来", "以后要", "以后别", "别再", "以后要记得", "记住这个"]


def keyword_check(user_message: str) -> str | None:
    """关键词快速匹配。

    Returns:
        "correction" | "instruction" | None
    """
    for kw in CORRECTION_KEYWORDS:
        if kw in user_message:
            return "correction"
    for kw in INSTRUCTION_KEYWORDS:
        if kw in user_message:
            return "instruction"
    return None


LLM_JUDGE_PROMPT = """判断以下用户消息是否包含值得记忆的内容。

你需要判断的是：
1. CORRECTION：用户在纠正 Agent 之前的错误做法（即使没有用"不对""错了"等词）
2. INSTRUCTION：用户在给出一个明确的要求或偏好，要求 Agent 以后记住或照做
3. SKIP：用户只是普通对话，没有纠错也没有显式指令

用户消息：{last_user_msg}
最近 Agent 回复：{last_agent_msg}

只输出一个词：CORRECTION 或 INSTRUCTION 或 SKIP"""


def llm_judge(user_message: str, agent_reply: str, model) -> str:
    """调用 LLM 判断是否触发学习。

    Args:
        user_message: 用户最后一条消息
        agent_reply: Agent 最近一次回复
        model: BaseChatModel 实例

    Returns:
        "correction" | "instruction" | "skip"
    """
    prompt = LLM_JUDGE_PROMPT.format(
        last_user_msg=user_message[:500],
        last_agent_msg=agent_reply[:500],
    )
    try:
        result = model.invoke(prompt)
        content = str(result.content).strip().upper()
        if "CORRECTION" in content:
            return "correction"
        elif "INSTRUCTION" in content:
            return "instruction"
        else:
            return "skip"
    except Exception:
        return "skip"


def detect_trigger(state: dict, model=None) -> str | None:
    """两层触发检测：关键词优先，LLM 兜底。

    Args:
        state: 当前 Agent 状态 (dict with 'messages' key)
        model: LLM 模型实例（可选，用于兜底判断）

    Returns:
        "correction" | "instruction" | None
    """
    # 提取最后一条用户消息
    messages = state.get("messages", [])
    last_user_msg = ""
    for m in reversed(messages):
        if hasattr(m, "content") and not hasattr(m, "tool_call_id"):
            last_user_msg = str(m.content)
            break

    if not last_user_msg:
        return None

    # 1. 关键词快速匹配
    kw_result = keyword_check(last_user_msg)
    if kw_result:
        return kw_result

    # 2. LLM 兜底（需要 model）
    if model is None:
        return None

    # 提取 Agent 最近回复
    last_agent_reply = ""
    for m in reversed(messages):
        if hasattr(m, "content") and hasattr(m, "type") and m.type == "ai":
            last_agent_reply = str(m.content)
            break

    result = llm_judge(last_user_msg, last_agent_reply, model)
    if result == "skip":
        return None
    return result
