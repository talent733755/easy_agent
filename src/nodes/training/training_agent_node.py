"""Training agent node: unified LLM node for all training phases."""

from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel

from src.state import AgentState


def _load_prompt(name: str) -> str:
    """Load a prompt template from the prompts/training directory."""
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / "training" / f"{name}.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


def _get_last_user_message(state: AgentState) -> str:
    """获取最后一条用户消息。"""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.content.strip()
    return ""


def _build_welcome_system_prompt() -> str:
    """构建欢迎阶段的系统提示。"""
    return (
        "你是「智能对练」系统的引导助手。你的唯一职责是输出以下欢迎语，"
        "不要添加任何额外内容。如果用户输入不是「开始陪练」相关，礼貌地引导他们输入「开始陪练」。\n\n"
        + _load_prompt("welcome")
    )


def _build_roleplay_system_prompt(training_context: dict) -> str:
    """构建角色扮演阶段的系统提示。"""
    template = _load_prompt("roleplay")

    scenario = training_context.get("scenario", {})
    customer_name = training_context.get("customer_name", "模拟客户")
    customer_info = training_context.get("customer_info", "")
    key_info = training_context.get("key_info", [])
    knowledge_refs = training_context.get("knowledge_refs", "")

    # 从客户信息中推断性格
    personality_hints = scenario.get("personality_hints", ["普通客户"])
    personality = "、".join(personality_hints)

    # 从客户信息中提取会员等级和年龄
    member_level = "普通会员"
    age = "未知"
    for line in customer_info.split("\n"):
        if "会员" in line and "等级" in line:
            member_level = line.split("：")[-1].strip() if "：" in line else line.split(":")[-1].strip()
        if "年龄" in line:
            age = line.split("：")[-1].strip() if "：" in line else line.split(":")[-1].strip()

    key_info_str = "\n".join(f"- {info}" for info in key_info) if key_info else customer_info[:500]

    return template.format(
        customer_name=customer_name,
        member_level=member_level,
        age=age,
        personality=personality,
        scenario_description=scenario.get("description", "自由对话"),
        key_info=key_info_str,
        knowledge_refs=knowledge_refs or "(无相关知识库内容)",
    )


def create_training_agent_node(model: BaseChatModel):
    """创建训练 Agent 节点。

    根据 training_phase 切换不同的 system prompt：
    - welcome: 输出引导语
    - roleplay: AI 扮演客户
    - evaluate: 此阶段由 evaluate_node 处理
    """
    def training_agent(state: AgentState) -> dict:
        phase = state.get("training_phase", "welcome")
        training_context = state.get("training_context", {})

        if phase == "welcome":
            system_content = _build_welcome_system_prompt()
        elif phase == "roleplay":
            system_content = _build_roleplay_system_prompt(training_context)
        else:
            # evaluate 或其他阶段不经过此节点
            return {}

        system_msg = SystemMessage(content=system_content)
        messages = [system_msg] + list(state.get("messages", []))

        response = model.invoke(messages)

        # 过滤 thinking block，只保留文本内容
        content = response.content
        if isinstance(content, list):
            text_parts = [
                block["text"] for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = "\n".join(text_parts) if text_parts else str(content)
            response = AIMessage(content=content)

        # 更新轮次计数
        if phase == "roleplay":
            rounds = training_context.get("rounds", 0) + 1
            training_context["rounds"] = rounds

        return {
            "messages": [response],
            "training_context": training_context,
        }

    return training_agent
