"""Intent classification node for beauty agent."""
import json
import re

from langchain_core.messages import HumanMessage

from src.state import AgentState


def _build_intent_prompt(user_input: str, config=None) -> str:
    """Build intent classification prompt with dynamic MCP intent types."""
    # Base intent types
    intent_descriptions = {
        "knowledge_query": ("查询知识库", "敏感肌适合什么产品", "面部护理流程"),
        "mixed": ("混合意图（既查客户又查知识）", "张女士适合做什么项目", "推荐适合李女士的疗程"),
        "general": ("一般对话", "你好", "今天天气怎么样"),
    }

    # Add MCP service intents from config
    if config and config.beauty:
        for svc_config in config.beauty.mcp_servers.values():
            intent = getattr(svc_config, "intent", "")
            if intent and intent not in intent_descriptions:
                intent_descriptions[intent] = (f"查询{intent.replace('query_', '')}信息", "", "")

    # Build prompt
    lines = ["你是一个意图识别助手，负责分析用户输入并识别其意图。\n", "## 意图类型\n"]
    idx = 1
    for intent_name, (desc, ex1, ex2) in intent_descriptions.items():
        examples = []
        if ex1:
            examples.append(f'"{ex1}"')
        if ex2:
            examples.append(f'"{ex2}"')
        ex_str = "、".join(examples)
        if ex_str:
            lines.append(f"{idx}. **{intent_name}** - {desc}\n   示例：{ex_str}\n")
        else:
            lines.append(f"{idx}. **{intent_name}** - {desc}\n")
        idx += 1

    lines.append(f"""
## 输出格式

返回 JSON：
{{
  "intent": "意图类型",
  "customer_name": "客户姓名（如果有）",
  "query_topic": "查询主题（如果有）"
}}

## 用户输入
{user_input}
""")
    return "\n".join(lines)


def intent_classify_node(state: AgentState, config=None) -> dict:
    """Classify user intent from the last message.

    Returns:
        dict with keys: intent, customer_name (optional), query_topic (optional)
    """
    # Get last user message
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    if not last_user_msg:
        return {"intent": "general"}

    # Build dynamic intent prompt
    prompt = _build_intent_prompt(last_user_msg, config)

    # Call LLM for classification using configured provider
    try:
        from src.config import load_config
        from src.providers.factory import get_provider

        cfg = config or load_config()
        provider = get_provider(cfg.active_provider, cfg.providers[cfg.active_provider])
        llm = provider.get_model()
        response = llm.invoke([HumanMessage(content=prompt)])
    except Exception:
        # Fallback to rule-based if LLM call fails
        return _rule_based_classify(last_user_msg)

    # Parse JSON response
    try:
        # Extract JSON from response (handle both string and list content)
        content = response.content
        if isinstance(content, list):
            # Anthropic-style: extract text block
            content = next((b["text"] for b in content if b.get("type") == "text"), "")
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())
        return {
            "intent": result.get("intent", "general"),
            "customer_name": result.get("customer_name", ""),
            "query_topic": result.get("query_topic", ""),
        }
    except (json.JSONDecodeError, KeyError):
        return _rule_based_classify(last_user_msg)


def _rule_based_classify(text: str) -> dict:
    """Fallback rule-based intent classification."""
    # Customer query keywords
    customer_keywords = ["档案", "客户", "消费", "服务记录", "女士", "先生"]
    has_customer = any(kw in text for kw in customer_keywords)

    # Knowledge query keywords
    knowledge_keywords = ["产品", "流程", "适合", "推荐", "疗程", "护理"]
    has_knowledge = any(kw in text for kw in knowledge_keywords)

    # Extract customer name (simplified)
    customer_name = ""
    # Match Chinese surname + 女士/先生 pattern
    name_match = re.search(
        r"([张王李赵刘陈杨黄周吴徐孙朱马胡郭林何高梁郑罗宋谢唐韩曹许邓萧冯曾程蔡彭潘袁于董余苏叶吕魏蒋田杜丁沈姜范江傅钟卢汪戴崔任米廖方石姚谭邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段雷钱汤尹黎易常武乔贺赖龚文])(女士|先生)",
        text,
    )
    if name_match:
        customer_name = name_match.group(0)

    if has_customer and has_knowledge:
        return {"intent": "mixed", "customer_name": customer_name}
    elif has_customer:
        return {"intent": "query_customer", "customer_name": customer_name}
    elif has_knowledge:
        return {"intent": "knowledge_query"}
    else:
        return {"intent": "general"}