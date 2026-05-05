"""Intent classification node for beauty agent."""
import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.state import AgentState


def intent_classify_node(state: AgentState) -> dict:
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

    # Load intent prompt
    prompt_path = Path("prompts/beauty_intent_v1.txt")
    if not prompt_path.exists():
        # Fallback to simple rule-based classification
        return _rule_based_classify(last_user_msg)

    prompt_template = prompt_path.read_text(encoding="utf-8")
    prompt = prompt_template.format(user_input=last_user_msg)

    # Call LLM for classification
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke([HumanMessage(content=prompt)])

    # Parse JSON response
    try:
        # Extract JSON from response
        content = response.content
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