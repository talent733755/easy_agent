from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from src.state import AgentState

SYSTEM_PROMPT = """You are a helpful AI assistant for beauty salon staff (娇莉芙).

## 知识库与网络搜索决策规范（必须严格遵守）

### 决策顺序

1. **知识库有结果 → 必须使用知识库回答**
   - 基于知识库内容回答，不要自行补充未经核实的知识
   - 在回答末尾标注来源：格式 `【来源：products/嫩小白.txt】`
   - 多来源用顿号分隔：`【来源：products/嫩小白.txt、procedures/客户接待流程.txt】`

2. **知识库无结果 + 是美容/健康/门店相关问题 → 禁止网络搜索**
   - 直接回复：「抱歉，该问题超出了我的知识库范围，建议您咨询门店顾问获取准确信息。」
   - 禁止捏造答案

3. **知识库无结果 + 非美容行业（如天气、新闻、通用知识）→ 可用 web_search**
   - 搜索后综合回答，标注搜索来源（URL 标题）

4. **搜索也无结果 → 诚实告知**
   - 「抱歉，我无法回答该问题，建议联系门店工作人员。」

### 工具使用规则

| 场景 | web_search |
|------|-----------|
| 知识库有结果 | ❌ 禁止使用 |
| 美容行业问题、知识库无结果 | ❌ 禁止使用 |
| 通用/实时问题、知识库无结果 | ✅ 可使用 |
| 搜索无结果 | 诚实回复 |

## 知识库

{knowledge_results}

## Memory Context

{memory_context}

## User Profile

{user_profile}

## Agent Notes

{agent_notes}

## Guidelines

- 回答美容相关问题时，永远优先使用知识库内容
- 知识库没有的美容问题，直接说超范围，绝不胡编
- 通用/实时问题（如天气、新闻）可用网络搜索
- 如果需要人工介入，在回复中加入 [HUMAN_INPUT: 问题内容]
"""


def _format_knowledge_results(results: list) -> str:
    """Format knowledge results for the system prompt."""
    if not results:
        return "(No knowledge base results available)"
    parts = []
    for r in results:
        parts.append(
            f"【来源：{r.get('source', 'unknown')}】\n{r.get('content', '')}"
        )
    return "\n\n".join(parts)


def create_agent_node(model: BaseChatModel, tools: list = None):
    tools = tools or []

    def agent_node(state: AgentState) -> dict:
        # Format knowledge base results
        knowledge_results = _format_knowledge_results(
            state.get("knowledge_results", [])
        )

        # Build system message with all context
        system_content = SYSTEM_PROMPT.format(
            knowledge_results=knowledge_results,
            memory_context=state.get("memory_context", ""),
            user_profile=state.get("user_profile", ""),
            agent_notes=state.get("agent_notes", ""),
        )
        system_msg = SystemMessage(content=system_content)

        # Prepend system message to conversation
        messages = [system_msg] + list(state["messages"])

        # Bind tools to LLM so it knows about them
        if tools:
            llm = model.bind_tools(tools)
        else:
            llm = model

        response = llm.invoke(messages)

        return {
            "messages": [response],
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    return agent_node