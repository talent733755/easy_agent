from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from src.state import AgentState

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

## Your Capabilities
- Execute tools to help users accomplish tasks
- Search and recall past conversations from memory
- Manage files, run shell commands, search the web

## Memory Context
{memory_context}

## User Profile
{user_profile}

## Agent Notes
{agent_notes}

## Guidelines
- Use tools when needed to get accurate information
- If a tool execution is dangerous, it will require user confirmation
- Be concise and direct in your responses
- If you need human input, include [HUMAN_INPUT: your question] in your response
"""


def create_agent_node(model: BaseChatModel, tools: list = None):
    tools = tools or []

    def agent_node(state: AgentState) -> dict:
        # Build system message with memory context
        system_content = SYSTEM_PROMPT.format(
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