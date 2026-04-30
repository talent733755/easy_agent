from src.state import AgentState
from src.memory.file_memory import FileMemory


NUDGE_PROMPT = """Review the following conversation and decide if any memory or skill should be saved.

## Update Rules
1. If the user expressed a new preference → respond with USER: <new preference>
2. If you learned new project knowledge → respond with MEMORY: <knowledge>
3. If you solved a non-trivial problem → respond with SKILL: <skill summary>
4. If nothing is worth saving → respond with NOTHING_TO_SAVE

## Conversation
{conversation}
"""


def nudge_check_node(state: AgentState, nudge_interval: int = 10, data_dir: str = "~/.easy_agent") -> dict:
    counter = state.get("nudge_counter", 0)

    if counter < nudge_interval:
        return {"nudge_counter": counter}

    # Trigger review
    fm = FileMemory(data_dir)

    # Extract recent conversation
    recent = state.get("messages", [])[-20:]
    conversation = "\n".join(
        f"[{getattr(m, 'type', 'unknown')}]: {str(m.content)[:300]}"
        for m in recent if hasattr(m, "content")
    )

    # Attempt LLM-based review — if no LLM available, skip
    try:
        from src.config import load_config
        from src.providers.factory import get_provider

        config = load_config()
        provider_config = config.providers.get(state.get("provider_name", config.active_provider))
        if provider_config:
            provider = get_provider(state["provider_name"], provider_config)
            model = provider.get_model()
            prompt = NUDGE_PROMPT.format(conversation=conversation)
            result = model.invoke(prompt)
            content = str(result.content)

            if "USER:" in content:
                user_update = content.split("USER:")[1].split("\n")[0].strip()
                fm.append_user(user_update)
            if "MEMORY:" in content:
                mem_update = content.split("MEMORY:")[1].split("\n")[0].strip()
                fm.append_memory(mem_update)
            if "SKILL:" in content:
                import time
                from pathlib import Path
                skill_dir = Path(data_dir).expanduser() / "skills"
                skill_dir.mkdir(parents=True, exist_ok=True)
                skill_content = content.split("SKILL:")[1].strip()
                date_str = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
                skill_path = skill_dir / f"{date_str}-review.md"
                skill_path.write_text(f"---\ncreated: {date_str}\n---\n\n{skill_content}")
    except Exception:
        pass  # Nudge failure should never block the agent

    return {"nudge_counter": 0}