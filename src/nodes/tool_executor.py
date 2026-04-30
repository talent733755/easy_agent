from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage


class ToolExecutor:
    def __init__(self, tools_by_name: dict[str, BaseTool]):
        self.tools = tools_by_name

    def execute(self, tool_calls: list) -> list[ToolMessage]:
        results = []
        for call in tool_calls:
            tool = self.tools.get(call.name)
            if tool is None:
                results.append(ToolMessage(
                    content=f"Error: tool '{call.name}' not found",
                    tool_call_id=call.id,
                ))
                continue
            try:
                output = tool.invoke(call.args)
                output_str = str(output)
                if len(output_str) > 4000:
                    output_str = output_str[:4000] + "\n... (truncated)"
                results.append(ToolMessage(
                    content=output_str,
                    tool_call_id=call.id,
                ))
            except Exception as e:
                results.append(ToolMessage(
                    content=f"Error executing {call.name}: {e}",
                    tool_call_id=call.id,
                ))
        return results