from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage


class ToolExecutor:
    def __init__(self, tools_by_name: dict[str, BaseTool]):
        self.tools = tools_by_name

    def _get_name(self, call) -> str:
        return call["name"] if isinstance(call, dict) else call.name

    def _get_args(self, call) -> dict:
        return call["args"] if isinstance(call, dict) else call.args

    def _get_id(self, call) -> str:
        return call["id"] if isinstance(call, dict) else call.id

    def execute(self, tool_calls: list) -> list[ToolMessage]:
        results = []
        for call in tool_calls:
            name = self._get_name(call)
            args = self._get_args(call)
            call_id = self._get_id(call)
            tool = self.tools.get(name)
            if tool is None:
                results.append(ToolMessage(
                    content=f"Error: tool '{name}' not found",
                    tool_call_id=call_id,
                ))
                continue
            try:
                output = tool.invoke(args)
                output_str = str(output)
                if len(output_str) > 4000:
                    output_str = output_str[:4000] + "\n... (truncated)"
                results.append(ToolMessage(
                    content=output_str,
                    tool_call_id=call_id,
                ))
            except Exception as e:
                results.append(ToolMessage(
                    content=f"Error executing {name}: {e}",
                    tool_call_id=call_id,
                ))
        return results