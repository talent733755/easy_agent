import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.state import AgentState
from src.nodes.agent_node import create_agent_node


class TestAgentNode:
    def test_agent_node_increments_iteration(self):
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_model.invoke.return_value = AIMessage(content="Hello!")

        node_fn = create_agent_node(mock_model)
        state: AgentState = {
            "messages": [HumanMessage(content="Hi")],
            "tools": [],
            "context_window": {"used_tokens": 0, "max_tokens": 128000, "threshold": 0.7},
            "memory_context": "",
            "user_profile": "",
            "agent_notes": "",
            "pending_human_input": False,
            "iteration_count": 0,
            "max_iterations": 15,
            "nudge_counter": 0,
            "provider_name": "zhipu",
        }

        result = node_fn(state)
        assert result["iteration_count"] == 1
        assert len(result["messages"]) == 1  # Returns only new messages

    def test_agent_node_injects_memory_context(self):
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model

        captured_messages = []
        def capture_invoke(msgs, **kwargs):
            captured_messages.extend(msgs)
            return AIMessage(content="Got it!")
        mock_model.invoke = capture_invoke

        node_fn = create_agent_node(mock_model)
        state: AgentState = {
            "messages": [HumanMessage(content="What did I ask before?")],
            "tools": [],
            "context_window": {"used_tokens": 0, "max_tokens": 128000, "threshold": 0.7},
            "memory_context": "Previous: user asked about Python",
            "user_profile": "Backend developer",
            "agent_notes": "User uses pnpm",
            "pending_human_input": False,
            "iteration_count": 0,
            "max_iterations": 15,
            "nudge_counter": 0,
            "provider_name": "zhipu",
        }

        node_fn(state)
        # Check that system message with context was injected
        system_msgs = [m for m in captured_messages if hasattr(m, 'content') and 'Backend developer' in str(m.content)]
        assert len(system_msgs) >= 1


from langchain_core.tools import tool as tool_decorator
from src.nodes.tool_executor import ToolExecutor
from src.nodes.observe_node import observe_node


class TestToolExecutor:
    def test_execute_single_tool(self):
        @tool_decorator
        def echo(text: str) -> str:
            """Echo back the text."""
            return text

        executor = ToolExecutor({"echo": echo})
        mock_call = MagicMock()
        mock_call.name = "echo"
        mock_call.args = {"text": "hello"}
        mock_call.id = "call_1"

        results = executor.execute([mock_call])
        assert len(results) == 1
        assert results[0].content == "hello"
        assert results[0].tool_call_id == "call_1"

    def test_execute_tool_error_is_captured(self):
        @tool_decorator
        def bad_tool(x: str) -> str:
            """A tool that raises an error."""
            raise ValueError("something wrong")

        executor = ToolExecutor({"bad_tool": bad_tool})
        mock_call = MagicMock()
        mock_call.name = "bad_tool"
        mock_call.args = {"x": "test"}
        mock_call.id = "call_err"

        results = executor.execute([mock_call])
        assert len(results) == 1
        assert "Error" in results[0].content

    def test_execute_unknown_tool(self):
        executor = ToolExecutor({})
        mock_call = MagicMock()
        mock_call.name = "nonexistent"
        mock_call.args = {}
        mock_call.id = "call_x"

        results = executor.execute([mock_call])
        assert len(results) == 1
        assert "not found" in results[0].content.lower()


class TestObserveNode:
    def test_observe_node_formats_tool_messages(self):
        msgs = [
            HumanMessage(content="Hi"),
            AIMessage(content="", tool_calls=[{"name": "echo", "args": {"text": "hi"}, "id": "c1"}]),
            ToolMessage(content="hi", tool_call_id="c1"),
        ]
        state = {
            "messages": msgs,
            "tools": [],
            "context_window": {"used_tokens": 0, "max_tokens": 128000, "threshold": 0.7},
            "memory_context": "",
            "user_profile": "",
            "agent_notes": "",
            "pending_human_input": False,
            "iteration_count": 1,
            "max_iterations": 15,
            "nudge_counter": 0,
            "provider_name": "zhipu",
        }
        result = observe_node(state)
        # Should add an observation message
        assert len(result["messages"]) == 1  # Returns only new messages


from src.nodes.human_gate import human_gate, TOOL_RISK_LEVELS


class TestHumanGate:
    def test_dangerous_tool_triggers_interrupt(self):
        @tool_decorator
        def dangerous_cmd(cmd: str) -> str:
            """A dangerous command tool."""
            return "done"

        TOOL_RISK_LEVELS["dangerous_cmd"] = "dangerous"
        try:
            msgs = [
                HumanMessage(content="run a command"),
                AIMessage(content="", tool_calls=[{"name": "dangerous_cmd", "args": {"cmd": "rm -rf /"}, "id": "c1"}]),
            ]
            state = {
                "messages": msgs,
                "tools": [dangerous_cmd],
                "context_window": {"used_tokens": 0, "max_tokens": 128000, "threshold": 0.7},
                "memory_context": "",
                "user_profile": "",
                "agent_notes": "",
                "pending_human_input": False,
                "iteration_count": 1,
                "max_iterations": 15,
                "nudge_counter": 0,
                "provider_name": "zhipu",
            }
            result = human_gate(state)
            assert result["pending_human_input"] is True
        finally:
            del TOOL_RISK_LEVELS["dangerous_cmd"]

    def test_safe_tool_passes_through(self):
        state = {
            "messages": [],
            "tools": [],
            "context_window": {"used_tokens": 0, "max_tokens": 128000, "threshold": 0.7},
            "memory_context": "",
            "user_profile": "",
            "agent_notes": "",
            "pending_human_input": False,
            "iteration_count": 0,
            "max_iterations": 15,
            "nudge_counter": 0,
            "provider_name": "zhipu",
        }
        result = human_gate(state)
        assert result["pending_human_input"] is False


import os
import tempfile
from src.nodes.memory_retrieve import memory_retrieve_node
from src.nodes.memory_save import memory_save_node


class TestMemoryRetrieve:
    def test_retrieves_from_file_memory(self):
        tmpdir = tempfile.mkdtemp()
        try:
            fm_path = os.path.join(tmpdir, "MEMORY.md")
            with open(fm_path, "w") as f:
                f.write("User prefers pnpm")
            um_path = os.path.join(tmpdir, "USER.md")
            with open(um_path, "w") as f:
                f.write("Backend developer")

            state = {
                "messages": [HumanMessage(content="What package manager should I use?")],
                "tools": [],
                "context_window": {"used_tokens": 0, "max_tokens": 128000, "threshold": 0.7},
                "memory_context": "",
                "user_profile": "",
                "agent_notes": "",
                "pending_human_input": False,
                "iteration_count": 0,
                "max_iterations": 15,
                "nudge_counter": 0,
                "provider_name": "zhipu",
            }
            result = memory_retrieve_node(state, data_dir=tmpdir)
            assert "pnpm" in result["agent_notes"] or "pnpm" in result["memory_context"]
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestMemorySave:
    def test_saves_messages_to_fts5(self):
        tmpdir = tempfile.mkdtemp()
        try:
            state = {
                "messages": [
                    HumanMessage(content="Hello"),
                    AIMessage(content="Hi there!"),
                ],
                "tools": [],
                "context_window": {"used_tokens": 0, "max_tokens": 128000, "threshold": 0.7},
                "memory_context": "",
                "user_profile": "",
                "agent_notes": "",
                "pending_human_input": False,
                "iteration_count": 1,
                "max_iterations": 15,
                "nudge_counter": 0,
                "provider_name": "zhipu",
            }
            result = memory_save_node(state, data_dir=tmpdir)
            assert result == {}  # No state changes needed
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


from src.nodes.nudge_check import nudge_check_node


class TestNudgeCheck:
    def test_nudge_not_triggered_below_threshold(self):
        state = {
            "messages": [HumanMessage(content="hi")],
            "tools": [],
            "context_window": {"used_tokens": 0, "max_tokens": 128000, "threshold": 0.7},
            "memory_context": "",
            "user_profile": "",
            "agent_notes": "",
            "pending_human_input": False,
            "iteration_count": 3,
            "max_iterations": 15,
            "nudge_counter": 3,
            "provider_name": "zhipu",
        }
        result = nudge_check_node(state, nudge_interval=10)
        assert result["nudge_counter"] == 3  # unchanged

    def test_nudge_triggers_at_threshold(self):
        state = {
            "messages": [
                HumanMessage(content="I prefer vim over vscode"),
                AIMessage(content="Noted, I'll remember that."),
            ],
            "tools": [],
            "context_window": {"used_tokens": 0, "max_tokens": 128000, "threshold": 0.7},
            "memory_context": "",
            "user_profile": "",
            "agent_notes": "",
            "pending_human_input": False,
            "iteration_count": 10,
            "max_iterations": 15,
            "nudge_counter": 10,
            "provider_name": "zhipu",
        }
        # nudge shouldn't crash even without LLM access
        result = nudge_check_node(state, nudge_interval=10)
        assert result["nudge_counter"] == 0  # reset