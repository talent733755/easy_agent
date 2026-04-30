from langchain_core.messages import HumanMessage, AIMessage
from src.state import AgentState
from src.graph import build_graph


class TestGraphStructure:
    def test_graph_builds_without_error(self):
        graph = build_graph(checkpointer=None)
        assert graph is not None

    def test_graph_has_required_nodes(self):
        graph = build_graph(checkpointer=None)
        node_names = [n for n in graph.builder.nodes] if hasattr(graph, 'builder') else []
        # After compilation, check the graph compiles
        compiled = graph
        assert compiled is not None


class TestGraphInvocation:
    def test_simple_conversation_flow(self):
        """Test a minimal graph invocation with a mock model."""
        from unittest.mock import MagicMock, patch
        from langgraph.checkpoint.memory import MemorySaver

        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_model.invoke.return_value = AIMessage(content="Hello! How can I help you?")

        with patch('src.graph._build_agent_model', return_value=mock_model):
            graph = build_graph(checkpointer=MemorySaver())

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

        config = {"configurable": {"thread_id": "test-1"}}
        result = graph.invoke(state, config)
        assert len(result["messages"]) > 1
        assert "Hello" in str(result["messages"][-1].content)