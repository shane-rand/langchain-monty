"""Shared helpers for the langchain-tests standard suites.

Not a test module itself (no ``test_`` prefix) — imported by the
``unit_tests``/``integration_tests`` standard-test subclasses.
"""

from __future__ import annotations

from langgraph.prebuilt.tool_node import ToolRuntime


def make_tool_runtime(tool_call_id: str = "standard-tests-call") -> ToolRuntime:
    """Build a standalone ``ToolRuntime`` for invoking ``eval_python`` outside a graph.

    ``eval_python`` declares a required ``runtime: ToolRuntime`` parameter that
    LangGraph's ``ToolNode`` normally injects. The standard tests in
    ``langchain_tests`` invoke tools directly (no graph), so we supply an
    equivalent runtime object as a regular arg instead.
    """
    return ToolRuntime(
        state={},
        context=None,
        config={},
        stream_writer=lambda _: None,
        tool_call_id=tool_call_id,
        store=None,
    )
