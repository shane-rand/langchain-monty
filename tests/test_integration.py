"""Integration tests: MontyCodeInterpreterMiddleware wired into a real deep agent.

Uses a minimal fake chat model that supports bind_tools so create_deep_agent
can be called without a real LLM API key. Monty itself runs for real — these
tests exercise the full eval_python tool-call loop.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from langchain_monty import MontyCodeInterpreterMiddleware, MontyLimits

# --------------------------------------------------------------------------- #
# Fake model                                                                  #
# --------------------------------------------------------------------------- #


class _FakeModel(BaseChatModel):
    """Minimal fake chat model that cycles through pre-configured responses.

    Supports ``bind_tools`` (returns self) so it can be passed to
    ``create_deep_agent`` without a real LLM backend.
    """

    responses: list[AIMessage]
    i: int = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> _FakeModel:
        return self

    def _generate(
        self,
        messages: list,
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = self.responses[self.i % len(self.responses)]
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-capable-model"


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _eval_call(code: str) -> AIMessage:
    """AIMessage that requests a single eval_python tool call."""
    return AIMessage(
        content="",
        tool_calls=[
            ToolCall(id=str(uuid.uuid4()), name="eval_python", args={"code": code})
        ],
    )


def _done(text: str = "done") -> AIMessage:
    return AIMessage(content=text)


def _agent(*responses: AIMessage, middleware=None, tools=None):
    if middleware is None:
        middleware = MontyCodeInterpreterMiddleware()
    return create_deep_agent(
        model=_FakeModel(responses=list(responses)),
        tools=tools or [],
        middleware=[middleware],
    )


def _tool_result(agent_result: dict) -> dict:
    """Extract the ToolMessage payload (index 2) as a parsed dict."""
    return json.loads(agent_result["messages"][2].content)


# --------------------------------------------------------------------------- #
# Agent creation                                                               #
# --------------------------------------------------------------------------- #


class TestAgentCreation:
    def test_returns_compiled_state_graph(self):
        from langgraph.graph.state import CompiledStateGraph

        agent = _agent(_done())
        assert isinstance(agent, CompiledStateGraph)

    def test_eval_python_tool_registered(self):
        agent = _agent(_done())
        tool_names = {n for n in agent.get_graph().nodes}
        assert "tools" in tool_names

    def test_multiple_middleware_instances_independent(self):
        @tool
        def x() -> int:
            """Tool x."""
            return 1

        @tool
        def y() -> int:
            """Tool y."""
            return 2

        a1 = _agent(_done(), middleware=MontyCodeInterpreterMiddleware(ptc=[x]))
        a2 = _agent(_done(), middleware=MontyCodeInterpreterMiddleware(ptc=[y]))
        assert a1 is not a2


# --------------------------------------------------------------------------- #
# eval_python — expression evaluation                                          #
# --------------------------------------------------------------------------- #


class TestExpressionEvaluation:
    def _run(self, code: str, middleware=None) -> dict:
        agent = _agent(_eval_call(code), _done(), middleware=middleware)
        result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
        return _tool_result(result)

    def test_integer_arithmetic(self):
        r = self._run("2 ** 10")
        assert r["result"] == 1024
        assert r["error"] is None

    def test_string_expression(self):
        r = self._run("'hi' * 3")
        assert r["result"] == "hihihi"

    def test_list_comprehension(self):
        r = self._run("[x * x for x in range(4)]")
        assert r["result"] == [0, 1, 4, 9]

    def test_dict_expression(self):
        r = self._run("{'a': 1, 'b': 2}")
        assert r["result"] == {"a": 1, "b": 2}

    def test_statement_only_gives_null_result(self):
        r = self._run("x = 42")
        assert r["result"] is None
        assert r["error"] is None

    def test_multiline_code(self):
        r = self._run("x = 3\ny = 4\nx * y")
        assert r["result"] == 12


# --------------------------------------------------------------------------- #
# eval_python — error handling                                                 #
# --------------------------------------------------------------------------- #


class TestErrorHandling:
    def _run(self, code: str) -> dict:
        agent = _agent(_eval_call(code), _done())
        result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
        return _tool_result(result)

    def test_syntax_error_returns_structured_error(self):
        r = self._run("???")
        assert r["result"] is None
        assert r["error"] is not None
        assert isinstance(r["error"]["type"], str)
        assert len(r["error"]["message"]) > 0

    def test_runtime_error_returns_structured_error(self):
        r = self._run("1 / 0")
        assert r["error"] is not None
        assert "ZeroDivisionError" in r["error"]["message"]

    def test_index_out_of_range_returns_structured_error(self):
        r = self._run("[1, 2][99]")
        assert r["error"] is not None

    def test_result_is_null_on_error(self):
        r = self._run("???")
        assert r["result"] is None


# --------------------------------------------------------------------------- #
# Programmatic tool calling (ptc)                                              #
# --------------------------------------------------------------------------- #


class TestPtcHostToolCalling:
    def test_allowlisted_tool_called_and_result_returned(self):
        @tool
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        agent = _agent(
            _eval_call("multiply(6, 7)"),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(ptc=[multiply]),
            tools=[multiply],
        )
        result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
        r = _tool_result(result)
        assert r["result"] == 42
        assert r["error"] is None

    def test_chained_host_calls(self):
        @tool
        def add_one(n: int) -> int:
            """Add one to n."""
            return n + 1

        # Call add_one twice: add_one(add_one(5)) == 7
        code = "x = add_one(5)\nadd_one(x)"
        agent = _agent(
            _eval_call(code),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(ptc=[add_one]),
            tools=[add_one],
        )
        result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
        r = _tool_result(result)
        assert r["result"] == 7

    def test_tool_not_in_ptc_does_not_return_its_value(self):
        @tool
        def secret(x: int) -> int:
            """A secret tool not in the allowlist."""
            return x

        agent = _agent(
            _eval_call("secret(99)"),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(ptc=[]),
            tools=[secret],
        )
        result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
        r = _tool_result(result)
        # The secret tool's return value (99) must never reach the result
        assert r.get("result") != 99


# --------------------------------------------------------------------------- #
# System prompt configuration                                                  #
# --------------------------------------------------------------------------- #


class TestSystemPromptConfig:
    def test_default_system_prompt_agent_works(self):
        agent = _agent(_done("ok"))
        result = agent.invoke({"messages": [{"role": "user", "content": "hi"}]})
        assert result["messages"][-1].content == "ok"

    def test_system_prompt_none_agent_still_works(self):
        agent = _agent(
            _done("ok"),
            middleware=MontyCodeInterpreterMiddleware(system_prompt=None),
        )
        result = agent.invoke({"messages": [{"role": "user", "content": "hi"}]})
        assert result["messages"][-1].content == "ok"

    def test_system_prompt_none_eval_python_still_executes(self):
        agent = _agent(
            _eval_call("10 + 5"),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(system_prompt=None),
        )
        result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
        r = _tool_result(result)
        assert r["result"] == 15


# --------------------------------------------------------------------------- #
# Custom limits                                                                #
# --------------------------------------------------------------------------- #


class TestCustomLimits:
    def test_custom_limits_accepted_and_code_runs(self):
        limits = MontyLimits(max_duration_secs=2.0, max_allocations=500_000)
        agent = _agent(
            _eval_call("sum(range(100))"),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(limits=limits),
        )
        result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
        r = _tool_result(result)
        assert r["result"] == 4950

    def test_strict_limits_still_execute_simple_code(self):
        # Tight limits are stored and passed through; simple code still runs.
        limits = MontyLimits(max_duration_secs=1.0, max_allocations=10_000)
        agent = _agent(
            _eval_call("1 + 1"),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(limits=limits),
        )
        result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
        r = _tool_result(result)
        assert r["result"] == 2
        assert r["error"] is None


# --------------------------------------------------------------------------- #
# Async invocation                                                             #
# --------------------------------------------------------------------------- #


class TestAsyncInvocation:
    @pytest.mark.asyncio
    async def test_async_simple_arithmetic(self):
        agent = _agent(_eval_call("7 * 8"), _done())
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})
        r = _tool_result(result)
        assert r["result"] == 56
        assert r["error"] is None

    @pytest.mark.asyncio
    async def test_async_syntax_error(self):
        agent = _agent(_eval_call("???"), _done())
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})
        r = _tool_result(result)
        assert r["error"] is not None

    @pytest.mark.asyncio
    async def test_async_with_ptc_host_tool(self):
        @tool
        def square(n: int) -> int:
            """Square a number."""
            return n * n

        agent = _agent(
            _eval_call("square(9)"),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(ptc=[square]),
            tools=[square],
        )
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})
        r = _tool_result(result)
        assert r["result"] == 81
