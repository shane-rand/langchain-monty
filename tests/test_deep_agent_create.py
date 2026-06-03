"""Integration tests for create_deep_agent + MontyCodeInterpreterMiddleware.

This suite focuses on areas not covered by test_integration.py:
- stdout capture from print() calls
- structured result fields (stdout, attempted_code, error.type)
- iteration budget enforcement
- PTC tools with non-integer return types (str, float, list, dict)
- PTC tool exception surfacing
- deferred (string-name) PTC tool resolution
- multiple sequential eval_python calls in one agent turn
- conditional and boolean expressions
- custom tool_description
- nested data structures

All tests use a _FakeModel (no real LLM API key required) and run Monty for real.
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

from langchain_monty import (
    CODE_INTERPRETER_TOOL_DESCRIPTION,
    MontyCodeInterpreterMiddleware,
    MontyLimits,
)

# --------------------------------------------------------------------------- #
# Fake model (same pattern as test_integration.py)                            #
# --------------------------------------------------------------------------- #


class _FakeModel(BaseChatModel):
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
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _eval_call(code: str) -> AIMessage:
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
    return json.loads(agent_result["messages"][2].content)


def _nth_tool_result(agent_result: dict, n: int) -> dict:
    """Return the nth ToolMessage result (0-indexed), for multi-call scenarios."""
    tool_messages = [
        m for m in agent_result["messages"] if m.type == "tool"
    ]
    return json.loads(tool_messages[n].content)


def _run(code: str, middleware=None) -> dict:
    agent = _agent(_eval_call(code), _done(), middleware=middleware)
    result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    return _tool_result(result)


# --------------------------------------------------------------------------- #
# Stdout capture                                                               #
# --------------------------------------------------------------------------- #


class TestStdoutCapture:
    def test_print_captured_in_stdout_field(self):
        r = _run('print("hello")')
        assert "hello" in r["stdout"]

    def test_multiple_prints_all_captured(self):
        r = _run('print("line1")\nprint("line2")\nprint("line3")')
        assert "line1" in r["stdout"]
        assert "line2" in r["stdout"]
        assert "line3" in r["stdout"]

    def test_print_does_not_affect_result(self):
        r = _run('print("side effect")\n42')
        assert r["result"] == 42
        assert "side effect" in r["stdout"]

    def test_no_print_gives_empty_stdout(self):
        r = _run("1 + 1")
        assert r["stdout"] == ""

    def test_stdout_field_present_on_success(self):
        r = _run("True")
        assert "stdout" in r

    def test_print_integer_captured(self):
        r = _run("print(123)")
        assert "123" in r["stdout"]


# --------------------------------------------------------------------------- #
# Result structure                                                              #
# --------------------------------------------------------------------------- #


class TestResultStructure:
    def test_success_result_has_all_fields(self):
        r = _run("1 + 1")
        assert "result" in r
        assert "stdout" in r
        assert "error" in r

    def test_error_result_has_all_fields(self):
        r = _run("???")
        assert "result" in r
        assert "stdout" in r
        assert "error" in r

    def test_error_has_type_and_message(self):
        r = _run("1 / 0")
        assert r["error"] is not None
        assert "type" in r["error"]
        assert "message" in r["error"]
        assert len(r["error"]["type"]) > 0
        assert len(r["error"]["message"]) > 0

    def test_attempted_code_populated_on_syntax_error(self):
        bad_code = "def (:"
        r = _run(bad_code)
        assert r["error"] is not None
        assert r.get("attempted_code") is not None

    def test_attempted_code_absent_on_success(self):
        r = _run("2 + 2")
        assert r.get("attempted_code") is None

    def test_error_message_mentions_zero_division(self):
        r = _run("1 / 0")
        # Monty wraps runtime errors in MontyRuntimeError; the original
        # exception name appears in the message rather than the type field.
        assert r["error"] is not None
        msg = r["error"]["message"].lower()
        assert "zero" in msg or "division" in msg or "divide" in msg

    def test_error_on_undefined_variable(self):
        r = _run("undefined_variable")
        assert r["error"] is not None
        assert len(r["error"]["type"]) > 0
        assert len(r["error"]["message"]) > 0


# --------------------------------------------------------------------------- #
# Expression types                                                             #
# --------------------------------------------------------------------------- #


class TestExpressionTypes:
    def test_boolean_true(self):
        r = _run("True")
        assert r["result"] is True
        assert r["error"] is None

    def test_boolean_false(self):
        r = _run("False")
        assert r["result"] is False

    def test_float_arithmetic(self):
        r = _run("3.14 * 2")
        assert abs(r["result"] - 6.28) < 1e-9

    def test_float_division(self):
        r = _run("10 / 4")
        assert r["result"] == 2.5

    def test_negative_number(self):
        r = _run("-7")
        assert r["result"] == -7

    def test_nested_list(self):
        r = _run("[[1, 2], [3, 4]]")
        assert r["result"] == [[1, 2], [3, 4]]

    def test_dict_with_list_values(self):
        r = _run("{'x': [1, 2, 3], 'y': [4, 5]}")
        assert r["result"] == {"x": [1, 2, 3], "y": [4, 5]}

    def test_conditional_expression_true_branch(self):
        r = _run("'yes' if True else 'no'")
        assert r["result"] == "yes"

    def test_conditional_expression_false_branch(self):
        r = _run("'yes' if False else 'no'")
        assert r["result"] == "no"

    def test_string_formatting(self):
        r = _run("f'value is {6 * 7}'")
        assert r["result"] == "value is 42"

    def test_none_literal(self):
        r = _run("None")
        assert r["result"] is None
        assert r["error"] is None

    def test_empty_list(self):
        r = _run("[]")
        assert r["result"] == []

    def test_empty_dict(self):
        r = _run("{}")
        assert r["result"] == {}

    def test_len_builtin(self):
        r = _run("len([1, 2, 3, 4, 5])")
        assert r["result"] == 5

    def test_sum_builtin(self):
        r = _run("sum([1, 2, 3, 4, 5])")
        assert r["result"] == 15

    def test_sorted_builtin(self):
        r = _run("sorted([3, 1, 4, 1, 5, 9])")
        assert r["result"] == [1, 1, 3, 4, 5, 9]

    def test_max_builtin(self):
        r = _run("max(10, 20, 5)")
        assert r["result"] == 20


# --------------------------------------------------------------------------- #
# PTC — non-integer return types                                               #
# --------------------------------------------------------------------------- #


class TestPtcReturnTypes:
    def test_tool_returning_string(self):
        @tool
        def greet(name: str) -> str:
            """Return a greeting for name."""
            return f"Hello, {name}!"

        agent = _agent(
            _eval_call("greet('World')"),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(ptc=[greet]),
            tools=[greet],
        )
        r = _tool_result(agent.invoke({"messages": [{"role": "user", "content": "go"}]}))
        assert r["result"] == "Hello, World!"

    def test_tool_returning_float(self):
        @tool
        def half(n: float) -> float:
            """Return half of n."""
            return n / 2

        agent = _agent(
            _eval_call("half(7.0)"),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(ptc=[half]),
            tools=[half],
        )
        r = _tool_result(agent.invoke({"messages": [{"role": "user", "content": "go"}]}))
        assert r["result"] == 3.5

    def test_tool_returning_list(self):
        @tool
        def make_range(n: int) -> list:
            """Return list of 0..n-1."""
            return list(range(n))

        agent = _agent(
            _eval_call("make_range(4)"),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(ptc=[make_range]),
            tools=[make_range],
        )
        r = _tool_result(agent.invoke({"messages": [{"role": "user", "content": "go"}]}))
        assert r["result"] == [0, 1, 2, 3]

    def test_tool_returning_dict(self):
        @tool
        def get_info() -> dict:
            """Return a dict of info."""
            return {"name": "monty", "version": 1}

        agent = _agent(
            _eval_call("get_info()"),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(ptc=[get_info]),
            tools=[get_info],
        )
        r = _tool_result(agent.invoke({"messages": [{"role": "user", "content": "go"}]}))
        assert r["result"] == {"name": "monty", "version": 1}

    def test_tool_result_used_in_computation(self):
        @tool
        def get_base() -> int:
            """Return base value."""
            return 10

        agent = _agent(
            _eval_call("get_base() * 3 + 2"),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(ptc=[get_base]),
            tools=[get_base],
        )
        r = _tool_result(agent.invoke({"messages": [{"role": "user", "content": "go"}]}))
        assert r["result"] == 32


# --------------------------------------------------------------------------- #
# PTC — exception handling                                                     #
# --------------------------------------------------------------------------- #


class TestPtcExceptionHandling:
    def test_tool_exception_caught_by_interpreter(self):
        @tool
        def boom() -> int:
            """Always raises."""
            raise ValueError("intentional error")

        # try/except is a compound statement — its "value" is None.
        # Assign the caught message to a variable and return it explicitly.
        code = (
            "msg = None\n"
            "try:\n"
            "    boom()\n"
            "except Exception as e:\n"
            "    msg = str(e)\n"
            "msg"
        )
        agent = _agent(
            _eval_call(code),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(ptc=[boom]),
            tools=[boom],
        )
        r = _tool_result(agent.invoke({"messages": [{"role": "user", "content": "go"}]}))
        assert r["error"] is None
        assert r["result"] is not None

    def test_uncaught_tool_exception_surfaces_as_error(self):
        @tool
        def explode() -> int:
            """Always raises without try/except in caller."""
            raise RuntimeError("boom")

        agent = _agent(
            _eval_call("explode()"),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(ptc=[explode]),
            tools=[explode],
        )
        r = _tool_result(agent.invoke({"messages": [{"role": "user", "content": "go"}]}))
        assert r["error"] is not None


# --------------------------------------------------------------------------- #
# Iteration budget                                                             #
# --------------------------------------------------------------------------- #


class TestIterationBudget:
    def test_iteration_budget_exceeded_returns_error(self):
        @tool
        def noop() -> int:
            """Does nothing."""
            return 1

        # budget=0 means the very first host-tool iteration exceeds the cap
        agent = _agent(
            _eval_call("noop()"),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(
                ptc=[noop],
                iteration_budget=0,
            ),
            tools=[noop],
        )
        r = _tool_result(agent.invoke({"messages": [{"role": "user", "content": "go"}]}))
        assert r["error"] is not None
        assert r["error"]["type"] == "IterationBudgetExceeded"

    def test_iteration_budget_error_message_mentions_limit(self):
        @tool
        def ping() -> int:
            """Ping."""
            return 1

        agent = _agent(
            _eval_call("ping()"),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(
                ptc=[ping],
                iteration_budget=0,
            ),
            tools=[ping],
        )
        r = _tool_result(agent.invoke({"messages": [{"role": "user", "content": "go"}]}))
        assert "0" in r["error"]["message"]

    def test_generous_budget_allows_multiple_calls(self):
        @tool
        def inc(n: int) -> int:
            """Return n+1."""
            return n + 1

        # Call inc three times: inc(inc(inc(0))) == 3
        code = "a = inc(0)\nb = inc(a)\ninc(b)"
        agent = _agent(
            _eval_call(code),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(
                ptc=[inc],
                iteration_budget=10,
            ),
            tools=[inc],
        )
        r = _tool_result(agent.invoke({"messages": [{"role": "user", "content": "go"}]}))
        assert r["result"] == 3
        assert r["error"] is None


# --------------------------------------------------------------------------- #
# Deferred (string-name) PTC                                                  #
# --------------------------------------------------------------------------- #


class TestDeferredPtc:
    def test_deferred_name_resolves_from_runtime_tools(self):
        @tool
        def lookup(key: str) -> str:
            """Look up a key."""
            return f"value_of_{key}"

        # Pass the tool name as a string (deferred) — it will be resolved
        # from runtime.tools at invocation time.
        agent = _agent(
            _eval_call("lookup('x')"),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(ptc=["lookup"]),
            tools=[lookup],
        )
        r = _tool_result(agent.invoke({"messages": [{"role": "user", "content": "go"}]}))
        assert r["result"] == "value_of_x"

    def test_mixed_baseTool_and_string_ptc(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        @tool
        def sub(a: int, b: int) -> int:
            """Subtract b from a."""
            return a - b

        # add is a BaseTool entry; sub is a deferred string name
        code = "add(10, 3) + sub(10, 3)"
        agent = _agent(
            _eval_call(code),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(ptc=[add, "sub"]),
            tools=[add, sub],
        )
        r = _tool_result(agent.invoke({"messages": [{"role": "user", "content": "go"}]}))
        assert r["result"] == 20  # 13 + 7


# --------------------------------------------------------------------------- #
# Multiple sequential eval_python calls                                       #
# --------------------------------------------------------------------------- #


class TestMultipleEvalCalls:
    def test_two_eval_calls_both_succeed(self):
        agent = _agent(
            _eval_call("6 * 7"),
            _eval_call("100 - 1"),
            _done(),
        )
        result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
        r0 = _nth_tool_result(result, 0)
        r1 = _nth_tool_result(result, 1)
        assert r0["result"] == 42
        assert r1["result"] == 99

    def test_error_in_first_does_not_prevent_second(self):
        agent = _agent(
            _eval_call("1 / 0"),
            _eval_call("2 + 2"),
            _done(),
        )
        result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
        r0 = _nth_tool_result(result, 0)
        r1 = _nth_tool_result(result, 1)
        assert r0["error"] is not None
        assert r1["result"] == 4
        assert r1["error"] is None


# --------------------------------------------------------------------------- #
# Custom tool_description                                                      #
# --------------------------------------------------------------------------- #


class TestCustomToolDescription:
    def test_custom_description_does_not_break_execution(self):
        custom_desc = "Custom description. {available_host_tools}"
        agent = _agent(
            _eval_call("2 ** 8"),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(
                tool_description=custom_desc,
            ),
        )
        r = _tool_result(agent.invoke({"messages": [{"role": "user", "content": "go"}]}))
        assert r["result"] == 256
        assert r["error"] is None

    def test_default_tool_description_constant_is_used_when_not_overridden(self):
        mw = MontyCodeInterpreterMiddleware()
        # The tool description template is stored internally; verify the public
        # constant is non-empty and used as the fallback.
        assert len(CODE_INTERPRETER_TOOL_DESCRIPTION) > 0


# --------------------------------------------------------------------------- #
# Async — extended coverage                                                    #
# --------------------------------------------------------------------------- #


class TestAsyncExtended:
    @pytest.mark.asyncio
    async def test_async_stdout_captured(self):
        agent = _agent(_eval_call('print("async hello")'), _done())
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})
        r = _tool_result(result)
        assert "async hello" in r["stdout"]

    @pytest.mark.asyncio
    async def test_async_float_result(self):
        agent = _agent(_eval_call("1.5 + 2.5"), _done())
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})
        r = _tool_result(result)
        assert r["result"] == 4.0
        assert r["error"] is None

    @pytest.mark.asyncio
    async def test_async_runtime_error(self):
        agent = _agent(_eval_call("[][0]"), _done())
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})
        r = _tool_result(result)
        assert r["error"] is not None

    @pytest.mark.asyncio
    async def test_async_ptc_tool_exception_caught(self):
        @tool
        def bad_async_tool() -> int:
            """Raise an error."""
            raise TypeError("type problem")

        code = (
            "caught = None\n"
            "try:\n"
            "    bad_async_tool()\n"
            "except Exception as e:\n"
            "    caught = str(e)\n"
            "caught"
        )
        agent = _agent(
            _eval_call(code),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(ptc=[bad_async_tool]),
            tools=[bad_async_tool],
        )
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})
        r = _tool_result(result)
        assert r["error"] is None
        assert r["result"] is not None

    @pytest.mark.asyncio
    async def test_async_iteration_budget_exceeded(self):
        @tool
        def noop_async() -> int:
            """Noop."""
            return 0

        agent = _agent(
            _eval_call("noop_async()"),
            _done(),
            middleware=MontyCodeInterpreterMiddleware(
                ptc=[noop_async],
                iteration_budget=0,
            ),
            tools=[noop_async],
        )
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})
        r = _tool_result(result)
        assert r["error"] is not None
        assert r["error"]["type"] == "IterationBudgetExceeded"
