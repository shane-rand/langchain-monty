"""Unit tests for module-level eval-loop helper functions.

These functions were previously nested inside _build_eval_python_tool.
Now that they are at module level, they can be tested directly without
instantiating MontyCodeInterpreterMiddleware.
"""

from unittest.mock import MagicMock

import pytest

from langchain_monty.middleware._driver import (
    _allowlist_rejection,
    _complete_result,
    _error_result,
    _iteration_budget_result,
    _mixed_style_result,
)
from langchain_monty.middleware.monty_code_interpreter_middleware import (
    _compile_kwargs,
    _format_limit,
    _render_description,
    _resolve_host_tools,
)
from langchain_monty.models import MontyLimits


# ---------------------------------------------------------------------------
# _format_limit
# ---------------------------------------------------------------------------


class TestFormatLimit:
    def test_none_returns_unlimited(self):
        assert _format_limit(None) == "unlimited"

    def test_numeric_value_returned_as_is(self):
        assert _format_limit(5) == 5
        assert _format_limit(3.14) == 3.14

    def test_zero_returned_as_is(self):
        assert _format_limit(0) == 0


# ---------------------------------------------------------------------------
# _iteration_budget_result
# ---------------------------------------------------------------------------


class TestIterationBudgetResult:
    def test_returns_error_dict(self):
        result = _iteration_budget_result(64)
        assert result["error"]["type"] == "IterationBudgetExceeded"

    def test_message_contains_budget_value(self):
        result = _iteration_budget_result(10)
        assert "10" in result["error"]["message"]

    def test_no_result_key(self):
        result = _iteration_budget_result(64)
        assert result.get("result") is None


# ---------------------------------------------------------------------------
# _error_result
# ---------------------------------------------------------------------------


class TestErrorResult:
    def test_generic_exception_uses_class_name(self):
        exc = ValueError("bad value")
        result = _error_result(exc)
        assert result["error"]["type"] == "ValueError"
        assert "bad value" in result["error"]["message"]

    def test_code_stored_when_provided(self):
        exc = ValueError("x")
        result = _error_result(exc, code="print(1)")
        assert result["attempted_code"] == "print(1)"

    def test_no_code_means_none(self):
        result = _error_result(ValueError("x"))
        assert result.get("attempted_code") is None

    def test_monty_runtime_error_unwrapped(self):
        from pydantic_monty import MontyRuntimeError

        inner = ZeroDivisionError("division by zero")
        exc = MagicMock(spec=MontyRuntimeError)
        exc.exception.return_value = inner
        exc.display.return_value = "Traceback..."
        result = _error_result(exc)
        assert result["error"]["type"] == "ZeroDivisionError"
        assert result["error"]["traceback"] == "Traceback..."

    def test_monty_typing_error_classified(self):
        from pydantic_monty import MontyTypingError

        exc = MagicMock(spec=MontyTypingError)
        exc.display.return_value = "type: error at line 1"
        result = _error_result(exc)
        assert result["error"]["type"] == "TypeCheckError"
        assert "static type check" in result["error"]["message"]
        assert result["error"]["traceback"] == "type: error at line 1"

    def test_monty_syntax_error_classified(self):
        from pydantic_monty import MontySyntaxError

        exc = MagicMock(spec=MontySyntaxError)
        exc.display.return_value = "SyntaxError: unexpected EOF"
        result = _error_result(exc)
        assert result["error"]["type"] == "SyntaxError"
        assert result["error"]["message"] == "SyntaxError: unexpected EOF"


# ---------------------------------------------------------------------------
# _complete_result
# ---------------------------------------------------------------------------


class TestCompleteResult:
    def _make_progress(self, output=None):
        progress = MagicMock()
        progress.output = output
        return progress

    def _make_stdout(self, text=""):
        s = MagicMock()
        s.output = text
        return s

    def test_result_and_stdout_present(self):
        stdout = self._make_stdout("hello\n")
        progress = self._make_progress(output=42)
        result = _complete_result(progress, stdout)
        assert result["stdout"] == "hello\n"

    def test_stdout_prefix_prepended(self):
        stdout = self._make_stdout("world")
        progress = self._make_progress()
        result = _complete_result(progress, stdout, stdout_prefix="hello ")
        assert result["stdout"] == "hello world"

    def test_empty_stdout_with_prefix(self):
        stdout = self._make_stdout("")
        progress = self._make_progress()
        result = _complete_result(progress, stdout, stdout_prefix="pre:")
        assert result["stdout"] == "pre:"


# ---------------------------------------------------------------------------
# _allowlist_rejection
# ---------------------------------------------------------------------------


class TestAllowlistRejection:
    def test_returns_exc_type_key(self):
        result = _allowlist_rejection("secret_tool", {})
        assert result["exc_type"] == "RuntimeError"

    def test_message_names_the_tool(self):
        result = _allowlist_rejection("forbidden", {"allowed": MagicMock()})
        assert "forbidden" in result["message"]

    def test_message_lists_available_tools(self):
        host_tools = {"search": MagicMock(), "read_file": MagicMock()}
        result = _allowlist_rejection("forbidden", host_tools)
        assert "search" in result["message"]
        assert "read_file" in result["message"]


# ---------------------------------------------------------------------------
# _mixed_style_result
# ---------------------------------------------------------------------------


class TestMixedStyleResult:
    def test_returns_error_type(self):
        result = _mixed_style_result()
        assert result["error"]["type"] == "UnawaitedHostCallError"

    def test_message_explains_the_issue(self):
        result = _mixed_style_result()
        assert "await" in result["error"]["message"].lower()


# ---------------------------------------------------------------------------
# _resolve_host_tools
# ---------------------------------------------------------------------------


class TestResolveHostTools:
    def _make_runtime(self, tools=None):
        runtime = MagicMock()
        runtime.tools = tools or []
        return runtime

    def _make_tool(self, name):
        t = MagicMock()
        t.name = name
        return t

    def test_returns_pre_supplied_tools(self):
        tool = self._make_tool("search")
        result = _resolve_host_tools(
            self._make_runtime(),
            ptc_tools={"search": tool},
            ptc=frozenset(["search"]),
        )
        assert result["search"] is tool

    def test_eval_python_excluded_from_pre_supplied(self):
        tool = self._make_tool("eval_python")
        result = _resolve_host_tools(
            self._make_runtime(),
            ptc_tools={"eval_python": tool},
            ptc=frozenset(["eval_python"]),
        )
        assert "eval_python" not in result

    def test_deferred_tools_resolved_from_runtime(self):
        runtime_tool = self._make_tool("read_file")
        runtime = self._make_runtime(tools=[runtime_tool])
        result = _resolve_host_tools(
            runtime,
            ptc_tools={},
            ptc=frozenset(["read_file"]),
        )
        assert result["read_file"] is runtime_tool

    def test_runtime_tool_not_in_allowlist_ignored(self):
        runtime_tool = self._make_tool("not_allowed")
        runtime = self._make_runtime(tools=[runtime_tool])
        result = _resolve_host_tools(
            runtime,
            ptc_tools={},
            ptc=frozenset(["allowed_only"]),
        )
        assert "not_allowed" not in result

    def test_eval_python_excluded_from_runtime_tools(self):
        evil = self._make_tool("eval_python")
        runtime = self._make_runtime(tools=[evil])
        result = _resolve_host_tools(
            runtime,
            ptc_tools={},
            ptc=frozenset(["eval_python"]),
        )
        assert "eval_python" not in result

    def test_pre_supplied_takes_priority_over_runtime(self):
        pre = self._make_tool("search")
        runtime_dup = self._make_tool("search")
        runtime = self._make_runtime(tools=[runtime_dup])
        result = _resolve_host_tools(
            runtime,
            ptc_tools={"search": pre},
            ptc=frozenset(["search"]),
        )
        assert result["search"] is pre


# ---------------------------------------------------------------------------
# _compile_kwargs
# ---------------------------------------------------------------------------


class TestCompileKwargs:
    def test_type_check_disabled_returns_empty(self):
        result = _compile_kwargs({}, type_check_enabled=False, ptc=frozenset())
        assert result == {}

    def test_type_check_enabled_returns_stubs(self):
        result = _compile_kwargs({}, type_check_enabled=True, ptc=frozenset())
        assert result.get("type_check") is True
        assert "type_check_stubs" in result


# ---------------------------------------------------------------------------
# _render_description
# ---------------------------------------------------------------------------


class TestRenderDescription:
    DEFAULT_TEMPLATE = (
        "Eval Python. Tools: {available_host_tools}. "
        "Duration: {max_duration_secs}s. "
        "Memory: {max_memory_bytes}B. "
        "Stack: {max_stack_depth}."
    )

    def test_empty_ptc_says_pure_compute(self):
        limits = MontyLimits()
        result = _render_description(
            ptc=frozenset(),
            ptc_tools={},
            limits=limits,
            schemas_in_description=False,
            description_template=self.DEFAULT_TEMPLATE,
        )
        assert "pure-compute" in result

    def test_tool_names_listed_when_not_schemas_in_description(self):
        limits = MontyLimits()
        tool = MagicMock()
        tool.name = "search"
        result = _render_description(
            ptc=frozenset(["search"]),
            ptc_tools={"search": tool},
            limits=limits,
            schemas_in_description=False,
            description_template=self.DEFAULT_TEMPLATE,
        )
        assert "search" in result

    def test_none_limit_shows_unlimited(self):
        limits = MontyLimits(max_duration_secs=None)
        result = _render_description(
            ptc=frozenset(),
            ptc_tools={},
            limits=limits,
            schemas_in_description=False,
            description_template=self.DEFAULT_TEMPLATE,
        )
        assert "unlimited" in result
