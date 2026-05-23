import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import SystemMessage

from langchain_monty import (
    MontyCodeInterpreterMiddleware,
    MontyLimits,
    CODE_INTERPRETER_SYSTEM_PROMPT,
)
from langchain_monty.models import EvalError, EvalPythonResult


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _make_request(system_message=None):
    req = MagicMock()
    req.system_message = system_message
    req.override = MagicMock(return_value=req)
    return req


def _make_runtime(tools=None):
    runtime = MagicMock()
    runtime.tools = tools or []
    runtime.config = {}
    return runtime


# --------------------------------------------------------------------------- #
# Initialisation                                                               #
# --------------------------------------------------------------------------- #


class TestInit:
    def test_default_creates_eval_python_tool(self):
        m = MontyCodeInterpreterMiddleware()
        assert len(m.tools) == 1
        assert m.tools[0].name == "eval_python"

    def test_system_prompt_set_by_default(self):
        m = MontyCodeInterpreterMiddleware()
        assert m.system_prompt is not None
        assert "eval_python" in m.system_prompt

    def test_system_prompt_none_disables_prompt(self):
        m = MontyCodeInterpreterMiddleware(system_prompt=None)
        assert m.system_prompt is None

    def test_ptc_list_appends_host_functions_to_prompt(self):
        m = MontyCodeInterpreterMiddleware(ptc=["search", "task"])
        assert m.system_prompt is not None
        assert "search(...)" in m.system_prompt
        assert "task(...)" in m.system_prompt

    def test_empty_ptc_says_pure_compute(self):
        m = MontyCodeInterpreterMiddleware(ptc=[])
        assert "pure compute" in (m.system_prompt or "").lower()

    def test_no_ptc_says_pure_compute(self):
        m = MontyCodeInterpreterMiddleware()
        assert "pure compute" in (m.system_prompt or "").lower()

    def test_custom_limits_stored(self):
        lim = MontyLimits(max_duration_secs=1.0)
        m = MontyCodeInterpreterMiddleware(limits=lim)
        assert m._limits is lim

    def test_default_limits_created(self):
        m = MontyCodeInterpreterMiddleware()
        assert isinstance(m._limits, MontyLimits)

    def test_iteration_budget_stored(self):
        m = MontyCodeInterpreterMiddleware(iteration_budget=10)
        assert m._iteration_budget == 10

    def test_ptc_stored_as_frozenset(self):
        m = MontyCodeInterpreterMiddleware(ptc=["a", "b"])
        assert m._ptc == frozenset({"a", "b"})

    def test_custom_system_prompt_used(self):
        m = MontyCodeInterpreterMiddleware(system_prompt="Custom prompt")
        assert "Custom prompt" in (m.system_prompt or "")


# --------------------------------------------------------------------------- #
# wrap_model_call                                                              #
# --------------------------------------------------------------------------- #


class TestWrapModelCall:
    def test_calls_handler_directly_when_no_system_prompt(self):
        m = MontyCodeInterpreterMiddleware(system_prompt=None)
        request = _make_request()
        handler = MagicMock(return_value="response")
        result = m.wrap_model_call(request, handler)
        handler.assert_called_once_with(request)
        assert result == "response"

    def test_appends_system_prompt_to_request(self):
        m = MontyCodeInterpreterMiddleware()
        request = _make_request(system_message=None)
        handler = MagicMock(return_value="response")
        m.wrap_model_call(request, handler)
        request.override.assert_called_once()
        _, kwargs = request.override.call_args
        new_msg = kwargs["system_message"]
        assert isinstance(new_msg, SystemMessage)

    def test_handler_called_with_overridden_request(self):
        m = MontyCodeInterpreterMiddleware()
        overridden = MagicMock()
        request = _make_request()
        request.override.return_value = overridden
        handler = MagicMock(return_value="response")
        m.wrap_model_call(request, handler)
        handler.assert_called_once_with(overridden)


# --------------------------------------------------------------------------- #
# awrap_model_call                                                             #
# --------------------------------------------------------------------------- #


class TestAwrapModelCall:
    @pytest.mark.asyncio
    async def test_calls_handler_directly_when_no_system_prompt(self):
        m = MontyCodeInterpreterMiddleware(system_prompt=None)
        request = _make_request()
        handler = AsyncMock(return_value="response")
        result = await m.awrap_model_call(request, handler)
        handler.assert_awaited_once_with(request)
        assert result == "response"

    @pytest.mark.asyncio
    async def test_appends_system_prompt_to_request(self):
        m = MontyCodeInterpreterMiddleware()
        request = _make_request(system_message=None)
        overridden = MagicMock()
        request.override.return_value = overridden
        handler = AsyncMock(return_value="response")
        await m.awrap_model_call(request, handler)
        handler.assert_awaited_once_with(overridden)


# --------------------------------------------------------------------------- #
# eval_python tool — sync driver                                               #
# --------------------------------------------------------------------------- #


class TestEvalPythonSync:
    def _invoke(self, middleware, code, runtime):
        return middleware._tool.func(code=code, runtime=runtime)

    def test_simple_completion(self):
        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        complete = MagicMock()
        complete.__class__ = type("MontyComplete", (), {})
        complete.output = 42
        complete.stdout = "hi\n"

        with patch(
            "langchain_monty.middleware.monty_code_interpreter_middleware.Monty"
        ) as MockMonty:
            instance = MockMonty.return_value
            instance.start.return_value = complete

            with patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.MontyComplete",
                complete.__class__,
            ):
                result = self._invoke(m, "42", runtime)

        assert result["result"] == 42
        assert result["error"] is None

    def test_compile_error_returns_structured_error(self):
        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        with patch(
            "langchain_monty.middleware.monty_code_interpreter_middleware.Monty",
            side_effect=SyntaxError("bad syntax"),
        ):
            result = self._invoke(m, "???", runtime)

        assert result["error"]["type"] == "SyntaxError"
        assert result["result"] is None

    def test_resource_exhaustion_returns_structured_error(self):
        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        complete = MagicMock()
        complete.__class__ = type("MontyComplete", (), {})

        with patch(
            "langchain_monty.middleware.monty_code_interpreter_middleware.Monty"
        ) as MockMonty:
            instance = MockMonty.return_value
            instance.start.side_effect = RuntimeError("out of memory")

            with patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.MontyComplete",
                complete.__class__,
            ):
                result = self._invoke(m, "x = 1", runtime)

        assert result["error"]["type"] == "RuntimeError"

    def test_host_tool_not_in_allowlist_resumes_with_error(self):
        m = MontyCodeInterpreterMiddleware(ptc=["search"])
        runtime = _make_runtime(tools=[])

        snap = MagicMock()
        snap.__class__ = type("FunctionSnapshot", (), {})
        snap.function_name = "forbidden_tool"
        snap.args = ()
        snap.kwargs = {}

        complete = MagicMock()
        complete.__class__ = type("MontyComplete", (), {})
        complete.output = None
        complete.stdout = ""

        call_count = 0

        def fake_resume(payload):
            nonlocal call_count
            call_count += 1
            assert "error" in payload
            assert "allowlist" in payload["error"]
            return complete

        snap.resume = fake_resume

        with patch(
            "langchain_monty.middleware.monty_code_interpreter_middleware.Monty"
        ) as MockMonty:
            instance = MockMonty.return_value
            instance.start.return_value = snap

            with (
                patch(
                    "langchain_monty.middleware.monty_code_interpreter_middleware.FunctionSnapshot",
                    snap.__class__,
                ),
                patch(
                    "langchain_monty.middleware.monty_code_interpreter_middleware.MontyComplete",
                    complete.__class__,
                ),
            ):
                result = self._invoke(m, "forbidden_tool()", runtime)

        assert call_count == 1

    def test_iteration_budget_exceeded(self):
        m = MontyCodeInterpreterMiddleware(ptc=["search"], iteration_budget=2)

        search_tool = MagicMock()
        search_tool.name = "search"
        search_tool.args = {"query": {}}
        search_tool.invoke.return_value = "result"

        runtime = _make_runtime(tools=[search_tool])

        snap_class = type("FunctionSnapshot", (), {})

        # Self-referential: resume always returns the same snapshot so the
        # driver loops forever until the iteration budget is hit.
        snap = MagicMock()
        snap.__class__ = snap_class
        snap.function_name = "search"
        snap.args = ("q",)
        snap.kwargs = {}
        snap.resume = MagicMock(return_value=snap)

        with patch(
            "langchain_monty.middleware.monty_code_interpreter_middleware.Monty"
        ) as MockMonty:
            instance = MockMonty.return_value
            instance.start.return_value = snap

            with patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.FunctionSnapshot",
                snap_class,
            ):
                result = self._invoke(m, "...", runtime)

        assert result["error"]["type"] == "IterationBudgetExceeded"


# --------------------------------------------------------------------------- #
# eval_python tool — async driver                                              #
# --------------------------------------------------------------------------- #


class TestEvalPythonAsync:
    async def _invoke(self, middleware, code, runtime):
        return await middleware._tool.coroutine(code=code, runtime=runtime)

    @pytest.mark.asyncio
    async def test_compile_error_returns_structured_error(self):
        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        with patch(
            "langchain_monty.middleware.monty_code_interpreter_middleware.Monty",
            side_effect=ValueError("parse error"),
        ):
            result = await self._invoke(m, "bad code", runtime)

        assert result["error"]["type"] == "ValueError"
        assert result["result"] is None

    @pytest.mark.asyncio
    async def test_simple_completion(self):
        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        complete = MagicMock()
        complete.__class__ = type("MontyComplete", (), {})
        complete.output = "done"
        complete.stdout = ""

        with patch(
            "langchain_monty.middleware.monty_code_interpreter_middleware.Monty"
        ) as MockMonty:
            instance = MockMonty.return_value
            instance.start.return_value = complete

            with patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.MontyComplete",
                complete.__class__,
            ):
                result = await self._invoke(m, '"done"', runtime)

        assert result["result"] == "done"
        assert result["error"] is None
