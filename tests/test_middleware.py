from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import SystemMessage

from langchain_monty import (
    MontyCodeInterpreterMiddleware,
    MontyLimits,
)

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


def _make_base_tool(name, args=None, description=None):
    """Create a mock BaseTool with the given name and args schema."""
    t = MagicMock()
    t.name = name
    t.args = args or {}
    t.description = description or f"{name} tool description"
    return t


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

    def test_ptc_tools_appends_schemas_to_prompt(self):
        search = _make_base_tool("search")
        task = _make_base_tool("task")
        m = MontyCodeInterpreterMiddleware(ptc=[search, task])
        assert m.system_prompt is not None
        assert "search" in m.system_prompt
        assert "task" in m.system_prompt

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

    def test_ptc_stored_as_frozenset_of_names(self):
        a = _make_base_tool("a")
        b = _make_base_tool("b")
        m = MontyCodeInterpreterMiddleware(ptc=[a, b])
        assert m._ptc == frozenset({"a", "b"})

    def test_ptc_tools_stored_as_dict(self):
        a = _make_base_tool("a")
        m = MontyCodeInterpreterMiddleware(ptc=[a])
        assert m._ptc_tools == {"a": a}

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
# eval_python tool -- sync driver                                              #
# --------------------------------------------------------------------------- #


class TestEvalPythonSync:
    def _invoke(self, middleware, code, runtime):
        return middleware._tool.func(code=code, runtime=runtime)

    def test_simple_completion(self):
        from pydantic_monty import CollectString, MontyComplete

        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = 42

        mock_stdout = MagicMock(spec=CollectString)
        mock_stdout.output = "hi\n"

        with (
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.Monty"
            ) as MockMonty,
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.CollectString",
                return_value=mock_stdout,
            ),
        ):
            instance = MockMonty.return_value
            instance.start.return_value = real_complete

            result = self._invoke(m, "42", runtime)

        assert result["result"] == 42
        assert result["stdout"] == "hi\n"
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
        assert result["attempted_code"] == "???"

    def test_resource_exhaustion_returns_structured_error(self):
        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        with patch(
            "langchain_monty.middleware.monty_code_interpreter_middleware.Monty"
        ) as MockMonty:
            instance = MockMonty.return_value
            instance.start.side_effect = RuntimeError("out of memory")

            result = self._invoke(m, "x = 1", runtime)

        assert result["error"]["type"] == "RuntimeError"
        assert result["attempted_code"] == "x = 1"

    def test_host_tool_not_in_allowlist_resumes_with_error(self):
        from pydantic_monty import CollectString, FunctionSnapshot, MontyComplete

        search = _make_base_tool("search")
        m = MontyCodeInterpreterMiddleware(ptc=[search])
        runtime = _make_runtime(tools=[])

        snap = MagicMock(spec=FunctionSnapshot)
        snap.function_name = "forbidden_tool"
        snap.args = ()
        snap.kwargs = {}

        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = None

        mock_stdout = MagicMock(spec=CollectString)
        mock_stdout.output = ""

        snap.resume.return_value = real_complete

        with (
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.Monty"
            ) as MockMonty,
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.CollectString",
                return_value=mock_stdout,
            ),
        ):
            instance = MockMonty.return_value
            instance.start.return_value = snap

            result = self._invoke(m, "forbidden_tool()", runtime)

        snap.resume.assert_called_once()
        call_args = snap.resume.call_args[0][0]
        assert call_args["exc_type"] == "RuntimeError"
        assert "allowlist" in call_args["message"]

    def test_iteration_budget_exceeded(self):
        from pydantic_monty import CollectString, FunctionSnapshot

        search = _make_base_tool("search", args={"query": {}})
        m = MontyCodeInterpreterMiddleware(ptc=[search], iteration_budget=2)
        runtime = _make_runtime(tools=[search])

        snap = MagicMock(spec=FunctionSnapshot)
        snap.function_name = "search"
        snap.args = ("q",)
        snap.kwargs = {}
        # resume always returns the same snapshot -> infinite loop
        snap.resume.return_value = snap

        mock_stdout = MagicMock(spec=CollectString)
        mock_stdout.output = ""

        with (
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.Monty"
            ) as MockMonty,
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.CollectString",
                return_value=mock_stdout,
            ),
        ):
            instance = MockMonty.return_value
            instance.start.return_value = snap

            result = self._invoke(m, "...", runtime)

        assert result["error"]["type"] == "IterationBudgetExceeded"

    def test_host_tool_invoked_and_result_returned(self):
        from pydantic_monty import CollectString, FunctionSnapshot, MontyComplete

        search = _make_base_tool("search", args={"query": {}})
        search.invoke.return_value = '["result1"]'
        m = MontyCodeInterpreterMiddleware(ptc=[search])

        runtime = _make_runtime(tools=[])

        snap = MagicMock(spec=FunctionSnapshot)
        snap.function_name = "search"
        snap.args = ()
        snap.kwargs = {"query": "test"}

        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = ["result1"]

        snap.resume.return_value = real_complete

        mock_stdout = MagicMock(spec=CollectString)
        mock_stdout.output = ""

        with (
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.Monty"
            ) as MockMonty,
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.CollectString",
                return_value=mock_stdout,
            ),
        ):
            instance = MockMonty.return_value
            instance.start.return_value = snap

            result = self._invoke(m, "search(query='test')", runtime)

        assert result["result"] == ["result1"]
        assert result["error"] is None

    def test_host_tool_exception_uses_external_exception(self):
        from pydantic_monty import CollectString, FunctionSnapshot, MontyComplete

        search = _make_base_tool("search", args={"query": {}})
        search.invoke.side_effect = ValueError("bad query")
        m = MontyCodeInterpreterMiddleware(ptc=[search])

        runtime = _make_runtime(tools=[])

        snap = MagicMock(spec=FunctionSnapshot)
        snap.function_name = "search"
        snap.args = ()
        snap.kwargs = {"query": "bad"}

        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = None

        snap.resume.return_value = real_complete

        mock_stdout = MagicMock(spec=CollectString)
        mock_stdout.output = ""

        with (
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.Monty"
            ) as MockMonty,
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.CollectString",
                return_value=mock_stdout,
            ),
        ):
            instance = MockMonty.return_value
            instance.start.return_value = snap

            result = self._invoke(m, "search(query='bad')", runtime)

        # resume should have been called with ExternalException format
        snap.resume.assert_called_once()
        call_args = snap.resume.call_args[0][0]
        assert "exception" in call_args
        assert isinstance(call_args["exception"], ValueError)

    def test_name_lookup_snapshot_handled(self):
        from pydantic_monty import CollectString, MontyComplete, NameLookupSnapshot

        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        name_snap = MagicMock(spec=NameLookupSnapshot)
        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = None
        name_snap.resume.return_value = real_complete

        mock_stdout = MagicMock(spec=CollectString)
        mock_stdout.output = ""

        with (
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.Monty"
            ) as MockMonty,
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.CollectString",
                return_value=mock_stdout,
            ),
        ):
            instance = MockMonty.return_value
            instance.start.return_value = name_snap

            result = self._invoke(m, "unknown_var", runtime)

        name_snap.resume.assert_called_once()
        assert result is not None

    def test_future_snapshot_handled(self):
        from pydantic_monty import CollectString, FutureSnapshot, MontyComplete

        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        future_snap = MagicMock(spec=FutureSnapshot)
        future_snap.pending_call_ids = [1, 2]
        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = None
        future_snap.resume.return_value = real_complete

        mock_stdout = MagicMock(spec=CollectString)
        mock_stdout.output = ""

        with (
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.Monty"
            ) as MockMonty,
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.CollectString",
                return_value=mock_stdout,
            ),
        ):
            instance = MockMonty.return_value
            instance.start.return_value = future_snap

            result = self._invoke(m, "await something()", runtime)

        future_snap.resume.assert_called_once()
        call_args = future_snap.resume.call_args[0][0]
        assert 1 in call_args
        assert 2 in call_args

    def test_resume_not_called_twice_when_deserialize_fails(self):
        """Regression: if _deserialize_return_value raises, the driver must
        not call resume() twice on the same snapshot (Progress already resumed).
        """
        from pydantic_monty import CollectString, FunctionSnapshot, MontyComplete

        search = _make_base_tool("search", args={"query": {}})
        # invoke succeeds, but _deserialize_return_value will raise
        search.invoke.return_value = object()  # non-serialisable sentinel
        m = MontyCodeInterpreterMiddleware(ptc=[search])
        runtime = _make_runtime(tools=[])

        snap = MagicMock(spec=FunctionSnapshot)
        snap.function_name = "search"
        snap.args = ()
        snap.kwargs = {"query": "test"}

        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = None
        snap.resume.return_value = real_complete

        mock_stdout = MagicMock(spec=CollectString)
        mock_stdout.output = ""

        # Patch _deserialize_return_value to raise
        with (
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.Monty"
            ) as MockMonty,
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.CollectString",
                return_value=mock_stdout,
            ),
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware._deserialize_return_value",
                side_effect=TypeError("not JSON serializable"),
            ),
        ):
            instance = MockMonty.return_value
            instance.start.return_value = snap

            result = self._invoke(m, "search(query='test')", runtime)

        # resume must be called exactly once — with the exception payload,
        # not twice (once with return_value, once with exception).
        snap.resume.assert_called_once()
        call_args = snap.resume.call_args[0][0]
        assert "exception" in call_args
        assert isinstance(call_args["exception"], TypeError)


# --------------------------------------------------------------------------- #
# eval_python tool -- async driver                                             #
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
        assert result["attempted_code"] == "bad code"

    @pytest.mark.asyncio
    async def test_simple_completion(self):
        from pydantic_monty import CollectString, MontyComplete

        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = "done"

        mock_stdout = MagicMock(spec=CollectString)
        mock_stdout.output = ""

        with (
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.Monty"
            ) as MockMonty,
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.CollectString",
                return_value=mock_stdout,
            ),
        ):
            instance = MockMonty.return_value
            instance.start.return_value = real_complete

            result = await self._invoke(m, '"done"', runtime)

        assert result["result"] == "done"
        assert result["error"] is None
