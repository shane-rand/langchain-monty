from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import SystemMessage

from langchain_monty import (
    MontyCodeInterpreterMiddleware,
    MontyLimits,
)


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


class TestEvalPythonSync:
    def _invoke(self, middleware, code, runtime):
        func = middleware._tool.func
        assert func is not None
        return func(code=code, runtime=runtime)

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
        snap.is_os_function = False
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
        snap.is_os_function = False  # spec mock attrs are truthy by default
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
        snap.is_os_function = False
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
        snap.is_os_function = False
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

        # Two-pass driver: the first (deferred) resume answers with a future
        # marker; since the mock never awaits, the driver reruns eagerly and
        # the FINAL resume carries the ExternalException payload.
        final_payload = snap.resume.call_args[0][0]
        assert "exception" in final_payload
        assert isinstance(final_payload["exception"], ValueError)

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
        snap.is_os_function = False
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
                "langchain_monty.middleware._bridge.deserialize_return_value",
                side_effect=TypeError("not JSON serializable"),
            ),
        ):
            instance = MockMonty.return_value
            instance.start.return_value = snap

            result = self._invoke(m, "search(query='test')", runtime)

        # The FINAL resume (eager pass) must carry the exception payload —
        # the driver must not resume the same snapshot a second time with a
        # return_value after the deserialization failure (that would be the
        # "Progress already resumed" double-resume bug this test guards).
        final_payload = snap.resume.call_args[0][0]
        assert "exception" in final_payload
        assert isinstance(final_payload["exception"], TypeError)
        # No call in either pass may have carried a return_value.
        for call in snap.resume.call_args_list:
            assert "return_value" not in call[0][0]


class TestEvalPythonAsync:
    """The async entrypoint builds the interpreter via ``Monty.acreate`` (so
    parsing/type-checking happens off the event loop), which is why these
    tests configure ``MockMonty.acreate`` as an AsyncMock rather than setting
    a side effect on the class call itself.
    """

    async def _invoke(self, middleware, code, runtime):
        coroutine = middleware._tool.coroutine
        assert coroutine is not None
        return await coroutine(code=code, runtime=runtime)

    @pytest.mark.asyncio
    async def test_compile_error_returns_structured_error(self):
        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        with patch(
            "langchain_monty.middleware.monty_code_interpreter_middleware.Monty"
        ) as MockMonty:
            MockMonty.acreate = AsyncMock(side_effect=ValueError("parse error"))
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
            MockMonty.acreate = AsyncMock(return_value=instance)

            result = await self._invoke(m, '"done"', runtime)

        assert result["result"] == "done"
        assert result["error"] is None


class TestEvalPythonAsyncGather:
    """Tests for the deferred-FunctionSnapshot + concurrent-FutureSnapshot path.

    The async driver defers each host-tool call (FunctionSnapshot → resume with
    {"future": call_id}) and then resolves the whole batch concurrently when
    Monty emits a FutureSnapshot.
    """

    async def _invoke(self, middleware, code, runtime):
        coroutine = middleware._tool.coroutine
        assert coroutine is not None
        return await coroutine(code=code, runtime=runtime)

    def _make_function_snap(self, name, call_id, kwargs=None):
        from pydantic_monty import FunctionSnapshot

        snap = MagicMock(spec=FunctionSnapshot)
        snap.function_name = name
        snap.call_id = call_id
        snap.is_os_function = False
        snap.args = ()
        snap.kwargs = kwargs or {}
        return snap

    @pytest.mark.asyncio
    async def test_function_snapshot_resumes_with_future_payload(self):
        """FunctionSnapshot must resume with {"future": call_id}, not {"return_value": ...}."""
        from pydantic_monty import CollectString, FutureSnapshot, MontyComplete

        search = _make_base_tool("search", args={"query": {}})
        search.ainvoke = AsyncMock(return_value="r")
        m = MontyCodeInterpreterMiddleware(ptc=[search])
        runtime = _make_runtime()

        snap = self._make_function_snap("search", call_id=7, kwargs={"query": "hi"})
        future_snap = MagicMock(spec=FutureSnapshot)
        future_snap.pending_call_ids = [7]
        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = None
        mock_stdout = MagicMock(spec=CollectString)
        mock_stdout.output = ""

        snap.resume.return_value = future_snap
        future_snap.resume.return_value = real_complete

        with (
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.Monty"
            ) as MockMonty,
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.CollectString",
                return_value=mock_stdout,
            ),
        ):
            MockMonty.return_value.start.return_value = snap
            MockMonty.acreate = AsyncMock(return_value=MockMonty.return_value)
            await self._invoke(m, "search(query='hi')", runtime)

        snap.resume.assert_called_once()
        call_payload = snap.resume.call_args[0][0]
        # Monty's ExternalFuture TypedDict requires the literal Ellipsis as
        # the value — the call is identified by the snapshot's call_id, not
        # by the payload. (Resuming with the call id raises TypeError.)
        assert "future" in call_payload
        assert call_payload["future"] is ...
        assert "return_value" not in call_payload

    @pytest.mark.asyncio
    async def test_future_snapshot_invokes_all_tools_and_passes_results(self):
        """All deferred calls must be awaited and their results forwarded in the FutureSnapshot resume."""
        from pydantic_monty import CollectString, FutureSnapshot, MontyComplete

        search = _make_base_tool("search", args={"query": {}})
        fetch = _make_base_tool("fetch", args={"url": {}})
        search.ainvoke = AsyncMock(return_value='["hit"]')
        fetch.ainvoke = AsyncMock(return_value='{"ok": true}')
        m = MontyCodeInterpreterMiddleware(ptc=[search, fetch])
        runtime = _make_runtime()

        snap_a = self._make_function_snap("search", call_id=1, kwargs={"query": "q"})
        snap_b = self._make_function_snap("fetch", call_id=2, kwargs={"url": "u"})
        future_snap = MagicMock(spec=FutureSnapshot)
        future_snap.pending_call_ids = [1, 2]
        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = "done"
        mock_stdout = MagicMock(spec=CollectString)
        mock_stdout.output = ""

        snap_a.resume.return_value = snap_b
        snap_b.resume.return_value = future_snap
        future_snap.resume.return_value = real_complete

        with (
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.Monty"
            ) as MockMonty,
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.CollectString",
                return_value=mock_stdout,
            ),
        ):
            MockMonty.return_value.start.return_value = snap_a
            MockMonty.acreate = AsyncMock(return_value=MockMonty.return_value)
            result = await self._invoke(m, "...", runtime)

        search.ainvoke.assert_awaited_once()
        fetch.ainvoke.assert_awaited_once()

        future_snap.resume.assert_called_once()
        results_arg = future_snap.resume.call_args[0][0]
        assert 1 in results_arg and 2 in results_arg
        assert "return_value" in results_arg[1]
        assert "return_value" in results_arg[2]
        assert result["error"] is None
        assert result["result"] == "done"

    @pytest.mark.asyncio
    async def test_iteration_budget_counts_each_host_call(self):
        """iteration_budget caps host-tool CALLS — a gather batch of N costs N.

        (Earlier semantics counted a whole batch as one round-trip, which
        let a single asyncio.gather fan out an unbounded number of host
        calls past the budget.)
        """
        from pydantic_monty import CollectString, FutureSnapshot, MontyComplete

        search = _make_base_tool("search", args={"query": {}})
        fetch = _make_base_tool("fetch", args={"url": {}})
        search.ainvoke = AsyncMock(return_value="r1")
        fetch.ainvoke = AsyncMock(return_value="r2")

        # Budget of 2: two deferred calls fit exactly.
        m = MontyCodeInterpreterMiddleware(ptc=[search, fetch], iteration_budget=2)
        runtime = _make_runtime()

        snap_a = self._make_function_snap("search", call_id=1, kwargs={"query": "q"})
        snap_b = self._make_function_snap("fetch", call_id=2, kwargs={"url": "u"})
        future_snap = MagicMock(spec=FutureSnapshot)
        future_snap.pending_call_ids = [1, 2]
        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = "done"
        mock_stdout = MagicMock(spec=CollectString)
        mock_stdout.output = ""

        snap_a.resume.return_value = snap_b
        snap_b.resume.return_value = future_snap
        future_snap.resume.return_value = real_complete

        with (
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.Monty"
            ) as MockMonty,
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.CollectString",
                return_value=mock_stdout,
            ),
        ):
            MockMonty.return_value.start.return_value = snap_a
            MockMonty.acreate = AsyncMock(return_value=MockMonty.return_value)
            result = await self._invoke(m, "...", runtime)

        assert result["error"] is None
        assert result["result"] == "done"

    @pytest.mark.asyncio
    async def test_iteration_budget_exceeded_by_fanout(self):
        """A gather fan-out larger than the budget is rejected."""
        from pydantic_monty import CollectString

        search = _make_base_tool("search", args={"query": {}})
        fetch = _make_base_tool("fetch", args={"url": {}})
        search.ainvoke = AsyncMock(return_value="r1")
        fetch.ainvoke = AsyncMock(return_value="r2")

        # Budget of 1: the second deferred call must trip the budget.
        m = MontyCodeInterpreterMiddleware(ptc=[search, fetch], iteration_budget=1)
        runtime = _make_runtime()

        snap_a = self._make_function_snap("search", call_id=1, kwargs={"query": "q"})
        snap_b = self._make_function_snap("fetch", call_id=2, kwargs={"url": "u"})
        snap_a.resume.return_value = snap_b
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
            MockMonty.return_value.start.return_value = snap_a
            MockMonty.acreate = AsyncMock(return_value=MockMonty.return_value)
            result = await self._invoke(m, "...", runtime)

        assert result["error"]["type"] == "IterationBudgetExceeded"

    @pytest.mark.asyncio
    async def test_future_snapshot_tool_error_surfaced_per_call(self):
        """A failing tool in the batch gets an exception payload; the other call is unaffected."""
        from pydantic_monty import CollectString, FutureSnapshot, MontyComplete

        search = _make_base_tool("search", args={"query": {}})
        fetch = _make_base_tool("fetch", args={"url": {}})
        search.ainvoke = AsyncMock(side_effect=RuntimeError("search down"))
        fetch.ainvoke = AsyncMock(return_value='{"ok": true}')
        m = MontyCodeInterpreterMiddleware(ptc=[search, fetch])
        runtime = _make_runtime()

        snap_a = self._make_function_snap("search", call_id=1, kwargs={"query": "q"})
        snap_b = self._make_function_snap("fetch", call_id=2, kwargs={"url": "u"})
        future_snap = MagicMock(spec=FutureSnapshot)
        future_snap.pending_call_ids = [1, 2]
        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = None
        mock_stdout = MagicMock(spec=CollectString)
        mock_stdout.output = ""

        snap_a.resume.return_value = snap_b
        snap_b.resume.return_value = future_snap
        future_snap.resume.return_value = real_complete

        with (
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.Monty"
            ) as MockMonty,
            patch(
                "langchain_monty.middleware.monty_code_interpreter_middleware.CollectString",
                return_value=mock_stdout,
            ),
        ):
            MockMonty.return_value.start.return_value = snap_a
            MockMonty.acreate = AsyncMock(return_value=MockMonty.return_value)
            await self._invoke(m, "...", runtime)

        future_snap.resume.assert_called_once()
        results_arg = future_snap.resume.call_args[0][0]
        assert "exception" in results_arg[1]
        assert isinstance(results_arg[1]["exception"], RuntimeError)
        assert "return_value" in results_arg[2]

    @pytest.mark.asyncio
    async def test_function_snapshot_not_in_allowlist_resumes_with_error_not_future(self):
        """A call to a tool not in the allowlist should get an immediate error, not a future deferral."""
        from pydantic_monty import CollectString, MontyComplete

        search = _make_base_tool("search", args={"query": {}})
        m = MontyCodeInterpreterMiddleware(ptc=[search])
        runtime = _make_runtime()

        snap = self._make_function_snap("forbidden", call_id=99, kwargs={})
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
            MockMonty.return_value.start.return_value = snap
            MockMonty.acreate = AsyncMock(return_value=MockMonty.return_value)
            await self._invoke(m, "forbidden()", runtime)

        snap.resume.assert_called_once()
        call_payload = snap.resume.call_args[0][0]
        assert "future" not in call_payload
        assert call_payload.get("exc_type") == "RuntimeError"
        assert "allowlist" in call_payload["message"]
