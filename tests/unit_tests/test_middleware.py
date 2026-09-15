from contextlib import contextmanager
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




@contextmanager
def _mock_pool(first_snapshot=None, *, stdout="", feed_error=None, os_handler=None):
    """Patch the sync worker pool, mirroring Monty's pool → session → feed shape.

    ``with Monty() as pool`` → ``with pool.checkout(...) as session`` →
    ``session.feed_start(code)``. Yields the patched ``Monty`` class so tests
    can assert on checkout behaviour.
    """
    from pydantic_monty import CollectString

    mock_stdout = MagicMock(spec=CollectString)
    mock_stdout.output = stdout

    with (
        patch("langchain_monty.middleware._driver.Monty") as MockMonty,
        patch(
            "langchain_monty.middleware._driver.CollectString",
            return_value=mock_stdout,
        ),
        patch(
            "langchain_monty.middleware._driver.OSAccess",
            return_value=os_handler if os_handler is not None else MagicMock(),
        ),
    ):
        pool = MockMonty.return_value.__enter__.return_value
        session = pool.checkout.return_value.__enter__.return_value
        if feed_error is not None:
            session.feed_start.side_effect = feed_error
        else:
            session.feed_start.return_value = first_snapshot
        yield MockMonty


@contextmanager
def _mock_async_pool(
    first_snapshot=None, *, stdout="", feed_error=None, os_handler=None
):
    """Async twin of ``_mock_pool`` for the ``AsyncMonty`` path."""
    from pydantic_monty import CollectString

    mock_stdout = MagicMock(spec=CollectString)
    mock_stdout.output = stdout

    with (
        patch("langchain_monty.middleware._driver.AsyncMonty") as MockMonty,
        patch(
            "langchain_monty.middleware._driver.CollectString",
            return_value=mock_stdout,
        ),
        patch(
            "langchain_monty.middleware._driver.OSAccess",
            return_value=os_handler if os_handler is not None else MagicMock(),
        ),
    ):
        pool = MockMonty.return_value
        pool.__aenter__ = AsyncMock(return_value=pool)
        pool.__aexit__ = AsyncMock(return_value=False)

        session = MagicMock()
        checkout = pool.checkout.return_value
        checkout.__aenter__ = AsyncMock(return_value=session)
        checkout.__aexit__ = AsyncMock(return_value=False)

        if feed_error is not None:
            session.feed_start = AsyncMock(side_effect=feed_error)
        else:
            session.feed_start = AsyncMock(return_value=first_snapshot)
        yield MockMonty


def _make_function_snap(name, *, call_id=1, args=(), kwargs=None, is_os=False):
    from pydantic_monty import FunctionSnapshot

    snap = MagicMock(spec=FunctionSnapshot)
    snap.function_name = name
    snap.call_id = call_id
    snap.is_os_function = is_os  # spec mock attrs are truthy by default
    snap.args = args
    snap.kwargs = kwargs or {}
    return snap


def _make_async_function_snap(name, *, call_id=1, args=(), kwargs=None, is_os=False):
    from pydantic_monty import AsyncFunctionSnapshot

    snap = MagicMock(spec=AsyncFunctionSnapshot)
    snap.function_name = name
    snap.call_id = call_id
    snap.is_os_function = is_os
    snap.args = args
    snap.kwargs = kwargs or {}
    snap.resume = AsyncMock()
    snap.resume_not_handled = AsyncMock()
    return snap


def _make_async_future_snap(pending_call_ids):
    from pydantic_monty import AsyncFutureSnapshot

    snap = MagicMock(spec=AsyncFutureSnapshot)
    snap.pending_call_ids = list(pending_call_ids)
    snap.resume = AsyncMock()
    return snap



class TestEvalPythonSync:
    def _invoke(self, middleware, code, runtime):
        func = middleware._tool.func
        assert func is not None
        return func(code=code, runtime=runtime)

    def test_simple_completion(self):
        from pydantic_monty import MontyComplete

        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = 42

        with _mock_pool(real_complete, stdout="hi\n"):
            result = self._invoke(m, "42", runtime)

        assert result["result"] == 42
        assert result["stdout"] == "hi\n"
        assert result["error"] is None

    def test_feed_error_returns_structured_error(self):
        """Parse/type-check failures surface from ``feed_start``, not construction."""
        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        with _mock_pool(feed_error=SyntaxError("bad syntax")):
            result = self._invoke(m, "???", runtime)

        assert result["error"]["type"] == "SyntaxError"
        assert result["result"] is None
        assert result["attempted_code"] == "???"

    def test_resource_exhaustion_returns_structured_error(self):
        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        with _mock_pool(feed_error=RuntimeError("out of memory")):
            result = self._invoke(m, "x = 1", runtime)

        assert result["error"]["type"] == "RuntimeError"
        assert result["attempted_code"] == "x = 1"

    def test_pool_startup_failure_returns_structured_error(self):
        """A worker that will not start is reported, not raised at the agent."""
        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        with patch(
            "langchain_monty.middleware._driver.Monty",
            side_effect=RuntimeError("monty binary not found"),
        ):
            result = self._invoke(m, "1", runtime)

        assert result["error"]["type"] == "RuntimeError"
        assert "monty binary not found" in result["error"]["message"]

    def test_host_tool_not_in_allowlist_resumes_with_error(self):
        from pydantic_monty import MontyComplete

        search = _make_base_tool("search")
        m = MontyCodeInterpreterMiddleware(ptc=[search])
        runtime = _make_runtime(tools=[])

        snap = _make_function_snap("forbidden_tool", call_id=1)

        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = None
        snap.resume.return_value = real_complete

        with _mock_pool(snap):
            self._invoke(m, "forbidden_tool()", runtime)

        snap.resume.assert_called_once()
        call_args = snap.resume.call_args[0][0]
        assert call_args["exc_type"] == "RuntimeError"
        assert "allowlist" in call_args["message"]

    def test_iteration_budget_exceeded(self):
        search = _make_base_tool("search", args={"query": {}})
        m = MontyCodeInterpreterMiddleware(ptc=[search], iteration_budget=2)
        runtime = _make_runtime(tools=[search])

        snap = _make_function_snap("search", call_id=1, args=("q",))
        # resume always returns the same snapshot -> unbounded host calls
        snap.resume.return_value = snap

        with _mock_pool(snap):
            result = self._invoke(m, "...", runtime)

        assert result["error"]["type"] == "IterationBudgetExceeded"

    def test_os_snapshot_serviced_by_handler(self):
        """OS calls surface as snapshots and are dispatched host-side.

        Monty only consults the ``os=`` handler from ``resume_auto()``, so a
        manually driven feed must answer OS snapshots itself — a driver that
        bounced them back would break every ``pathlib``/clock call.
        """
        from pydantic_monty import MontyComplete

        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        snap = _make_function_snap(
            "Path.read_text", call_id=1, args=("/data/a.txt",), is_os=True
        )
        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = None
        snap.resume.return_value = real_complete

        os_handler = MagicMock(return_value="hello")

        with _mock_pool(snap, os_handler=os_handler):
            self._invoke(m, "Path('/data/a.txt').read_text()", runtime)

        os_handler.assert_called_once_with("Path.read_text", ("/data/a.txt",), {})
        snap.resume.assert_called_once_with({"return_value": "hello"})
        snap.resume_not_handled.assert_not_called()

    def test_os_snapshot_not_handled_falls_back_to_monty(self):
        """A declining handler hands the call back to Monty's own behaviour."""
        from pydantic_monty import NOT_HANDLED, MontyComplete

        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        snap = _make_function_snap("os.fork", call_id=1, is_os=True)
        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = None
        snap.resume_not_handled.return_value = real_complete

        os_handler = MagicMock(return_value=NOT_HANDLED)

        with _mock_pool(snap, os_handler=os_handler):
            self._invoke(m, "import os", runtime)

        snap.resume_not_handled.assert_called_once_with()
        snap.resume.assert_not_called()

    def test_host_tool_invoked_and_result_returned(self):
        from pydantic_monty import MontyComplete

        search = _make_base_tool("search", args={"query": {}})
        search.invoke.return_value = '["result1"]'
        m = MontyCodeInterpreterMiddleware(ptc=[search])

        runtime = _make_runtime(tools=[])

        snap = _make_function_snap("search", call_id=1, kwargs={"query": "test"})

        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = ["result1"]
        snap.resume.return_value = real_complete

        with _mock_pool(snap):
            result = self._invoke(m, "search(query='test')", runtime)

        assert result["result"] == ["result1"]
        assert result["error"] is None

    def test_host_tool_exception_uses_external_exception(self):
        from pydantic_monty import MontyComplete

        search = _make_base_tool("search", args={"query": {}})
        search.invoke.side_effect = ValueError("bad query")
        m = MontyCodeInterpreterMiddleware(ptc=[search])

        runtime = _make_runtime(tools=[])

        snap = _make_function_snap("search", call_id=1, kwargs={"query": "bad"})

        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = None
        snap.resume.return_value = real_complete

        with _mock_pool(snap):
            self._invoke(m, "search(query='bad')", runtime)

        # Two-pass driver: the first (deferred) resume answers with a future
        # marker; since the mock never awaits, the driver reruns eagerly and
        # the FINAL resume carries the ExternalException payload.
        final_payload = snap.resume.call_args[0][0]
        assert "exception" in final_payload
        assert isinstance(final_payload["exception"], ValueError)

    def test_eager_rerun_uses_a_fresh_session(self):
        """The eager pass must not reuse the deferred pass's dirtied session.

        Session globals persist across feeds, so re-feeding the same session
        would let a half-run snippet's state leak into the rerun.
        """
        from pydantic_monty import MontyComplete

        search = _make_base_tool("search", args={"query": {}})
        search.invoke.return_value = '"ok"'
        m = MontyCodeInterpreterMiddleware(ptc=[search])
        runtime = _make_runtime(tools=[])

        snap = _make_function_snap("search", call_id=1, kwargs={"query": "q"})
        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = None
        snap.resume.return_value = real_complete

        with _mock_pool(snap) as MockMonty:
            self._invoke(m, "search(query='q')", runtime)

        pool = MockMonty.return_value.__enter__.return_value
        assert pool.checkout.call_count == 2

    def test_name_lookup_snapshot_handled(self):
        from pydantic_monty import MontyComplete, NameLookupSnapshot

        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        name_snap = MagicMock(spec=NameLookupSnapshot)
        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = None
        name_snap.resume.return_value = real_complete

        with _mock_pool(name_snap):
            result = self._invoke(m, "unknown_var", runtime)

        # Resumed with no value at all: that is what makes the sandbox raise
        # NameError rather than silently binding something.
        name_snap.resume.assert_called_once_with()
        assert result is not None

    def test_future_snapshot_handled(self):
        from pydantic_monty import FutureSnapshot, MontyComplete

        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        future_snap = MagicMock(spec=FutureSnapshot)
        future_snap.pending_call_ids = [1, 2]
        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = None
        future_snap.resume.return_value = real_complete

        with _mock_pool(future_snap):
            result = self._invoke(m, "await something()", runtime)

        future_snap.resume.assert_called_once()
        call_args = future_snap.resume.call_args[0][0]
        assert 1 in call_args
        assert 2 in call_args
        assert result is not None

    def test_resume_not_called_twice_when_deserialize_fails(self):
        """Regression: if _deserialize_return_value raises, the driver must
        not call resume() twice on the same snapshot (Progress already resumed).
        """
        from pydantic_monty import MontyComplete

        search = _make_base_tool("search", args={"query": {}})
        # invoke succeeds, but _deserialize_return_value will raise
        search.invoke.return_value = object()  # non-serialisable sentinel
        m = MontyCodeInterpreterMiddleware(ptc=[search])
        runtime = _make_runtime(tools=[])

        snap = _make_function_snap("search", call_id=1, kwargs={"query": "test"})

        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = None
        snap.resume.return_value = real_complete

        with (
            _mock_pool(snap),
            patch(
                "langchain_monty.middleware._bridge.deserialize_return_value",
                side_effect=TypeError("not JSON serializable"),
            ),
        ):
            self._invoke(m, "search(query='test')", runtime)

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
    """The async entrypoint drives ``AsyncMonty`` directly rather than wrapping
    the sync pool in a thread, which is why these tests patch ``AsyncMonty``
    and give the snapshots awaitable ``resume`` methods.
    """

    async def _invoke(self, middleware, code, runtime):
        coroutine = middleware._tool.coroutine
        assert coroutine is not None
        return await coroutine(code=code, runtime=runtime)

    @pytest.mark.asyncio
    async def test_feed_error_returns_structured_error(self):
        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        with _mock_async_pool(feed_error=ValueError("parse error")):
            result = await self._invoke(m, "bad code", runtime)

        assert result["error"]["type"] == "ValueError"
        assert result["result"] is None
        assert result["attempted_code"] == "bad code"

    @pytest.mark.asyncio
    async def test_simple_completion(self):
        from pydantic_monty import MontyComplete

        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = "done"

        with _mock_async_pool(real_complete):
            result = await self._invoke(m, '"done"', runtime)

        assert result["result"] == "done"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_os_snapshot_serviced_by_handler(self):
        from pydantic_monty import MontyComplete

        m = MontyCodeInterpreterMiddleware()
        runtime = _make_runtime()

        snap = _make_async_function_snap(
            "Path.read_text", call_id=1, args=("/data/a.txt",), is_os=True
        )
        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = None
        snap.resume.return_value = real_complete

        os_handler = MagicMock(return_value="hello")

        with _mock_async_pool(snap, os_handler=os_handler):
            await self._invoke(m, "Path('/data/a.txt').read_text()", runtime)

        snap.resume.assert_awaited_once_with({"return_value": "hello"})


class TestEvalPythonAsyncGather:
    """Tests for the deferred-FunctionSnapshot + concurrent-FutureSnapshot path.

    The async driver defers each host-tool call (FunctionSnapshot → resume with
    {"future": ...}) and then resolves the whole batch concurrently when
    Monty emits a FutureSnapshot.
    """

    async def _invoke(self, middleware, code, runtime):
        coroutine = middleware._tool.coroutine
        assert coroutine is not None
        return await coroutine(code=code, runtime=runtime)

    @pytest.mark.asyncio
    async def test_function_snapshot_resumes_with_future_payload(self):
        """Resume with {"future": ...}, never {"return_value": ...}."""
        from pydantic_monty import MontyComplete

        search = _make_base_tool("search", args={"query": {}})
        search.ainvoke = AsyncMock(return_value="r")
        m = MontyCodeInterpreterMiddleware(ptc=[search])
        runtime = _make_runtime()

        snap = _make_async_function_snap("search", call_id=7, kwargs={"query": "hi"})
        future_snap = _make_async_future_snap([7])
        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = None

        snap.resume.return_value = future_snap
        future_snap.resume.return_value = real_complete

        with _mock_async_pool(snap):
            await self._invoke(m, "search(query='hi')", runtime)

        snap.resume.assert_awaited_once()
        call_payload = snap.resume.call_args[0][0]
        # Monty's ExternalFuture TypedDict requires the literal Ellipsis as
        # the value — the call is identified by the snapshot's call_id, not
        # by the payload. (Resuming with the call id raises TypeError.)
        assert "future" in call_payload
        assert call_payload["future"] is ...
        assert "return_value" not in call_payload

    @pytest.mark.asyncio
    async def test_future_snapshot_invokes_all_tools_and_passes_results(self):
        """Every deferred call is awaited and forwarded in the FutureSnapshot resume."""
        from pydantic_monty import MontyComplete

        search = _make_base_tool("search", args={"query": {}})
        fetch = _make_base_tool("fetch", args={"url": {}})
        search.ainvoke = AsyncMock(return_value='["hit"]')
        fetch.ainvoke = AsyncMock(return_value='{"ok": true}')
        m = MontyCodeInterpreterMiddleware(ptc=[search, fetch])
        runtime = _make_runtime()

        snap_a = _make_async_function_snap("search", call_id=1, kwargs={"query": "q"})
        snap_b = _make_async_function_snap("fetch", call_id=2, kwargs={"url": "u"})
        future_snap = _make_async_future_snap([1, 2])
        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = "done"

        snap_a.resume.return_value = snap_b
        snap_b.resume.return_value = future_snap
        future_snap.resume.return_value = real_complete

        with _mock_async_pool(snap_a):
            result = await self._invoke(m, "...", runtime)

        search.ainvoke.assert_awaited_once()
        fetch.ainvoke.assert_awaited_once()

        future_snap.resume.assert_awaited_once()
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
        from pydantic_monty import MontyComplete

        search = _make_base_tool("search", args={"query": {}})
        fetch = _make_base_tool("fetch", args={"url": {}})
        search.ainvoke = AsyncMock(return_value="r1")
        fetch.ainvoke = AsyncMock(return_value="r2")

        # Budget of 2: two deferred calls fit exactly.
        m = MontyCodeInterpreterMiddleware(ptc=[search, fetch], iteration_budget=2)
        runtime = _make_runtime()

        snap_a = _make_async_function_snap("search", call_id=1, kwargs={"query": "q"})
        snap_b = _make_async_function_snap("fetch", call_id=2, kwargs={"url": "u"})
        future_snap = _make_async_future_snap([1, 2])
        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = "done"

        snap_a.resume.return_value = snap_b
        snap_b.resume.return_value = future_snap
        future_snap.resume.return_value = real_complete

        with _mock_async_pool(snap_a):
            result = await self._invoke(m, "...", runtime)

        assert result["error"] is None
        assert result["result"] == "done"

    @pytest.mark.asyncio
    async def test_iteration_budget_exceeded_by_fanout(self):
        """A gather fan-out larger than the budget is rejected."""
        search = _make_base_tool("search", args={"query": {}})
        fetch = _make_base_tool("fetch", args={"url": {}})
        search.ainvoke = AsyncMock(return_value="r1")
        fetch.ainvoke = AsyncMock(return_value="r2")

        # Budget of 1: the second deferred call must trip the budget.
        m = MontyCodeInterpreterMiddleware(ptc=[search, fetch], iteration_budget=1)
        runtime = _make_runtime()

        snap_a = _make_async_function_snap("search", call_id=1, kwargs={"query": "q"})
        snap_b = _make_async_function_snap("fetch", call_id=2, kwargs={"url": "u"})
        snap_a.resume.return_value = snap_b

        with _mock_async_pool(snap_a):
            result = await self._invoke(m, "...", runtime)

        assert result["error"]["type"] == "IterationBudgetExceeded"

    @pytest.mark.asyncio
    async def test_future_snapshot_tool_error_surfaced_per_call(self):
        """One failing tool gets an exception payload; the other call is unaffected."""
        from pydantic_monty import MontyComplete

        search = _make_base_tool("search", args={"query": {}})
        fetch = _make_base_tool("fetch", args={"url": {}})
        search.ainvoke = AsyncMock(side_effect=RuntimeError("search down"))
        fetch.ainvoke = AsyncMock(return_value='{"ok": true}')
        m = MontyCodeInterpreterMiddleware(ptc=[search, fetch])
        runtime = _make_runtime()

        snap_a = _make_async_function_snap("search", call_id=1, kwargs={"query": "q"})
        snap_b = _make_async_function_snap("fetch", call_id=2, kwargs={"url": "u"})
        future_snap = _make_async_future_snap([1, 2])
        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = None

        snap_a.resume.return_value = snap_b
        snap_b.resume.return_value = future_snap
        future_snap.resume.return_value = real_complete

        with _mock_async_pool(snap_a):
            await self._invoke(m, "...", runtime)

        future_snap.resume.assert_awaited_once()
        results_arg = future_snap.resume.call_args[0][0]
        assert "exception" in results_arg[1]
        assert isinstance(results_arg[1]["exception"], RuntimeError)
        assert "return_value" in results_arg[2]

    @pytest.mark.asyncio
    async def test_function_snapshot_not_in_allowlist_resumes_with_error_not_future(
        self,
    ):
        """A tool outside the allowlist gets an immediate error, not a deferral."""
        from pydantic_monty import MontyComplete

        search = _make_base_tool("search", args={"query": {}})
        m = MontyCodeInterpreterMiddleware(ptc=[search])
        runtime = _make_runtime()

        snap = _make_async_function_snap("forbidden", call_id=99)
        real_complete = MagicMock(spec=MontyComplete)
        real_complete.output = None
        snap.resume.return_value = real_complete

        with _mock_async_pool(snap):
            await self._invoke(m, "forbidden()", runtime)

        snap.resume.assert_awaited_once()
        call_payload = snap.resume.call_args[0][0]
        assert "future" not in call_payload
        assert call_payload.get("exc_type") == "RuntimeError"
        assert "allowlist" in call_payload["message"]
