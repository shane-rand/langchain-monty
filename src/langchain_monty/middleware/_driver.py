"""Drives one ``eval_python`` call over pydantic-monty's feed/resume protocol.

``_MontyDriver`` wraps the Monty VM execution boundary: checking a session out
of a worker pool, stepping it through ``FunctionSnapshot`` / ``FutureSnapshot``
/ ``NameLookupSnapshot`` dispatch, bridging host-tool calls, servicing OS calls,
and persisting/reviving the VM across a ``GraphInterrupt`` (HITL replay).

Execution happens in a ``monty`` worker subprocess owned by a pool. The pool is
opened per ``eval_python`` call and each *pass* checks out its own session:
session globals persist across feeds, so re-feeding a session that a deferred
pass already dirtied would leak state into the eager rerun.

The async path differs in two ways: ``FutureSnapshot`` batches run concurrently
via ``asyncio.gather``, and it drives ``AsyncMonty`` directly rather than
wrapping the sync pool, so no VM step needs a worker thread.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

from langchain.tools import BaseTool, ToolRuntime
from langgraph.errors import GraphInterrupt
from pydantic_monty import (
    NOT_HANDLED,
    AsyncFunctionSnapshot,
    AsyncFutureSnapshot,
    AsyncMonty,
    AsyncMontySession,
    AsyncNameLookupSnapshot,
    CollectString,
    ExternalResult,
    FunctionSnapshot,
    FutureSnapshot,
    Monty,
    MontyComplete,
    MontyError,
    MontyRuntimeError,
    MontySession,
    MontySyntaxError,
    MontyTypingError,
    NameLookupSnapshot,
    OSAccess,
)

from langchain_monty.helpers import (
    jsonable_output,
    make_exception_result,
    make_return_value_result,
)
from langchain_monty.middleware._bridge import (
    invoke_host_tool_async,
    invoke_host_tool_sync,
)
from langchain_monty.middleware._hitl_snapshot import (
    _adelete_hitl_record,
    _aload_hitl_record,
    _apersist_hitl_record,
    _burn_answered_interrupts,
    _delete_hitl_record,
    _load_hitl_record,
    _persist_hitl_record,
)
from langchain_monty.middleware._tool_rendering import _normalize_call_args
from langchain_monty.models import (
    EvalError,
    EvalPythonResult,
    LangchainStoreMontySnapshot,
    MontyLimits,
)


class _UnawaitedHostCalls(Exception):
    """Deferred pass ended with never-awaited futures; driver should restart eagerly."""


class _MontyDriver:
    """Runs one ``eval_python`` call: check out a VM (or revive one) and drive it.

    Args:
        code: Sandbox source for this call. Ignored on the HITL resume path,
            where the VM is revived from a persisted snapshot instead.
        host_tools: Allowlisted host tools, keyed by name.
        runtime: Live ``ToolRuntime`` forwarded into bridged host calls.
        limits: Per-call resource budgets.
        iteration_budget: Hard cap on host-tool calls for this call.
        session_kwargs: Keyword arguments for ``pool.checkout()`` (type-check
            stubs, etc.); produced by the middleware at build time.
    """

    def __init__(
        self,
        *,
        code: str,
        host_tools: dict[str, BaseTool],
        runtime: ToolRuntime,
        limits: MontyLimits,
        iteration_budget: int,
        session_kwargs: dict[str, Any],
    ) -> None:
        self._code = code
        self._host_tools = host_tools
        self._runtime = runtime
        self._limits = limits
        self._budget = iteration_budget
        self._session_kwargs = session_kwargs

    def run_sync(self) -> dict[str, Any]:
        """Entry point for the sync tool: resume a paused call or run a fresh one."""
        record = _load_hitl_record(self._runtime)
        try:
            # One worker for this call; the pool is torn down with the `with`.
            with Monty(min_processes=1, max_processes=1) as pool:
                if record is None:
                    return self._drive_sync(pool)

                # Resume from the interrupted host call; skip parsing/type-check.
                try:
                    result = self._pass_sync(pool, defer=False, resume=record)
                except GraphInterrupt:
                    raise  # capture handler already overwrote the record
                except Exception as exc:  # noqa: BLE001
                    result = _error_result(exc, self._code)
                _delete_hitl_record(self._runtime)
                return result
        except GraphInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            return _error_result(exc, self._code)

    async def run_async(self) -> dict[str, Any]:
        """Async twin of ``run_sync``."""
        record = await _aload_hitl_record(self._runtime)
        try:
            async with AsyncMonty(min_processes=1, max_processes=1) as pool:
                if record is None:
                    return await self._drive_async(pool)

                try:
                    result = await self._pass_async(pool, defer=False, resume=record)
                except GraphInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001
                    result = _error_result(exc, self._code)
                await _adelete_hitl_record(self._runtime)
                return result
        except GraphInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            return _error_result(exc, self._code)

    def _drive_sync(self, pool: Monty) -> dict[str, Any]:
        """Deferred pass first; fall back to eager if code never awaited its futures."""
        try:
            return self._pass_sync(pool, defer=True)
        except _UnawaitedHostCalls:
            return self._pass_sync(pool, defer=False)

    def _pass_sync(
        self,
        pool: Monty,
        *,
        defer: bool,
        resume: LangchainStoreMontySnapshot | None = None,
    ) -> dict[str, Any]:
        """One pass over Monty's feed/resume protocol (sync).

        Dispatches on progress type each iteration:
        - ``MontyComplete``: done.
        - ``NameLookupSnapshot``: unknown name; resume w/o value → sandbox NameError.
        - ``FutureSnapshot``: sandbox awaited; run batch sequentially, resume results.
        - ``FunctionSnapshot``: OS call (serviced host-side) or host call. Eager:
          invoke now; deferred: record a future.

        When ``resume`` is given the persisted snapshot is revived on a fresh
        session instead of feeding ``self._code``.

        Mirror of ``_pass_async``; keep the two in sync when changing the protocol.
        """
        stdout = CollectString()
        os_handler = OSAccess()
        deferred: dict[Any, tuple[BaseTool, dict[str, Any]]] = {}
        executed_any = False
        host_calls, stdout_prefix, interrupts_answered, pending_answer = (
            _init_pass_state(resume)
        )

        # A fresh session per pass: an eager rerun must not see globals left
        # behind by the deferred pass that failed.
        with self._checkout_sync(pool) as session:
            try:
                if resume is not None:
                    _burn_answered_interrupts(interrupts_answered)
                    progress = session.load_snapshot(
                        base64.b64decode(resume.monty_snapshot),
                        print_callback=stdout,
                        os=os_handler,
                    )
                else:
                    progress = session.feed_start(
                        self._code,
                        print_callback=stdout,
                        os=os_handler,
                    )
                while not isinstance(progress, MontyComplete):
                    if isinstance(progress, NameLookupSnapshot):
                        progress = progress.resume()
                        continue

                    if isinstance(progress, FutureSnapshot):
                        results: dict[int, ExternalResult] = {}
                        for cid in progress.pending_call_ids:
                            if cid in deferred:
                                tool, kwargs = deferred.pop(cid)
                                executed_any = True
                                results[cid] = invoke_host_tool_sync(
                                    tool, kwargs, self._runtime
                                )
                            else:
                                results[cid] = make_exception_result(
                                    RuntimeError(f"no deferred host call for id {cid}")
                                )
                        progress = progress.resume(results)
                        continue

                    # FunctionSnapshot from here on.
                    if progress.is_os_function:
                        os_result = _os_result(progress, os_handler)
                        progress = (
                            progress.resume_not_handled()
                            if os_result is None
                            else progress.resume(os_result)
                        )
                        continue

                    name = progress.function_name
                    tool = self._host_tools.get(name)
                    if tool is None:
                        pending_answer = False
                        progress = progress.resume(
                            _allowlist_rejection(name, self._host_tools)
                        )
                        continue

                    host_calls += 1
                    if host_calls > self._budget:
                        return _iteration_budget_result(self._budget)

                    tool_kwargs = _normalize_call_args(progress, tool)
                    if defer:
                        deferred[progress.call_id] = (tool, tool_kwargs)
                        progress = progress.resume({"future": ...})
                    else:
                        try:
                            payload = invoke_host_tool_sync(
                                tool, tool_kwargs, self._runtime
                            )
                        except GraphInterrupt:
                            # Dump paused VM (not yet resumed) so LangGraph replay
                            # can continue from here. host_calls - 1: interrupted
                            # call gets re-counted when the resume loop re-invokes it.
                            _persist_hitl_record(
                                self._runtime,
                                progress,
                                stdout_prefix + (stdout.output or ""),
                                host_calls - 1,
                                interrupts_answered,
                            )
                            raise
                        if pending_answer:
                            interrupts_answered += 1
                            pending_answer = False
                        progress = progress.resume(payload)
            except MontyError:
                if defer and deferred and not executed_any:
                    # Error may be a symptom of deferral (e.g. "len() of coroutine").
                    # No host tool ran, so an eager rerun is safe.
                    raise _UnawaitedHostCalls from None
                raise

            # Inside the session: MontyComplete.output is converted on access,
            # so read it before the worker goes back to the pool.
            return _finish_pass(
                deferred, executed_any, progress, stdout, stdout_prefix
            )

    async def _drive_async(self, pool: AsyncMonty) -> dict[str, Any]:
        """Async driver: deferred pass, then eager rerun if needed."""
        try:
            return await self._pass_async(pool, defer=True)
        except _UnawaitedHostCalls:
            return await self._pass_async(pool, defer=False)

    async def _pass_async(
        self,
        pool: AsyncMonty,
        *,
        defer: bool,
        resume: LangchainStoreMontySnapshot | None = None,
    ) -> dict[str, Any]:
        """Async twin of ``_pass_sync``.

        The one async-specific behaviour is that ``FutureSnapshot`` batches run
        concurrently via ``asyncio.gather``. Everything else mirrors the sync
        pass against ``AsyncMonty``'s awaitable feed/resume surface; keep the
        two in sync when changing the protocol.
        """
        stdout = CollectString()
        os_handler = OSAccess()
        deferred: dict[Any, tuple[BaseTool, dict[str, Any]]] = {}
        executed_any = False
        host_calls, stdout_prefix, interrupts_answered, pending_answer = (
            _init_pass_state(resume)
        )

        async with self._checkout_async(pool) as session:
            try:
                if resume is not None:
                    _burn_answered_interrupts(interrupts_answered)
                    progress = await session.load_snapshot(
                        base64.b64decode(resume.monty_snapshot),
                        print_callback=stdout,
                        os=os_handler,
                    )
                else:
                    progress = await session.feed_start(
                        self._code,
                        print_callback=stdout,
                        os=os_handler,
                    )
                while not isinstance(progress, MontyComplete):
                    if isinstance(progress, AsyncNameLookupSnapshot):
                        progress = await progress.resume()
                        continue

                    if isinstance(progress, AsyncFutureSnapshot):
                        pending = list(progress.pending_call_ids)
                        batch = [
                            _run_deferred(cid, *deferred.pop(cid), self._runtime)
                            for cid in pending
                            if cid in deferred
                        ]
                        # Pre-fill with errors; real results overwrite.
                        results: dict[int, ExternalResult] = {
                            cid: make_exception_result(
                                RuntimeError(f"no deferred host call for id {cid}")
                            )
                            for cid in pending
                        }
                        if batch:
                            executed_any = True
                            for cid, payload in await asyncio.gather(*batch):
                                results[cid] = payload

                        progress = await progress.resume(results)
                        continue

                    # AsyncFunctionSnapshot from here on.
                    if progress.is_os_function:
                        os_result = _os_result(progress, os_handler)
                        progress = await (
                            progress.resume_not_handled()
                            if os_result is None
                            else progress.resume(os_result)
                        )
                        continue

                    name = progress.function_name
                    tool = self._host_tools.get(name)
                    if tool is None:
                        pending_answer = False
                        progress = await progress.resume(
                            _allowlist_rejection(name, self._host_tools)
                        )
                        continue

                    host_calls += 1
                    if host_calls > self._budget:
                        return _iteration_budget_result(self._budget)

                    tool_kwargs = _normalize_call_args(progress, tool)
                    if defer:
                        deferred[progress.call_id] = (tool, tool_kwargs)
                        progress = await progress.resume({"future": ...})
                    else:
                        try:
                            payload = await invoke_host_tool_async(
                                tool, tool_kwargs, self._runtime
                            )
                        except GraphInterrupt:
                            await _apersist_hitl_record(
                                self._runtime,
                                progress,
                                stdout_prefix + (stdout.output or ""),
                                host_calls - 1,
                                interrupts_answered,
                            )
                            raise
                        if pending_answer:
                            interrupts_answered += 1
                            pending_answer = False
                        progress = await progress.resume(payload)
            except MontyError:
                if defer and deferred and not executed_any:
                    raise _UnawaitedHostCalls from None
                raise

            return _finish_pass(
                deferred, executed_any, progress, stdout, stdout_prefix
            )

    def _checkout_sync(self, pool: Monty) -> MontySession:
        return pool.checkout(limits=self._limits.to_monty(), **self._session_kwargs)

    def _checkout_async(self, pool: AsyncMonty) -> AsyncMontySession:
        return pool.checkout(limits=self._limits.to_monty(), **self._session_kwargs)


def _os_result(
    progress: FunctionSnapshot | AsyncFunctionSnapshot,
    os_handler: OSAccess,
) -> ExternalResult | None:
    """Service an OS snapshot host-side.

    Monty only consults the ``os=`` handler from ``resume_auto()``; a manually
    driven feed surfaces every OS call as a snapshot instead, so the driver
    dispatches it itself. A raising handler is reported to the sandbox as an
    exception rather than tearing down the whole call.

    Returns ``None`` when the handler declined (``NOT_HANDLED``), meaning the
    caller should ``resume_not_handled()`` and let Monty apply its own
    unhandled-operation behaviour.
    """
    try:
        value = os_handler(progress.function_name, progress.args, progress.kwargs)
    except Exception as exc:  # noqa: BLE001
        return make_exception_result(exc)
    if value is NOT_HANDLED:
        return None
    return make_return_value_result(value)


def _init_pass_state(
    resume: LangchainStoreMontySnapshot | None,
) -> tuple[int, str, int, bool]:
    """Per-pass accumulators seeded from a resumed snapshot (identical sync/async).

    Returns ``(host_calls, stdout_prefix, interrupts_answered, pending_answer)``.
    """
    if resume is None:
        return 0, "", 0, False
    return (
        resume.host_calls,
        resume.stdout,
        resume.interrupts_answered,
        True,  # first FunctionSnapshot is the re-invoked interrupted call
    )


def _finish_pass(
    deferred: dict[Any, tuple[BaseTool, dict[str, Any]]],
    executed_any: bool,
    progress: MontyComplete,
    stdout: CollectString,
    stdout_prefix: str,
) -> dict[str, Any]:
    """Terminal result of a pass (identical sync/async).

    Raises ``_UnawaitedHostCalls`` to request an eager rerun when futures were
    recorded but never awaited.
    """
    if deferred:
        if executed_any:
            return _mixed_style_result()
        raise _UnawaitedHostCalls
    return _complete_result(progress, stdout, stdout_prefix=stdout_prefix)


async def _run_deferred(
    call_id: int,
    tool: BaseTool,
    kwargs: dict[str, Any],
    runtime: ToolRuntime,
) -> tuple[int, ExternalResult]:
    """Execute one deferred host call, tagging the result with its id.

    GraphInterrupt re-raises.
    """
    try:
        return (
            call_id,
            await invoke_host_tool_async(tool, kwargs, runtime),
        )
    except GraphInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001
        return (call_id, make_exception_result(exc))


def _iteration_budget_result(iteration_budget: int) -> dict[str, Any]:
    return EvalPythonResult(
        error=EvalError(
            type="IterationBudgetExceeded",
            message=(
                f"interpreter made more than {iteration_budget}"
                " host-tool calls in a single eval_python call"
            ),
        )
    ).model_dump()


def _error_result(exc: BaseException, code: str = "") -> dict[str, Any]:
    """Map any exception to the structured error payload."""
    err_type = type(exc).__name__
    message = str(exc)
    traceback_text: str | None = None

    if isinstance(exc, MontyRuntimeError):
        inner = exc.exception()
        err_type = type(inner).__name__
        message = str(inner)
        traceback_text = exc.display("traceback")
    elif isinstance(exc, MontyTypingError):
        err_type = "TypeCheckError"
        message = "static type check failed before execution; no code was run"
        # Rendering format is chosen at checkout (type_check_format="concise").
        traceback_text = exc.display()
    elif isinstance(exc, MontySyntaxError):
        err_type = "SyntaxError"
        message = exc.display("msg")

    return EvalPythonResult(
        error=EvalError(type=err_type, message=message, traceback=traceback_text),
        attempted_code=code or None,
    ).model_dump()


def _complete_result(
    progress: MontyComplete,
    stdout: CollectString,
    *,
    stdout_prefix: str = "",
) -> dict[str, Any]:
    # stdout_prefix carries text printed before a HITL interrupt
    # (not captured in the VM snapshot).
    return EvalPythonResult(
        result=jsonable_output(progress),
        stdout=stdout_prefix + (stdout.output or ""),
    ).model_dump()


def _allowlist_rejection(name: str, host_tools: dict[str, BaseTool]) -> dict:
    return {
        "exc_type": "RuntimeError",
        "message": (
            f"host function {name!r} is not in the "
            f"allowlist; available: {sorted(host_tools)}"
        ),
    }


def _mixed_style_result() -> dict[str, Any]:
    return EvalPythonResult(
        error=EvalError(
            type="UnawaitedHostCallError",
            message=(
                "the code awaited some host-function calls but "
                "discarded others without awaiting them; use ONE "
                "style per snippet — either plain calls "
                "(result = tool(...)) or await/asyncio.gather for "
                "every host call"
            ),
        )
    ).model_dump()
