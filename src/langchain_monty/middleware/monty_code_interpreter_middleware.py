"""LangChain middleware exposing pydantic-monty to an agent as ``eval_python``."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphInterrupt
from pydantic_monty import (
    CollectString,
    ExternalResult,
    FutureSnapshot,
    Monty,
    MontyComplete,
    MontyError,
    MontyRuntimeError,
    MontySyntaxError,
    MontyTypingError,
    NameLookupSnapshot,
    OSAccess,
    load_snapshot,
)

from langchain_monty.middleware._bridge import (
    invoke_host_tool_async,
    invoke_host_tool_sync,
)
from langchain_monty.helpers import (
    jsonable_output,
    make_exception_result,
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
from langchain_monty.middleware._tool_rendering import (
    _append_to_system_message,
    _format_tool_schema,
    _normalize_call_args,
    _render_type_stubs,
    _resolve_deferred_from_request,
)
from langchain_monty.models import (
    EvalError,
    EvalPythonResult,
    LangchainStoreMontySnapshot,
    MontyLimits,
)

CODE_INTERPRETER_SYSTEM_PROMPT = """## `eval_python` (Monty runtime)

You have access to ``eval_python``, which runs Python code in Pydantic's
Monty interpreter — a Rust-implemented Python subset that starts in
microseconds and has no access to the host's filesystem, network, or
environment. The only way out of the sandbox is calling functions that have
been explicitly exposed to it.

Use it when you need to:

- Compose host tools programmatically (loops, fan-out, conditional dispatch)
- Reshape, filter, score, or deduplicate structured data
- Do math, parsing, or aggregation that would otherwise burn LLM turns
- Handle expected errors locally rather than via another model turn

Don't use it for:

- Anything one direct tool call already handles
- Generating prose, summaries, or final user-facing text
- Code that needs imports outside Monty's small stdlib subset
  (currently: ``sys``, ``os``, ``typing``, ``asyncio``, ``re``, ``datetime``,
  ``json``, ``dataclasses``). No classes yet either.

Host functions support two call styles — pick ONE per snippet, never mix:

- Plain calls: ``hits = search("q")`` — simplest, runs calls one at a time.
- Concurrent: ``await asyncio.gather(search("a"), search("b"))`` inside an
  ``async def`` — use when independent host calls can run in parallel.

Iteration tip: your code is statically type-checked against the host
function signatures before it runs, and Monty surfaces precise error
messages with line numbers. If your first attempt hits an unsupported
feature or a bad argument, the error will tell you exactly what to change.
"""


CODE_INTERPRETER_TOOL_DESCRIPTION = """Execute Python code in the Monty sandbox.

Monty is a minimal Python interpreter (Rust implementation by Pydantic). No
filesystem, no network, no subprocess. The only externally-visible effects
are calls into the host functions listed below; everything else is pure
compute against your inputs.

Host functions available (programmatic tool calling allowlist):
{available_host_tools}

The code is parsed and executed once per call. The value of the final
expression (if any) is returned as ``result``. Exceptions raised inside the
sandbox come back as a structured ``error``, not raised on the host.

Resource limits per call:
- max_duration_secs: {max_duration_secs}
- max_memory_bytes: {max_memory_bytes}
- max_stack_depth: {max_stack_depth}

Example:
    hits = search("LangGraph release notes", max_results=5)
    [h["title"] for h in hits if "0.6" in h["title"]]
"""


class _UnawaitedHostCalls(Exception):
    """Deferred pass ended with never-awaited futures; driver should restart eagerly."""


class MontyCodeInterpreterMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Middleware that adds an ``eval_python`` tool backed by Monty.

    Args:
        ptc: Programmatic tool calling allowlist. ``BaseTool`` entries are
            available immediately; ``str`` entries are deferred and resolved
            at runtime from the agent's bound tools (useful for tools injected
            by other middleware). Tools not in this list are invisible to the
            sandbox. Default ``None`` means pure-compute only.
        limits: Per-call resource budgets.
        system_prompt: Override for the appended system-prompt block. ``None``
            disables the prompt contribution; full schemas fall back to the
            tool description instead.
        tool_description: Override for the ``eval_python`` tool description.
            Placeholders: ``{available_host_tools}``, ``{max_duration_secs}``,
            ``{max_memory_bytes}``, ``{max_stack_depth}``.
        iteration_budget: Hard cap on host-tool calls per ``eval_python`` call.
            Defends against tight loops and unbounded fan-outs. Default 64.
        type_check: When ``True`` (default), code is statically type-checked
            against stub signatures before execution. Set to ``False`` if
            Monty's checker rejects code you need to run.
    """

    def __init__(
        self,
        *,
        ptc: Sequence[BaseTool | str] | None = None,
        limits: MontyLimits | None = None,
        system_prompt: str | None = CODE_INTERPRETER_SYSTEM_PROMPT,
        tool_description: str | None = None,
        iteration_budget: int = 64,
        type_check: bool = True,
    ) -> None:
        super().__init__()

        tool_entries: list[BaseTool] = []
        deferred_names: list[str] = []
        for entry in ptc or ():
            if isinstance(entry, str):
                deferred_names.append(entry)
            else:
                tool_entries.append(entry)

        self._ptc: frozenset[str] = frozenset(
            [t.name for t in tool_entries] + deferred_names
        )
        self._ptc_tools: dict[str, BaseTool] = {t.name: t for t in tool_entries}
        self._deferred_names: frozenset[str] = frozenset(deferred_names)
        self._limits = limits or MontyLimits()
        self._iteration_budget = iteration_budget
        self._type_check = type_check
        self._description_template = (
            tool_description or CODE_INTERPRETER_TOOL_DESCRIPTION
        )
        if system_prompt is not None:
            prompt_parts: list[str] = [system_prompt]
            if self._ptc_tools:
                schemas = "\n\n".join(
                    _format_tool_schema(t)
                    for t in sorted(self._ptc_tools.values(), key=lambda t: t.name)
                )
                prompt_parts.append(
                    "\n\nHost functions exposed to the interpreter:\n\n" + schemas
                )
            if self._deferred_names:
                names_list = ", ".join(f"``{n}``" for n in sorted(self._deferred_names))
                prompt_parts.append(
                    "\n\nAdditional host functions available at runtime "
                    "(injected by other middleware): "
                    + names_list
                    + ".\nCall these the same way as other host functions. "
                    "Their return values are Python objects (dicts/lists), "
                    "not JSON strings."
                )
            if not self._ptc_tools and not self._deferred_names:
                prompt_parts.append(
                    "\n\nNo host functions are exposed; the interpreter is "
                    "pure compute only."
                )
            self.system_prompt: str | None = "".join(prompt_parts)
        else:
            self.system_prompt = None

        self._tool = self._build_eval_python_tool()
        self.tools: Sequence[BaseTool] = [self._tool]

    def _build_eval_python_tool(self) -> BaseTool:
        ptc = self._ptc
        ptc_tools = self._ptc_tools
        limits = self._limits
        budget = self._iteration_budget
        type_check = self._type_check
        description_template = self._description_template
        schemas_in_desc = self.system_prompt is None

        def eval_python(
            code: Annotated[
                str,
                "Python source to execute in the Monty sandbox. The value of "
                "the final expression is returned as ``result``. Stdout is "
                "captured. Exceptions return as a structured ``error``.",
            ],
            runtime: ToolRuntime,
        ) -> dict[str, Any]:
            host_tools = _resolve_host_tools(runtime, ptc_tools, ptc)

            record = _load_hitl_record(runtime)
            if record is not None:
                # Resume from the interrupted host call; skip parsing/type-check.
                try:
                    result = _sync_pass(
                        None,
                        host_tools,
                        runtime,
                        defer=False,
                        resume=record,
                        limits=limits,
                        iteration_budget=budget,
                    )
                except GraphInterrupt:
                    raise  # capture handler already overwrote the record
                except Exception as exc:  # noqa: BLE001
                    result = _error_result(exc, code)
                _delete_hitl_record(runtime)
                return result

            try:
                monty = Monty(
                    code, inputs=[], **_compile_kwargs(host_tools, type_check, ptc)
                )
            except GraphInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                return _error_result(exc, code)
            try:
                return _drive_sync(
                    monty,
                    host_tools=host_tools,
                    runtime=runtime,
                    limits=limits,
                    iteration_budget=budget,
                )
            except GraphInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                return _error_result(exc, code)

        async def aeval_python(
            code: Annotated[
                str,
                "Python source to execute in the Monty sandbox. The value of "
                "the final expression is returned as ``result``. Stdout is "
                "captured. Exceptions return as a structured ``error``.",
            ],
            runtime: ToolRuntime,
        ) -> dict[str, Any]:
            host_tools = _resolve_host_tools(runtime, ptc_tools, ptc)

            record = await _aload_hitl_record(runtime)
            if record is not None:
                try:
                    result = await _async_pass(
                        None,
                        host_tools,
                        runtime,
                        defer=False,
                        resume=record,
                        limits=limits,
                        iteration_budget=budget,
                    )
                except GraphInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001
                    result = _error_result(exc, code)
                await _adelete_hitl_record(runtime)
                return result

            try:
                # acreate parses/type-checks on a worker thread, off the loop.
                monty = await Monty.acreate(
                    code, inputs=[], **_compile_kwargs(host_tools, type_check, ptc)
                )
            except GraphInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                return _error_result(exc, code)
            try:
                return await _drive_async(
                    monty,
                    host_tools=host_tools,
                    runtime=runtime,
                    limits=limits,
                    iteration_budget=budget,
                )
            except GraphInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                return _error_result(exc, code)

        return StructuredTool.from_function(
            name="eval_python",
            func=eval_python,
            coroutine=aeval_python,
            description=_render_description(
                ptc, ptc_tools, limits, schemas_in_desc, description_template
            ),
        )

    def _amend_model_request(
        self, request: ModelRequest[ContextT]
    ) -> ModelRequest[ContextT]:
        """Append the system-prompt block plus any runtime-resolved schemas."""
        if self.system_prompt is None:
            return request

        block = self.system_prompt

        if self._deferred_names:
            resolved = _resolve_deferred_from_request(request, self._deferred_names)
            if resolved:
                schemas = "\n\n".join(
                    _format_tool_schema(t)
                    for t in sorted(resolved, key=lambda t: t.name)
                )
                block += (
                    "\n\nRuntime-resolved host functions (full signatures):\n\n"
                    + schemas
                )

        new_system_message = _append_to_system_message(request.system_message, block)
        return request.override(system_message=new_system_message)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        return handler(self._amend_model_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        # Base class does not delegate async->sync for wrap hooks.
        return await handler(self._amend_model_request(request))


def _format_limit(value: Any) -> Any:
    return "unlimited" if value is None else value


def _render_description(
    ptc: frozenset[str],
    ptc_tools: dict[str, BaseTool],
    limits: MontyLimits,
    schemas_in_description: bool,
    description_template: str,
) -> str:
    allowlisted = sorted(ptc)
    if not allowlisted:
        listing = "(none — interpreter is pure-compute only)"
    elif schemas_in_description:
        parts = [
            _format_tool_schema(ptc_tools[n]) for n in allowlisted if n in ptc_tools
        ]
        deferred = [n for n in allowlisted if n not in ptc_tools]
        if deferred:
            parts.append(
                "Runtime-resolved host functions (injected by other "
                "middleware): " + ", ".join(deferred)
            )
        listing = "\n\n".join(parts)
    else:
        listing = (
            ", ".join(allowlisted) + "\n(full signatures are in the system prompt)"
        )
    return description_template.format(
        available_host_tools=listing,
        max_duration_secs=_format_limit(limits.max_duration_secs),
        max_memory_bytes=_format_limit(limits.max_memory_bytes),
        max_stack_depth=_format_limit(limits.max_stack_depth),
    )


def _resolve_host_tools(
    runtime: ToolRuntime,
    ptc_tools: dict[str, BaseTool],
    ptc: frozenset[str],
) -> dict[str, BaseTool]:
    """Merge construction-time tools with deferred ones resolved from the runtime."""
    resolved_tools = {
        name: tool for name, tool in ptc_tools.items() if name != "eval_python"
    }
    bound = getattr(runtime, "tools", None) or []
    for t in bound:
        if t.name in ptc and t.name != "eval_python" and t.name not in resolved_tools:
            resolved_tools[t.name] = t
    return resolved_tools


def _compile_kwargs(
    host_tools: dict[str, BaseTool],
    type_check_enabled: bool,
    ptc: frozenset[str],
) -> dict[str, Any]:
    if not type_check_enabled:
        return {}
    stubs = _render_type_stubs(host_tools, ptc)
    return {"type_check": True, "type_check_stubs": stubs}


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
        traceback_text = exc.display("concise")
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


def _drive_sync(
    monty: Monty,
    host_tools: dict[str, BaseTool],
    runtime: ToolRuntime,
    *,
    limits: MontyLimits,
    iteration_budget: int,
) -> dict[str, Any]:
    """Deferred pass first; fall back to eager if code never awaited its futures."""
    try:
        return _sync_pass(
            monty,
            host_tools,
            runtime,
            defer=True,
            limits=limits,
            iteration_budget=iteration_budget,
        )
    except _UnawaitedHostCalls:
        return _sync_pass(
            monty,
            host_tools,
            runtime,
            defer=False,
            limits=limits,
            iteration_budget=iteration_budget,
        )


def _sync_pass(
    monty: Monty | None,
    host_tools: dict[str, BaseTool],
    runtime: ToolRuntime,
    *,
    defer: bool,
    resume: LangchainStoreMontySnapshot | None = None,
    limits: MontyLimits,
    iteration_budget: int,
) -> dict[str, Any]:
    """One pass over Monty's start/resume protocol (sync).

    Dispatches on progress type each iteration:
    - ``MontyComplete``: done.
    - ``FunctionSnapshot``: host call. Eager: invoke now. Deferred: record a future.
    - ``FutureSnapshot``: sandbox awaited; run batch sequentially, resume with results.
    - ``NameLookupSnapshot``: unknown name; resume without value → NameError in sandbox.

    When ``resume`` is given, ``monty`` is ignored and the persisted snapshot
    is revived instead of calling ``monty.start()``.

    Mirror of ``_async_pass``; keep the two in sync when changing the protocol.
    """
    stdout = CollectString()
    os_handler = OSAccess()
    deferred: dict[Any, tuple[BaseTool, dict[str, Any]]] = {}
    executed_any = False
    host_calls, stdout_prefix, interrupts_answered, pending_answer = _init_pass_state(
        resume
    )

    try:
        if resume is not None:
            _burn_answered_interrupts(interrupts_answered)
            progress = load_snapshot(
                base64.b64decode(resume.monty_snapshot),
                print_callback=stdout,
            )
        else:
            progress = monty.start(
                inputs={},
                limits=limits.to_monty(),
                print_callback=stdout,
                os=os_handler,
            )
        while not isinstance(progress, MontyComplete):
            if isinstance(progress, NameLookupSnapshot):
                progress = progress.resume(os=os_handler)
                continue

            if isinstance(progress, FutureSnapshot):
                results: dict[int, ExternalResult] = {}
                for cid in progress.pending_call_ids:
                    if cid in deferred:
                        tool, kwargs = deferred.pop(cid)
                        executed_any = True
                        results[cid] = invoke_host_tool_sync(tool, kwargs, runtime)
                    else:
                        results[cid] = make_exception_result(
                            RuntimeError(f"no deferred host call for id {cid}")
                        )
                progress = progress.resume(results, os=os_handler)
                continue

            # FunctionSnapshot from here on.
            if progress.is_os_function:
                # OS calls auto-dispatch via os_handler; if one reaches here
                # (doesn't persist across resume()), bounce it back.
                progress = progress.resume_not_handled(os=os_handler)
                continue

            name = progress.function_name
            tool = host_tools.get(name)
            if tool is None:
                pending_answer = False
                progress = progress.resume(
                    _allowlist_rejection(name, host_tools),
                    os=os_handler,
                )
                continue

            host_calls += 1
            if host_calls > iteration_budget:
                return _iteration_budget_result(iteration_budget)

            tool_kwargs = _normalize_call_args(progress, tool)
            if defer:
                deferred[progress.call_id] = (tool, tool_kwargs)
                progress = progress.resume({"future": ...}, os=os_handler)
            else:
                try:
                    payload = invoke_host_tool_sync(tool, tool_kwargs, runtime)
                except GraphInterrupt:
                    # Dump paused VM (not yet resumed) so LangGraph replay
                    # can continue from here. host_calls - 1: interrupted call
                    # gets re-counted when the resume loop re-invokes it.
                    _persist_hitl_record(
                        runtime,
                        progress,
                        stdout_prefix + (stdout.output or ""),
                        host_calls - 1,
                        interrupts_answered,
                    )
                    raise
                if pending_answer:
                    interrupts_answered += 1
                    pending_answer = False
                progress = progress.resume(payload, os=os_handler)
    except MontyError:
        if defer and deferred and not executed_any:
            # Error may be a symptom of deferral (e.g. "len() of coroutine").
            # No host tool ran, so an eager rerun is safe.
            raise _UnawaitedHostCalls from None
        raise

    return _finish_pass(deferred, executed_any, progress, stdout, stdout_prefix)


async def _drive_async(
    monty: Monty,
    host_tools: dict[str, BaseTool],
    runtime: ToolRuntime,
    *,
    limits: MontyLimits,
    iteration_budget: int,
) -> dict[str, Any]:
    """Async driver: deferred pass, then eager rerun if needed."""
    try:
        return await _async_pass(
            monty,
            host_tools,
            runtime,
            defer=True,
            limits=limits,
            iteration_budget=iteration_budget,
        )
    except _UnawaitedHostCalls:
        return await _async_pass(
            monty,
            host_tools,
            runtime,
            defer=False,
            limits=limits,
            iteration_budget=iteration_budget,
        )


async def _async_pass(
    monty: Monty | None,
    host_tools: dict[str, BaseTool],
    runtime: ToolRuntime,
    *,
    defer: bool,
    resume: LangchainStoreMontySnapshot | None = None,
    limits: MontyLimits,
    iteration_budget: int,
) -> dict[str, Any]:
    """Async twin of ``_sync_pass``.

    Two async-specific behaviours:
    1. ``FutureSnapshot`` batches run concurrently via ``asyncio.gather``.
    2. Blocking Rust VM steps are offloaded to a worker thread via
       ``asyncio.to_thread`` to avoid freezing the event loop.

    Mirror of ``_sync_pass``; keep the two in sync when changing the protocol.
    """
    stdout = CollectString()
    os_handler = OSAccess()
    deferred: dict[Any, tuple[BaseTool, dict[str, Any]]] = {}
    executed_any = False
    host_calls, stdout_prefix, interrupts_answered, pending_answer = _init_pass_state(
        resume
    )

    try:
        if resume is not None:
            _burn_answered_interrupts(interrupts_answered)
            progress = await asyncio.to_thread(
                load_snapshot,
                base64.b64decode(resume.monty_snapshot),
                print_callback=stdout,
            )
        else:
            progress = await asyncio.to_thread(
                monty.start,
                inputs={},
                limits=limits.to_monty(),
                print_callback=stdout,
                os=os_handler,
            )
        while not isinstance(progress, MontyComplete):
            if isinstance(progress, NameLookupSnapshot):
                progress = await asyncio.to_thread(progress.resume, os=os_handler)
                continue

            if isinstance(progress, FutureSnapshot):
                pending = list(progress.pending_call_ids)
                batch = [
                    _run_deferred(cid, *deferred.pop(cid), runtime)
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

                progress = await asyncio.to_thread(
                    progress.resume, results, os=os_handler
                )
                continue

            # FunctionSnapshot from here on.
            if progress.is_os_function:
                progress = await asyncio.to_thread(
                    progress.resume_not_handled, os=os_handler
                )
                continue

            name = progress.function_name
            tool = host_tools.get(name)
            if tool is None:
                pending_answer = False
                progress = await asyncio.to_thread(
                    progress.resume,
                    _allowlist_rejection(name, host_tools),
                    os=os_handler,
                )
                continue

            host_calls += 1
            if host_calls > iteration_budget:
                return _iteration_budget_result(iteration_budget)

            tool_kwargs = _normalize_call_args(progress, tool)
            if defer:
                deferred[progress.call_id] = (tool, tool_kwargs)
                progress = await asyncio.to_thread(
                    progress.resume, {"future": ...}, os=os_handler
                )
            else:
                try:
                    payload = await invoke_host_tool_async(tool, tool_kwargs, runtime)
                except GraphInterrupt:
                    await _apersist_hitl_record(
                        runtime,
                        progress,
                        stdout_prefix + (stdout.output or ""),
                        host_calls - 1,
                        interrupts_answered,
                    )
                    raise
                if pending_answer:
                    interrupts_answered += 1
                    pending_answer = False
                progress = await asyncio.to_thread(
                    progress.resume, payload, os=os_handler
                )
    except MontyError:
        if defer and deferred and not executed_any:
            raise _UnawaitedHostCalls from None
        raise

    return _finish_pass(deferred, executed_any, progress, stdout, stdout_prefix)
