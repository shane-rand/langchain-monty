"""Middleware exposing a sandboxed Python interpreter (pydantic-monty) to an agent.

``pydantic-monty`` is a tiny Python-subset VM written in Rust by Pydantic
(it will back code-mode in Pydantic AI). This module wires it into LangChain's
v1 agent middleware system as an ``eval_python`` tool.

How the pieces fit together
===========================

1.  **Middleware shape** — ``MontyCodeInterpreterMiddleware`` is a standard
    ``AgentMiddleware`` subclass that contributes one tool (``self.tools``)
    and appends an instruction block to the system prompt via
    ``wrap_model_call`` / ``awrap_model_call``. There is no mutable instance
    state; everything per-call flows through ``ToolRuntime``.

2.  **The host-call bridge** — Monty has no network/filesystem access. The
    only way sandbox code reaches the outside world is by calling a function
    name we allowlisted (the ``ptc`` parameter). When sandbox code calls such
    a function, Monty *pauses* and hands the host a ``FunctionSnapshot``
    describing the call. We look up the matching LangChain ``BaseTool``,
    invoke it through its normal machinery as a full ``ToolCall`` (so
    tracing, retries and injected parameters all work), then ``resume()``
    Monty with the result. The drive loops (``_drive_sync`` /
    ``_drive_async``) implement exactly this start/pause/resume protocol,
    using an adaptive two-pass strategy so that both plain calls
    (``x = search("q")``) and awaited concurrency
    (``await asyncio.gather(...)``) work in both drivers — see the long
    comment above the drivers for the full reasoning.

3.  **Static type checking before execution** — Monty ships a static type
    checker. Because we know the JSON schema of every allowlisted tool, we
    can render Python *stub signatures* for them
    (``def search(query: str, max_results: int = ...) -> Any: ...``) and ask
    Monty to type-check the LLM's code against those stubs *before running
    anything*. Hallucinated keyword arguments, wrong argument types and
    misspelled tool names are caught instantly with precise line/column
    diagnostics — no execution and no wasted host-tool calls. Controlled by
    the ``type_check`` constructor flag (on by default).

4.  **Human-in-the-loop / interrupts** — if a host tool raises
    ``GraphInterrupt`` (e.g. ``HumanInTheLoopMiddleware`` asking for
    approval), we *re-raise it* instead of feeding it into the sandbox.
    LangGraph then checkpoints the graph and pauses. What happens when the
    graph resumes depends on whether a LangGraph store is configured:

    *With a store* (``runtime.store`` is set), the paused Monty VM is
    serialized at interrupt time (``FunctionSnapshot.dump()``) into the
    store, keyed by the eval_python tool call id. When LangGraph replays
    the tool call, the snapshot is revived (``pydantic_monty
    .load_snapshot()``) and execution *continues from the interrupted host
    call*: host tools that already ran are NOT re-invoked, stdout is
    preserved, and the interrupted tool is re-invoked once — its internal
    ``interrupt()`` now returns the recorded human answer. The driver-side
    state the snapshot can't carry (stdout text, budget counter, interrupt
    bookkeeping) travels in ``LangchainStoreMontySnapshot``; see the
    resume path in ``_sync_pass`` / ``_async_pass``.

    *Without a store*, the original replay model applies: the whole tool
    call re-runs from the top, so host tools called before the interrupt
    point are re-invoked and should be idempotent when combined with HITL.

    Scope and limitations of snapshot-resume: it covers the eager
    (plain-call) execution path; interrupts escaping an awaited
    ``asyncio.gather`` batch fall back to full replay. A single host tool
    that calls ``interrupt()`` more than once per invocation desyncs the
    positional interrupt-counter bookkeeping (see
    ``_burn_answered_interrupts``); one interrupt per tool call — the
    ``HumanInTheLoopMiddleware`` shape — is fully supported.

Why we keep a hand-rolled drive loop instead of ``Monty.run_async()``
=====================================================================

Monty's high-level ``run()`` / ``run_async()`` accept an
``external_functions=`` dict and implement dispatch internally. We
deliberately do NOT use them, because the start/resume loop gives us three
things the high-level API can't:

* interception of ``GraphInterrupt`` *before* it would enter the sandbox
  (an interrupt must pause the graph, not become a catchable sandbox
  exception);
* the per-call iteration budget (defence against a tight loop of host
  calls);
* allowlist enforcement with a structured, sandbox-visible error message.
"""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
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
from langchain_core.messages import ContentBlock, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

# GraphInterrupt is how LangGraph implements interrupts (HITL approval,
# breakpoints). It subclasses Exception, so a naive ``except Exception``
# around a tool call would swallow it — we must special-case it everywhere
# we catch broadly. Command is the "tool wants to mutate graph state" return
# type; it cannot be honoured from inside a nested interpreter (see
# _extract_tool_output).
from langgraph.errors import GraphInterrupt
from langgraph.types import Command, interrupt
from pydantic_monty import (
    CollectString,
    ExternalResult,
    FunctionSnapshot,
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

from langchain_monty.models import (
    EvalError,
    EvalPythonResult,
    LangchainStoreMontySnapshot,
    MontyLimits,
)

# --------------------------------------------------------------------------- #
# Prompt blocks                                                               #
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Internal control flow                                                       #
# --------------------------------------------------------------------------- #


class _UnawaitedHostCalls(Exception):
    """Internal signal: the deferred pass ended with never-awaited host calls.

    Raised by ``_sync_pass`` / ``_async_pass`` (deferred mode) when the
    sandbox code turned out to use plain-call style — it called host
    functions but never awaited the futures we handed back. Because the
    deferred pass executes no host tools until the sandbox awaits, nothing
    has run yet and the driver can safely restart the same ``Monty``
    instance in eager mode. Never escapes the drivers.
    """


# --------------------------------------------------------------------------- #
# Middleware                                                                  #
# --------------------------------------------------------------------------- #


class MontyCodeInterpreterMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Middleware that adds an ``eval_python`` tool backed by Monty.

    Works with any LangChain v1 agent (``create_agent``) and with deepagents
    (``create_deep_agent``) — nothing here is deepagents-specific.

    Args:
        ptc: *Programmatic tool calling* allowlist. Each entry is either a
            ``BaseTool`` instance or a ``str`` tool name.

            ``BaseTool`` entries are available immediately — their schemas
            appear in the system prompt at construction time.

            ``str`` entries are *deferred*: they register the name in the
            allowlist and are resolved at runtime from the tools bound to
            the agent. This is useful for tools injected by other middleware
            (e.g. deepagents' ``FilesystemMiddleware`` contributes ``ls``,
            ``read_file``, ``write_file``, ``edit_file``, ``glob``,
            ``grep``). Their full schemas are rendered *dynamically* into
            the system prompt on each model call once they can be resolved
            from the request's tool list (see ``wrap_model_call``).

            Tools not in this list are invisible to the sandbox even if they
            are bound to the agent. Default ``None`` means pure-compute
            only — no host tools exposed.
        limits: ``MontyLimits`` controlling per-call resource budgets.
        system_prompt: Override for the appended system-prompt block. Pass
            ``None`` to disable the prompt contribution entirely; in that
            case the full host-function schemas are embedded in the
            ``eval_python`` tool description instead, so the model still
            sees the signatures somewhere.
        tool_description: Override for the ``eval_python`` tool description.
            Placeholders: ``{available_host_tools}``, ``{max_duration_secs}``,
            ``{max_memory_bytes}``, ``{max_stack_depth}``.
        iteration_budget: Hard cap on host-tool *calls* per ``eval_python``
            call (each allowlisted call counts one, whether made plainly or
            inside an ``asyncio.gather`` fan-out). Defends against an agent
            that writes a tight loop calling a host tool forever, and
            against unbounded concurrent fan-outs. Default 64.
        type_check: When ``True`` (default), the submitted code is statically
            type-checked by Monty against stub signatures generated from the
            allowlisted tools' schemas *before* execution. Type errors come
            back as a structured ``error`` with precise line/column
            diagnostics and nothing is executed. Set to ``False`` if Monty's
            checker (a strict subset of Python's type system) rejects code
            you need to run.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain_monty import MontyCodeInterpreterMiddleware

        # BaseTool instances — schemas shown in prompt immediately
        agent = create_agent(
            model="anthropic:claude-sonnet-4-6",
            middleware=[MontyCodeInterpreterMiddleware(ptc=[search_tool])],
        )

        # Mix of BaseTool and str — str names resolved at runtime from
        # tools injected by other middleware (e.g. FilesystemMiddleware)
        from deepagents import create_deep_agent

        agent = create_deep_agent(
            model="anthropic:claude-sonnet-4-6",
            middleware=[
                MontyCodeInterpreterMiddleware(
                    ptc=[my_api_tool, "read_file", "ls", "grep"],
                ),
            ],
        )
        ```
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

        # Partition ptc entries into BaseTool instances (immediate) and str
        # names (deferred — resolved at runtime from the agent's bound tools).
        tool_entries: list[BaseTool] = []
        deferred_names: list[str] = []
        for entry in ptc or ():
            if isinstance(entry, str):
                deferred_names.append(entry)
            else:
                tool_entries.append(entry)

        # _ptc is the *complete* allowlist (names only); _ptc_tools holds the
        # tool objects we already have; _deferred_names is the subset we can
        # only resolve once the agent is running and gives us its tool list.
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

        # ------------------------------------------------------------------
        # System-prompt block (static part).
        #
        # Full tool schemas live in the system prompt ONLY — the tool
        # description just lists names. Rendering the same schema text into
        # both places (an earlier design) doubled the token cost of every
        # model call for zero benefit.
        #
        # The block built here contains everything known at construction
        # time: the usage guidance plus the schemas of immediately-supplied
        # tools. Schemas for *deferred* names can't be known yet; they are
        # appended per-request in wrap_model_call once the names resolve
        # against the request's tool list.
        # ------------------------------------------------------------------
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

    # --------------------------------------------------------------- tool

    def _build_eval_python_tool(self) -> BaseTool:
        """Build the ``eval_python`` StructuredTool.

        Closes over ``self._ptc``, ``self._limits``, ``self._type_check``
        and ``self._iteration_budget``. The Monty interpreter object itself
        is constructed fresh per invocation so two concurrent ``eval_python``
        calls on different threads cannot share state.

        Schema-visibility contract: the only parameter the LLM sees on the
        ``eval_python`` tool is ``code: str``. The ``runtime: ToolRuntime``
        parameter is auto-injected by LangChain and stripped from the
        generated schema — that's how LangChain treats any ``ToolRuntime``-
        typed argument since the v1 tools refactor. We rely on that here
        rather than building a custom ``args_schema``.
        """
        ptc = self._ptc
        ptc_tools = self._ptc_tools
        limits = self._limits
        iteration_budget = self._iteration_budget
        type_check_enabled = self._type_check
        description_template = self._description_template
        # When the system prompt is disabled the schemas have no other home,
        # so they fall back into the tool description (see __init__ notes).
        schemas_in_description = self.system_prompt is None

        def _render_description() -> str:
            """Render the static tool description.

            By default this lists host-function *names* only and points at
            the system prompt for full signatures — the schemas themselves
            live in the prompt to avoid paying for them twice per model
            call. Resource-limit placeholders show "unlimited" for any limit
            the user disabled by setting it to None.
            """
            allowlisted = sorted(ptc)
            if not allowlisted:
                listing = "(none — interpreter is pure-compute only)"
            elif schemas_in_description:
                # No system prompt block exists, so this is the only place
                # the model can learn the signatures — render them fully.
                parts = [
                    _format_tool_schema(ptc_tools[n])
                    for n in allowlisted
                    if n in ptc_tools
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
                    ", ".join(allowlisted)
                    + "\n(full signatures are in the system prompt)"
                )

            def _fmt_limit(value: Any) -> Any:
                return "unlimited" if value is None else value

            return description_template.format(
                available_host_tools=listing,
                max_duration_secs=_fmt_limit(limits.max_duration_secs),
                max_memory_bytes=_fmt_limit(limits.max_memory_bytes),
                max_stack_depth=_fmt_limit(limits.max_stack_depth),
            )

        def _resolve_host_tools(runtime: ToolRuntime) -> dict[str, BaseTool]:
            """Pick the host tools the interpreter is allowed to call.

            Uses tool objects stored at construction time (from ``ptc``),
            supplemented by any allowlisted tools found on the runtime that
            weren't supplied directly (this is how deferred ``str`` names
            resolve). ``eval_python`` is always excluded to prevent the
            sandbox from invoking itself recursively through the bridge.
            """
            out = {
                name: tool for name, tool in ptc_tools.items() if name != "eval_python"
            }
            bound = getattr(runtime, "tools", None) or []
            for t in bound:
                if t.name in ptc and t.name != "eval_python" and t.name not in out:
                    out[t.name] = t
            return out

        def _compile_kwargs(host_tools: dict[str, BaseTool]) -> dict[str, Any]:
            """Keyword args for ``Monty(...)`` / ``Monty.acreate(...)``.

            When type checking is enabled we generate Python stub signatures
            from the resolved tools' JSON schemas and hand them to Monty as
            ``type_check_stubs``. Monty then validates the LLM's code against
            the *actual* host-function signatures before executing anything:
            a hallucinated kwarg or a wrong argument type comes back as a
            precise ``main.py:line:col`` diagnostic instead of a runtime
            failure halfway through a run that already burned host calls.
            """
            if not type_check_enabled:
                return {}
            stubs = _render_type_stubs(host_tools, ptc)
            return {"type_check": True, "type_check_stubs": stubs}

        # ---------------- structured-result constructors ------------------
        # Every exit path of eval_python funnels through one of these so the
        # LLM always sees the same JSON shape: {result, stdout, error, ...}.

        def _iteration_budget_result() -> dict[str, Any]:
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
            """Map any exception to the structured error payload.

            Monty's own exception wrappers carry much better diagnostics
            than ``str(exc)``, so we unwrap them:

            - ``MontyRuntimeError`` wraps the *real* sandbox exception
              (e.g. ZeroDivisionError) and can render a full traceback with
              line numbers — the LLM gets ``type="ZeroDivisionError"`` plus
              the formatted traceback, exactly like CPython would print.
            - ``MontyTypingError`` (pre-execution static check) renders
              per-line diagnostics via ``display("concise")``.
            - ``MontySyntaxError`` reports the parse failure location.

            Anything else (host-side bugs) falls back to class name +
            message so the failure is at least visible to the agent rather
            than crashing the run.
            """
            err_type = type(exc).__name__
            message = str(exc)
            traceback_text: str | None = None

            if isinstance(exc, MontyRuntimeError):
                # .exception() returns the underlying Python exception the
                # sandbox raised; its class name is what the agent should
                # branch on ("ZeroDivisionError", "KeyError", ...).
                inner = exc.exception()
                err_type = type(inner).__name__
                message = str(inner)
                traceback_text = exc.display("traceback")
            elif isinstance(exc, MontyTypingError):
                err_type = "TypeCheckError"
                message = (
                    "static type check failed before execution; "
                    "no code was run"
                )
                traceback_text = exc.display("concise")
            elif isinstance(exc, MontySyntaxError):
                err_type = "SyntaxError"
                message = exc.display("msg")

            return EvalPythonResult(
                error=EvalError(
                    type=err_type, message=message, traceback=traceback_text
                ),
                attempted_code=code or None,
            ).model_dump()

        def _complete_result(
            progress: MontyComplete,
            stdout: CollectString,
            *,
            stdout_prefix: str = "",
        ) -> dict[str, Any]:
            # stdout_prefix carries text printed before a HITL interrupt:
            # the print_callback is host-side state, not part of the VM
            # snapshot, so a resumed run starts with a fresh collector and
            # the earlier output travels in the snapshot record instead.
            return EvalPythonResult(
                result=_jsonable_output(progress),
                stdout=stdout_prefix + (stdout.output or ""),
            ).model_dump()

        # --------------------------- drivers -------------------------------
        #
        # Both drivers run a two-pass "adaptive call style" strategy, because
        # Monty's pause/resume protocol forces the host to choose, AT CALL
        # TIME, between two incompatible answers to a host-function call:
        #
        #   eager    resume with the concrete return value. Plain calls
        #            (``x = search("q")``) work; but sandbox code that
        #            *awaits* (``await asyncio.gather(...)``) breaks,
        #            because gather() receives plain values instead of
        #            awaitables.
        #   deferred resume with a future marker ({"future": ...}). Awaited
        #            code works (and gather batches run concurrently); but
        #            plain calls silently evaluate to an un-awaited
        #            coroutine object — corrupting results instead of
        #            erroring.
        #
        # The snapshot carries no "is the sandbox awaiting this?" signal, so
        # we can't pick per call. Instead:
        #
        #   PASS 1 (deferred): nothing executes until the sandbox awaits, so
        #          this pass is side-effect-free for code that never awaits.
        #   detect: if the pass ends (complete OR sandbox error) with
        #          deferred calls that were never awaited and NO host tool
        #          has executed yet, the code is plain-call style — rerun in
        #   PASS 2 (eager): host calls resolve inline. Cheap: Monty
        #          instances are restartable without re-parsing, and pass 1
        #          executed no host tools, so no side effects are duplicated.
        #
        # Mixed-style code (awaited some calls, discarded others) is the one
        # case we refuse: a rerun WOULD duplicate the already-executed host
        # calls, so the agent gets a structured error telling it to pick one
        # style. Both drivers implement the same strategy, so a given
        # snippet behaves identically under invoke and ainvoke; the only
        # async-driver difference is that awaited batches run concurrently.

        def _allowlist_rejection(name: str, host_tools: dict[str, BaseTool]) -> dict:
            """Resume payload for a call to a non-allowlisted function.

            Uses the by-name exception form (allowed for builtin exception
            names) so the sandbox sees an ordinary RuntimeError naming the
            available functions — the agent can self-correct instead of
            guessing.
            """
            return {
                "exc_type": "RuntimeError",
                "message": (
                    f"host function {name!r} is not in the "
                    f"allowlist; available: {sorted(host_tools)}"
                ),
            }

        def _mixed_style_result() -> dict[str, Any]:
            """Structured error for mixed awaited/plain host-call styles."""
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

        def _drive_sync(
            monty: Monty,
            host_tools: dict[str, BaseTool],
            runtime: ToolRuntime,
        ) -> dict[str, Any]:
            """Synchronous driver: deferred pass, then eager rerun if needed."""
            try:
                return _sync_pass(monty, host_tools, runtime, defer=True)
            except _UnawaitedHostCalls:
                return _sync_pass(monty, host_tools, runtime, defer=False)

        def _sync_pass(
            monty: Monty | None,
            host_tools: dict[str, BaseTool],
            runtime: ToolRuntime,
            *,
            defer: bool,
            resume: LangchainStoreMontySnapshot | None = None,
        ) -> dict[str, Any]:
            """One pass over Monty's start/resume protocol (sync).

            Monty pauses every time interpreter code needs something from
            the host. ``start()`` and each subsequent ``resume()`` return
            one of four progress states; the loop dispatches on the type:

            - ``MontyComplete`` — finished; ``.output`` is the final value.
            - ``FunctionSnapshot`` — paused on a host-function call. Eager
              pass: invoke the LangChain tool now and resume with the value
              (or the exception, which surfaces inside the sandbox as a
              normal Python exception the code can ``try/except``).
              Deferred pass: record the call, resume with a future marker.
            - ``NameLookupSnapshot`` — sandbox referenced an unknown bare
              name. We resume without a value, which makes Monty raise a
              regular ``NameError`` inside the sandbox.
            - ``FutureSnapshot`` — the sandbox awaited its deferred calls;
              execute them (sequentially here; the async driver's twin runs
              them concurrently) and resume with per-call results.

            ``GraphInterrupt`` raised by a host tool propagates out of this
            loop on purpose — but the eager branch first dumps the paused
            VM into the store so the inevitable LangGraph replay can
            continue from the pause point. See the module docstring's HITL
            section.

            When ``resume`` is given (LangGraph replaying an interrupted
            call), ``monty`` is ignored (may be ``None``): the persisted
            ``FunctionSnapshot`` is revived in place of ``monty.start()``.
            The loop's normal FunctionSnapshot dispatch then re-invokes the
            interrupted call — the snapshot carries its own
            ``function_name``/``args``/``kwargs`` — so no special re-invoke
            step exists. Resume runs are always eager (``defer=False``):
            only the eager path produces snapshots.
            """
            stdout = CollectString()
            os_handler = OSAccess()
            # call_id -> (tool, kwargs) for deferred-but-not-yet-awaited
            # calls; executed_any flips once any host tool actually ran
            # (which is what makes an eager rerun unsafe).
            deferred: dict[Any, tuple[BaseTool, dict[str, Any]]] = {}
            executed_any = False
            # Counts host-function calls (eager invocations + deferrals),
            # enforcing iteration_budget against runaway tool-call loops and
            # oversized fan-outs alike. Resume runs re-count the interrupted
            # call, so the persisted value excludes it.
            host_calls = resume.host_calls if resume is not None else 0
            # Text printed before the interrupt (resume runs only) — the
            # print_callback is not part of the VM snapshot.
            stdout_prefix = resume.stdout if resume is not None else ""
            # interrupt() answers consumed so far within this logical tool
            # call; new captures persist this so the NEXT resume can burn
            # the positional counter forward (_burn_answered_interrupts).
            interrupts_answered = (
                resume.interrupts_answered if resume is not None else 0
            )
            # On a resume run, the first FunctionSnapshot dispatched is the
            # re-invoked interrupted call; completing it consumes one
            # recorded human answer, which the bookkeeping must observe.
            pending_answer = resume is not None

            try:
                if resume is not None:
                    # Advance LangGraph's positional interrupt counter past
                    # the answers consumed before this snapshot was taken;
                    # the re-invoked call's interrupt() then lands on ITS
                    # recorded answer rather than an earlier one.
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
                        # No external name providers; resuming with no value
                        # makes Monty raise NameError inside the sandbox.
                        progress = progress.resume(os=os_handler)
                        continue

                    if isinstance(progress, FutureSnapshot):
                        # The sandbox awaited. Execute every deferred call in
                        # the batch — sequentially, this is the sync path —
                        # and resume with a per-call result map. Ids Monty
                        # asks about that we never deferred (shouldn't
                        # happen) get an explicit error payload rather than
                        # a silent hang.
                        results: dict[int, ExternalResult] = {}
                        for cid in progress.pending_call_ids:
                            if cid in deferred:
                                tool, kwargs = deferred.pop(cid)
                                executed_any = True
                                results[cid] = _invoke_host_tool_sync(
                                    tool, kwargs, runtime
                                )
                            else:
                                results[cid] = _make_exception_result(
                                    RuntimeError(
                                        f"no deferred host call for id {cid}"
                                    )
                                )
                        progress = progress.resume(results, os=os_handler)
                        continue

                    # From here on: FunctionSnapshot.
                    if progress.is_os_function:
                        # OS calls (datetime.now, Path.exists, ...) are
                        # handled by the os_handler passed to start(); they
                        # auto-dispatch and shouldn't reach this loop. If one
                        # does (auto-dispatch does not persist across
                        # resume() calls), fall back to Monty's default
                        # unhandled behaviour.
                        progress = progress.resume_not_handled(os=os_handler)
                        continue

                    name = progress.function_name
                    tool = host_tools.get(name)
                    if tool is None:
                        # Rejected immediately (never deferred) so the error
                        # points at the exact call site. If this is the
                        # re-invoked interrupted call of a resume run (the
                        # tool vanished between pause and resume), its
                        # recorded answer was never consumed — don't count
                        # it as answered.
                        pending_answer = False
                        progress = progress.resume(
                            _allowlist_rejection(name, host_tools),
                            os=os_handler,
                        )
                        continue

                    host_calls += 1
                    if host_calls > iteration_budget:
                        return _iteration_budget_result()

                    tool_kwargs = _normalize_call_args(progress, tool)
                    if defer:
                        # ExternalFuture payload is literally {"future": ...}
                        # (Ellipsis): Monty tracks WHICH call via the
                        # snapshot's own call_id; the host only signals
                        # "result comes later".
                        deferred[progress.call_id] = (tool, tool_kwargs)
                        progress = progress.resume(
                            {"future": ...}, os=os_handler
                        )
                    else:
                        # Eager: GraphInterrupt must NOT be converted into a
                        # sandbox exception — it has to bubble up so
                        # LangGraph pauses the graph for human input.
                        # _invoke_host_tool_sync re-raises it; every other
                        # exception comes back as a payload the sandbox code
                        # can try/except.
                        try:
                            payload = _invoke_host_tool_sync(
                                tool, tool_kwargs, runtime
                            )
                        except GraphInterrupt:
                            # Dump the paused VM (progress has NOT been
                            # resumed, so dump() is legal) so the LangGraph
                            # replay can continue from here instead of
                            # re-running every host call. host_calls - 1:
                            # the interrupted call gets re-counted when the
                            # resumed loop re-invokes it.
                            _persist_hitl_record(
                                runtime,
                                progress,
                                stdout_prefix + (stdout.output or ""),
                                host_calls - 1,
                                interrupts_answered,
                            )
                            raise
                        if pending_answer:
                            # The re-invoked interrupted call completed,
                            # consuming its recorded human answer.
                            interrupts_answered += 1
                            pending_answer = False
                        progress = progress.resume(payload, os=os_handler)
            except MontyError:
                if defer and deferred and not executed_any:
                    # The failure may well be a *symptom* of the deferral
                    # (e.g. "len() of coroutine" because a plain-style call
                    # got a future). No host tool ran, so an eager rerun is
                    # safe and will either succeed or reproduce the genuine
                    # error faithfully.
                    raise _UnawaitedHostCalls from None
                raise

            if deferred:
                if executed_any:
                    return _mixed_style_result()  # rerun would duplicate calls
                raise _UnawaitedHostCalls  # plain-call style: rerun eagerly
            return _complete_result(progress, stdout, stdout_prefix=stdout_prefix)

        async def _drive_async(
            monty: Monty,
            host_tools: dict[str, BaseTool],
            runtime: ToolRuntime,
        ) -> dict[str, Any]:
            """Async driver: deferred pass, then eager rerun if needed."""
            try:
                return await _async_pass(monty, host_tools, runtime, defer=True)
            except _UnawaitedHostCalls:
                return await _async_pass(monty, host_tools, runtime, defer=False)

        async def _async_pass(
            monty: Monty | None,
            host_tools: dict[str, BaseTool],
            runtime: ToolRuntime,
            *,
            defer: bool,
            resume: LangchainStoreMontySnapshot | None = None,
        ) -> dict[str, Any]:
            """One pass over the protocol (async twin of ``_sync_pass``).

            Two async-specific behaviours on top of the sync pass:

            1.  **Concurrent batches.** In the deferred pass, when the
                sandbox awaits (``FutureSnapshot``), the whole batch of
                pending host calls is resolved with ``asyncio.gather`` — so
                sandbox code like
                ``await asyncio.gather(search("a"), search("b"))`` really
                does run both host tools at the same time.

            2.  **Event-loop hygiene.** ``monty.start()`` and
                ``snapshot.resume()`` are blocking Rust calls (they release
                the GIL but block the *calling thread* — potentially for the
                full ``max_duration_secs`` on compute-heavy code). Running
                them directly on the event loop would freeze every other
                coroutine in the process, so each VM step is offloaded to a
                worker thread via ``asyncio.to_thread``. This mirrors what
                Monty's own ``run_async()`` does internally with
                spawn_blocking.
            """
            stdout = CollectString()
            os_handler = OSAccess()
            deferred: dict[Any, tuple[BaseTool, dict[str, Any]]] = {}
            executed_any = False
            # See _sync_pass for what each of these resume-seeded values
            # means; the bookkeeping is identical in both drivers.
            host_calls = resume.host_calls if resume is not None else 0
            stdout_prefix = resume.stdout if resume is not None else ""
            interrupts_answered = (
                resume.interrupts_answered if resume is not None else 0
            )
            pending_answer = resume is not None

            async def _run_deferred(
                call_id: int, tool: BaseTool, kwargs: dict[str, Any]
            ) -> tuple[int, ExternalResult]:
                """Execute one deferred host call; tag the result with its id.

                Tool failures are captured per-call so one failing tool in a
                gather batch doesn't poison its siblings — each call id gets
                its own return_value or exception payload. GraphInterrupt is
                the exception to the rule: it re-raises (out through
                asyncio.gather) so the graph can pause for human input.
                """
                try:
                    return (
                        call_id,
                        await _invoke_host_tool_async(tool, kwargs, runtime),
                    )
                except GraphInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001 — surfaced in-sandbox
                    return (call_id, _make_exception_result(exc))

            try:
                if resume is not None:
                    # See _sync_pass: burn the positional interrupt counter
                    # forward, then revive the paused VM in place of
                    # monty.start(). Deserialization is a blocking Rust
                    # call, so it's offloaded like every other VM step.
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
                        # Same as sync: no value -> NameError in the sandbox.
                        progress = await asyncio.to_thread(
                            progress.resume, os=os_handler
                        )
                        continue

                    if isinstance(progress, FutureSnapshot):
                        pending = list(progress.pending_call_ids)
                        batch = [
                            _run_deferred(cid, *deferred.pop(cid))
                            for cid in pending
                            if cid in deferred
                        ]
                        # Pre-fill every pending id with an error payload so
                        # an id we never deferred (shouldn't happen) fails
                        # loudly instead of hanging; real results overwrite.
                        results: dict[int, ExternalResult] = {
                            cid: _make_exception_result(
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

                    # From here on: FunctionSnapshot.
                    if progress.is_os_function:
                        # See _sync_pass for why OS calls are bounced back.
                        progress = await asyncio.to_thread(
                            progress.resume_not_handled, os=os_handler
                        )
                        continue

                    name = progress.function_name
                    tool = host_tools.get(name)
                    if tool is None:
                        # See _sync_pass for the pending_answer edge here.
                        pending_answer = False
                        progress = await asyncio.to_thread(
                            progress.resume,
                            _allowlist_rejection(name, host_tools),
                            os=os_handler,
                        )
                        continue

                    host_calls += 1
                    if host_calls > iteration_budget:
                        return _iteration_budget_result()

                    tool_kwargs = _normalize_call_args(progress, tool)
                    if defer:
                        deferred[progress.call_id] = (tool, tool_kwargs)
                        progress = await asyncio.to_thread(
                            progress.resume, {"future": ...}, os=os_handler
                        )
                    else:
                        try:
                            payload = await _invoke_host_tool_async(
                                tool, tool_kwargs, runtime
                            )
                        except GraphInterrupt:
                            # See _sync_pass: dump the (un-resumed) paused
                            # VM so the replay continues from here.
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
                    raise _UnawaitedHostCalls from None  # see _sync_pass
                raise

            if deferred:
                if executed_any:
                    return _mixed_style_result()  # rerun would duplicate calls
                raise _UnawaitedHostCalls  # plain-call style: rerun eagerly
            return _complete_result(progress, stdout, stdout_prefix=stdout_prefix)

        # --------------------- the tool entrypoints ------------------------

        def eval_python(
            code: Annotated[
                str,
                "Python source to execute in the Monty sandbox. The value of "
                "the final expression is returned as ``result``. Stdout is "
                "captured. Exceptions return as a structured ``error``.",
            ],
            runtime: ToolRuntime,
        ) -> dict[str, Any]:
            host_tools = _resolve_host_tools(runtime)

            record = _load_hitl_record(runtime)
            if record is not None:
                # LangGraph is replaying this tool call after a HITL
                # interrupt and a snapshot of the paused VM exists:
                # continue from the interrupted host call instead of
                # re-running the code (and every host tool before the
                # pause) from the top. Parsing and type-checking are
                # skipped — the code already ran up to the pause point.
                # Resume runs go straight to one eager pass: the two-pass
                # style detection only makes sense from a cold start, and
                # snapshots are only ever taken in eager mode.
                try:
                    result = _sync_pass(
                        None, host_tools, runtime, defer=False, resume=record
                    )
                except GraphInterrupt:
                    # A later host call interrupted again; the capture
                    # handler already overwrote the record. Keep it.
                    raise
                except MontyError as exc:
                    result = _error_result(exc, code)
                except Exception as exc:  # noqa: BLE001 — host-side failure
                    result = _error_result(exc, code)
                # Success or structured error, the call is finished —
                # don't leave an orphaned snapshot behind.
                _delete_hitl_record(runtime)
                return result

            try:
                # Parse + (optionally) static-type-check in one shot. Syntax
                # and typing failures mean nothing was executed.
                monty = Monty(code, inputs=[], **_compile_kwargs(host_tools))
            except GraphInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 — parse/type-check error
                return _error_result(exc, code)
            try:
                return _drive_sync(monty, host_tools=host_tools, runtime=runtime)
            except GraphInterrupt:
                # Pause-the-graph signal from a host tool (e.g. HITL
                # approval). Must NOT become a structured error — LangGraph
                # catches it, checkpoints, and replays this tool call after
                # the human responds.
                raise
            except MontyError as exc:
                # Sandbox-side failure (runtime error, resource exhaustion).
                return _error_result(exc, code)
            except Exception as exc:  # noqa: BLE001 — host-side failure
                # Bridge/host bugs also come back structured rather than
                # crashing the agent run; the class name makes them
                # distinguishable from sandbox errors in traces.
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
            host_tools = _resolve_host_tools(runtime)

            record = await _aload_hitl_record(runtime)
            if record is not None:
                # See eval_python: continue from the interrupted host call
                # instead of replaying the whole run.
                try:
                    result = await _async_pass(
                        None, host_tools, runtime, defer=False, resume=record
                    )
                except GraphInterrupt:
                    raise  # record already overwritten by the capture handler
                except MontyError as exc:
                    result = _error_result(exc, code)
                except Exception as exc:  # noqa: BLE001 — host-side failure
                    result = _error_result(exc, code)
                await _adelete_hitl_record(runtime)
                return result

            try:
                # Monty.acreate parses (and type-checks) on a worker thread
                # so a large source / stub set never blocks the event loop.
                monty = await Monty.acreate(
                    code, inputs=[], **_compile_kwargs(host_tools)
                )
            except GraphInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 — parse/type-check error
                return _error_result(exc, code)
            try:
                return await _drive_async(
                    monty, host_tools=host_tools, runtime=runtime
                )
            except GraphInterrupt:
                raise  # see eval_python: pause-the-graph, not an error
            except MontyError as exc:
                return _error_result(exc, code)
            except Exception as exc:  # noqa: BLE001 — host-side failure
                return _error_result(exc, code)

        return StructuredTool.from_function(
            name="eval_python",
            func=eval_python,
            coroutine=aeval_python,
            description=_render_description(),
        )

    # ---------------------------------------------------------- prompt hooks

    def _amend_model_request(
        self, request: ModelRequest[ContextT]
    ) -> ModelRequest[ContextT]:
        """Append this middleware's system-prompt block to the request.

        The block has a static part (built once in ``__init__``: guidance +
        schemas of immediately-supplied tools) and a dynamic part rendered
        here on every model call: full schemas for *deferred* tool names
        that have, by now, been resolved against the tools bound to this
        request. This is what makes ``ptc=["read_file", ...]`` entries show
        real signatures to the model even though the tool objects only
        exist after other middleware ran.

        Shared by the sync and async hooks so the two can't drift apart.
        """
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
        # The base class does not delegate async->sync for wrap hooks, so we
        # implement both; the actual logic lives in _amend_model_request.
        return await handler(self._amend_model_request(request))


# --------------------------------------------------------------------------- #
# Host-tool invocation bridge                                                 #
# --------------------------------------------------------------------------- #
#
# LangChain tools come in two flavours of input:
#
#   tool.invoke({"query": "x"})                      # bare-args dict
#   tool.invoke({"name": ..., "args": ..., "id": ..., "type": "tool_call"})
#
# Only the second (a full ToolCall) makes LangChain populate injected
# parameters like InjectedToolCallId, and only explicit values in args
# satisfy a `runtime: ToolRuntime` / InjectedState parameter. Tools from
# deepagents' FilesystemMiddleware (read_file, ls, ...) REQUIRE a runtime to
# reach graph state, so invoking them with a bare dict fails validation.
#
# The bridge therefore always builds a full ToolCall and forwards the live
# ToolRuntime (and its state/store) into any injected slot the tool
# declares. The sandbox can never forge these values: interpreter-supplied
# kwargs matching injected names are stripped in _normalize_call_args BEFORE
# this code adds the real ones.


def _injected_args_for(tool: BaseTool, runtime: ToolRuntime) -> dict[str, Any]:
    """Values for the tool's injected parameters, sourced from the live runtime.

    Detection is two-layered because LangChain marks injection two ways:

    1. ``tool._injected_args_keys`` — populated for ``ToolRuntime``-typed
       parameters on tools built since the v1 refactor.
    2. Schema difference — the tool's *full* input schema minus its *visible*
       args (``tool.args``) yields names hidden from the model, which is
       exactly the injected set for annotation-style markers
       (InjectedState, InjectedStore, ...).

    Both probes are wrapped defensively: mocks and dict-schema tools won't
    have these attributes, and a tool we can't introspect simply gets no
    injected values (matching the old behaviour).
    """
    names: set[str] = set()

    keys = getattr(tool, "_injected_args_keys", None)
    if isinstance(keys, (set, frozenset)):
        names |= set(keys)

    try:
        full = set(tool.get_input_schema().model_fields)
        visible = set(tool.args or {})
        # Only trust the diff for names we KNOW are injection points;
        # anything else hidden from the schema is the tool author's business.
        names |= (full - visible) & _INJECTED_PARAM_NAMES
    except Exception:  # noqa: BLE001 — introspection is best-effort
        pass

    out: dict[str, Any] = {}
    if "runtime" in names:
        out["runtime"] = runtime
    if "state" in names:
        out["state"] = getattr(runtime, "state", None)
    if "store" in names:
        out["store"] = getattr(runtime, "store", None)
    # NOTE: tool_call_id is intentionally absent — LangChain fills it from
    # the ToolCall's "id" field automatically when we invoke with a full
    # ToolCall (see _build_tool_call).
    return out


def _build_tool_call(
    tool: BaseTool, kwargs: dict[str, Any], runtime: ToolRuntime
) -> dict[str, Any]:
    """Assemble the full ToolCall dict for one bridged host call.

    The synthetic id marks the call as originating from inside eval_python
    (handy when reading traces) and doubles as the value LangChain injects
    into InjectedToolCallId parameters.
    """
    return {
        "name": tool.name,
        "args": {**kwargs, **_injected_args_for(tool, runtime)},
        "id": f"eval_python:{uuid.uuid4().hex[:12]}",
        "type": "tool_call",
    }


def _extract_tool_output(raw: Any) -> Any:
    """Convert whatever the tool returned into a value Monty can resume with.

    Invoking with a full ToolCall makes LangChain wrap results in a
    ToolMessage, so unwrap that first. Three cases:

    - ToolMessage with status="error": the tool's error handler converted an
      exception to a message; surface it inside the sandbox as an exception
      (raise here; the caller converts to an exception payload).
    - ToolMessage with an artifact (content_and_artifact tools): the
      artifact is the machine-readable half — that's what sandbox code
      wants. The content half is the human/LLM-readable summary.
    - Plain return values (or ToolMessage content): strings are JSON-decoded
      when possible (_deserialize_return_value) so sandbox code indexes into
      lists/dicts rather than strings.

    Command returns cannot work here: a Command mutates *graph* state and is
    applied by the enclosing ToolNode — but we are already inside a tool, so
    there is no node to apply it. Raising makes the limitation visible in
    the sandbox instead of handing the code an opaque object.
    """
    if isinstance(raw, Command):
        raise RuntimeError(
            "host tool returned a Command (graph-state update); Command-"
            "returning tools cannot be called from inside eval_python — "
            "call this tool directly instead"
        )
    if isinstance(raw, ToolMessage):
        text = raw.content if isinstance(raw.content, str) else raw.text
        if raw.status == "error":
            raise RuntimeError(text)
        if raw.artifact is not None:
            return raw.artifact
        return _deserialize_return_value(text)
    return _deserialize_return_value(raw)


def _invoke_host_tool_sync(
    tool: BaseTool, kwargs: dict[str, Any], runtime: ToolRuntime
) -> ExternalResult:
    """Run one host tool synchronously; map the outcome to a resume payload.

    GraphInterrupt re-raises (graph must pause); every other exception
    becomes an in-sandbox exception the interpreter code can try/except.
    The try block covers result extraction too, so a failure while decoding
    the return value can't leave the snapshot half-resumed.
    """
    try:
        raw = tool.invoke(
            _build_tool_call(tool, kwargs, runtime),
            config=getattr(runtime, "config", None) or None,
        )
        return _make_return_value_result(_extract_tool_output(raw))
    except GraphInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 — surfaced in-sandbox
        return _make_exception_result(exc)


async def _invoke_host_tool_async(
    tool: BaseTool, kwargs: dict[str, Any], runtime: ToolRuntime
) -> ExternalResult:
    """Async twin of _invoke_host_tool_sync (used by the gather batches)."""
    try:
        raw = await tool.ainvoke(
            _build_tool_call(tool, kwargs, runtime),
            config=getattr(runtime, "config", None) or None,
        )
        return _make_return_value_result(_extract_tool_output(raw))
    except GraphInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 — surfaced in-sandbox
        return _make_exception_result(exc)


# --------------------------------------------------------------------------- #
# HITL snapshot persistence                                                   #
# --------------------------------------------------------------------------- #
#
# When a host tool raises GraphInterrupt mid-run, LangGraph checkpoints and
# later REPLAYS the whole eval_python tool call. These helpers persist the
# paused Monty VM (FunctionSnapshot.dump()) plus the driver-side sidecar
# state into the agent's BaseStore so the replay can continue from the pause
# point instead of re-running every host call — see the module docstring's
# HITL section and LangchainStoreMontySnapshot for the record shape.
#
# Every helper degrades silently: no store, a broken record, or a failing
# backend must never break eval_python — the worst acceptable outcome is the
# original full-replay behaviour. In particular a persistence failure at
# capture time must never mask the GraphInterrupt itself, which LangGraph
# needs in order to pause the graph.


def _snapshot_namespace(runtime: ToolRuntime) -> tuple[str, str, str]:
    """Store namespace for HITL snapshots, scoped per conversation thread.

    Must be byte-identical between the capture (put) and resume (get/delete)
    sides — a mismatch degrades silently to full replay, so all three go
    through this one helper.
    """
    config = getattr(runtime, "config", None) or {}
    configurable = config.get("configurable") or {}
    thread_id = configurable.get("thread_id")
    return (
        "langchain_monty",
        "snapshots",
        str(thread_id) if thread_id is not None else "default",
    )


def _hitl_store_key(runtime: ToolRuntime) -> str | None:
    """The store key for this eval_python call, or None if unusable.

    ``tool_call_id`` is the one identity LangGraph keeps stable across the
    interrupt/replay cycle — the replayed task re-runs the SAME tool call —
    which is what makes it the join key between the two executions.
    """
    tool_call_id = getattr(runtime, "tool_call_id", None)
    return tool_call_id if isinstance(tool_call_id, str) and tool_call_id else None


def _make_hitl_record(
    progress: FunctionSnapshot,
    stdout_text: str,
    host_calls: int,
    interrupts_answered: int,
) -> LangchainStoreMontySnapshot:
    """Build the store record for one paused VM (see the model's docstring)."""
    return LangchainStoreMontySnapshot(
        monty_snapshot=base64.b64encode(progress.dump()).decode("ascii"),
        stdout=stdout_text,
        host_calls=host_calls,
        interrupts_answered=interrupts_answered,
    )


def _persist_hitl_record(
    runtime: ToolRuntime,
    progress: FunctionSnapshot,
    stdout_text: str,
    host_calls: int,
    interrupts_answered: int,
) -> None:
    """Dump the paused VM into the store; failures degrade to full replay."""
    store = getattr(runtime, "store", None)
    key = _hitl_store_key(runtime)
    if store is None or key is None:
        return
    try:
        record = _make_hitl_record(
            progress, stdout_text, host_calls, interrupts_answered
        )
        store.put(_snapshot_namespace(runtime), key, record.model_dump())
    except Exception:  # noqa: BLE001 — see section comment
        pass


async def _apersist_hitl_record(
    runtime: ToolRuntime,
    progress: FunctionSnapshot,
    stdout_text: str,
    host_calls: int,
    interrupts_answered: int,
) -> None:
    """Async twin of _persist_hitl_record."""
    store = getattr(runtime, "store", None)
    key = _hitl_store_key(runtime)
    if store is None or key is None:
        return
    try:
        record = _make_hitl_record(
            progress, stdout_text, host_calls, interrupts_answered
        )
        await store.aput(_snapshot_namespace(runtime), key, record.model_dump())
    except Exception:  # noqa: BLE001 — see section comment
        pass


def _load_hitl_record(runtime: ToolRuntime) -> LangchainStoreMontySnapshot | None:
    """Fetch and validate the snapshot record for this tool call, if any."""
    store = getattr(runtime, "store", None)
    key = _hitl_store_key(runtime)
    if store is None or key is None:
        return None
    try:
        item = store.get(_snapshot_namespace(runtime), key)
        if item is None:
            return None
        return LangchainStoreMontySnapshot.model_validate(item.value)
    except Exception:  # noqa: BLE001 — see section comment
        return None


async def _aload_hitl_record(
    runtime: ToolRuntime,
) -> LangchainStoreMontySnapshot | None:
    """Async twin of _load_hitl_record."""
    store = getattr(runtime, "store", None)
    key = _hitl_store_key(runtime)
    if store is None or key is None:
        return None
    try:
        item = await store.aget(_snapshot_namespace(runtime), key)
        if item is None:
            return None
        return LangchainStoreMontySnapshot.model_validate(item.value)
    except Exception:  # noqa: BLE001 — see section comment
        return None


def _delete_hitl_record(runtime: ToolRuntime) -> None:
    """Remove the snapshot record once the call finishes (no orphans)."""
    store = getattr(runtime, "store", None)
    key = _hitl_store_key(runtime)
    if store is None or key is None:
        return
    try:
        store.delete(_snapshot_namespace(runtime), key)
    except Exception:  # noqa: BLE001 — see section comment
        pass


async def _adelete_hitl_record(runtime: ToolRuntime) -> None:
    """Async twin of _delete_hitl_record."""
    store = getattr(runtime, "store", None)
    key = _hitl_store_key(runtime)
    if store is None or key is None:
        return
    try:
        await store.adelete(_snapshot_namespace(runtime), key)
    except Exception:  # noqa: BLE001 — see section comment
        pass


def _burn_answered_interrupts(count: int) -> None:
    """Advance LangGraph's positional interrupt counter past answered ones.

    ``interrupt()`` matches resume values to calls POSITIONALLY per task
    execution (``scratchpad.resume[interrupt_counter]``). A snapshot-resumed
    run skips the host tools that already ran — including the ``interrupt()``
    calls they made — so without correction, the re-invoked interrupted
    tool's ``interrupt()`` would land on counter 0 and silently receive the
    FIRST recorded answer instead of its own.

    Each placeholder call here hits the "already answered" branch inside
    ``interrupt()`` (``idx < len(scratchpad.resume)``): it returns a stored
    answer — discarded — and advances the counter by one. After ``count``
    burns, the next real ``interrupt()`` lands on the right index.

    The placeholder value is only ever surfaced to a human if the counter
    bookkeeping is wrong (no stored answer at that index), in which case the
    graph pauses on it — loud and diagnosable rather than silently feeding a
    tool someone else's answer.
    """
    for _ in range(count):
        interrupt(
            "langchain-monty internal: interrupt-counter sync for snapshot "
            "resume — if you are seeing this, please report a bug"
        )


# --------------------------------------------------------------------------- #
# Prompt / stub rendering helpers                                             #
# --------------------------------------------------------------------------- #


def _visible_parameters(tool: BaseTool) -> list[tuple[str, str, bool]]:
    """Normalize a tool's visible JSON-schema args to (name, json_type, required).

    Single source of truth for both renderers below, so the signature the
    model *reads* in the prompt and the stub the type checker *enforces*
    can never disagree. Order follows ``tool.args`` declaration order — the
    same order _normalize_call_args uses to map positional arguments, which
    keeps all three views of the signature consistent.
    """
    args: dict[str, Any] = getattr(tool, "args", None) or {}
    if not isinstance(args, dict):
        return []

    # JSON Schema puts the required-ness on the parent object, not on the
    # individual property, so pull it from the model schema when available.
    required: set[str] = set()
    schema = getattr(tool, "args_schema", None)
    if schema is not None:
        try:
            raw = getattr(schema, "model_json_schema", lambda: {})() or {}
            required = set(raw.get("required", []))
        except Exception:  # noqa: BLE001 — schema introspection is best-effort
            required = set()

    out: list[tuple[str, str, bool]] = []
    for name, prop in args.items():
        if not isinstance(prop, dict):
            prop = {}
        typ = prop.get("type", "any")
        is_required = name in required or "default" not in prop
        out.append((name, typ, is_required))
    return out


def _format_tool_schema(tool: BaseTool) -> str:
    """Format a tool's name, argument signature, and description for prompts.

    Produces a block like::

        search(query: string, max_results: integer)
          Search the document index.

          Parameters:
            query (string, required) — The search query.
            max_results (integer, optional) — Number of results to return.
    """
    params = _visible_parameters(tool)
    args: dict[str, Any] = getattr(tool, "args", None) or {}

    # Signature line with JSON-schema types; optional params show "= ..."
    # (the stub-file convention) rather than a repr'd default that may not
    # round-trip cleanly.
    sig = ", ".join(
        f"{name}: {typ}" if required else f"{name}: {typ} = ..."
        for name, typ, required in params
    )

    header = f"{tool.name}({sig})"
    lines = [header]

    description = (getattr(tool, "description", None) or "").strip()
    if description:
        # Indent the description block under the signature.
        indented = "\n".join(f"  {line}" for line in description.splitlines())
        lines.append(indented)

    if params and description:
        lines.append("  Parameters:")
        for name, typ, required in params:
            prop = args.get(name) if isinstance(args, dict) else None
            desc = prop.get("description", "") if isinstance(prop, dict) else ""
            qualifier = "required" if required else "optional"
            entry = f"    {name} ({typ}, {qualifier})"
            if desc:
                entry += f" — {desc}"
            lines.append(entry)

    return "\n".join(lines)


# JSON Schema primitive -> Python annotation, for stub generation. Anything
# unmapped (anyOf, $ref, missing type) degrades to Any, which the checker
# accepts everywhere — stubs must only ever *under*-constrain, never reject
# valid calls.
_JSON_TYPE_TO_PY: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
    "null": "None",
}


def _render_type_stubs(
    host_tools: dict[str, BaseTool], allowlist: frozenset[str]
) -> str:
    """Render Python stub signatures for Monty's static type checker.

    For every allowlisted name we emit a ``def name(...) -> Any: ...`` stub:

    - Resolved tools get their real parameter names/types from the JSON
      schema, so the checker rejects hallucinated kwargs and wrong argument
      types with line-precise diagnostics *before* execution.
    - Allowlisted-but-unresolved names (deferred tools that didn't appear in
      the runtime's tool list) get a permissive ``(*args, **kwargs)`` stub
      so referencing them never fails the static check — the runtime
      allowlist still governs what actually executes.

    Optional parameters are rendered with the stub-file convention
    ``param: type = ...`` — the Ellipsis default sidesteps repr() pitfalls
    for non-literal defaults while still marking the parameter optional.

    Names that aren't valid Python identifiers are skipped: the sandbox has
    no way to spell a call to them anyway, and emitting them would make the
    whole stub file unparseable (which would disable checking entirely).
    """
    lines = ["from typing import Any", ""]
    for name in sorted(allowlist):
        if not name.isidentifier():
            continue  # see docstring — unspellable in sandbox code
        tool = host_tools.get(name)
        if tool is None:
            lines.append(f"def {name}(*args: Any, **kwargs: Any) -> Any: ...")
            continue
        parts = []
        for pname, json_type, required in _visible_parameters(tool):
            if not pname.isidentifier():
                # One bad parameter name would break the stub; degrade this
                # tool to a permissive signature instead.
                parts = ["*args: Any", "**kwargs: Any"]
                break
            py_type = _JSON_TYPE_TO_PY.get(json_type, "Any")
            parts.append(
                f"{pname}: {py_type}" if required else f"{pname}: {py_type} = ..."
            )
        lines.append(f"def {name}({', '.join(parts)}) -> Any: ...")
    return "\n".join(lines) + "\n"


def _resolve_deferred_from_request(
    request: ModelRequest[Any], deferred_names: frozenset[str]
) -> list[BaseTool]:
    """Find deferred-name tools among the tools bound to this model request.

    By the time wrap_model_call runs, middleware that injects tools (e.g.
    FilesystemMiddleware) has already contributed them to the request, so
    this is where deferred string names finally meet their tool objects —
    letting us show the model real signatures instead of bare names.
    """
    tools = getattr(request, "tools", None) or []
    resolved: list[BaseTool] = []
    try:
        for t in tools:
            if getattr(t, "name", None) in deferred_names:
                resolved.append(t)
    except TypeError:
        # request.tools wasn't iterable (e.g. a test double) — treat as none.
        return []
    return resolved


def _append_to_system_message(
    system_message: SystemMessage | None, text: str
) -> SystemMessage:
    """Append a text block to a (possibly absent) system message.

    Kept local on purpose: an earlier version imported the equivalent helper
    from ``deepagents.middleware._utils``, a private module — a fragile
    dependency for ten lines of code, and the only thing tying this package
    to deepagents at runtime.
    """
    new_content: list[ContentBlock] = (
        list(system_message.content_blocks) if system_message else []
    )
    if new_content:
        text = f"\n\n{text}"
    new_content.append({"type": "text", "text": text})
    return SystemMessage(content_blocks=new_content)


# --------------------------------------------------------------------------- #
# Monty payload helpers                                                       #
# --------------------------------------------------------------------------- #


def _make_return_value_result(value: Any) -> dict[str, Any]:
    """Build a successful ``ExternalResult`` dict for ``snapshot.resume()``.

    Monty's ``resume()`` accepts ``ExternalResult``, a union of TypedDict
    variants. ``{"return_value": ...}`` is the success form: Monty unwraps
    the value on the interpreter side and the sandbox call returns it.
    """
    return {"return_value": value}


def _make_exception_result(exc: Exception) -> dict[str, Exception]:
    """Build an exception ``ExternalResult`` dict for ``snapshot.resume()``.

    Two ways exist to signal an exception to Monty:

    - ``{"exc_type": "...", "message": "..."}`` only accepts a fixed set of
      builtin exception names; framework types (``ToolException``, ...) are
      rejected with ``TypeError: Unknown exception type``.
    - ``{"exception": exc}`` passes the real Python exception object and
      lets Monty map it internally — works for *any* class, so it's what we
      use for tool failures.

    The sandbox sees the exception as if the host function raised it, so
    interpreter code can ``try/except`` around host calls normally.
    """
    return {"exception": exc}


def _jsonable_output(progress: MontyComplete) -> Any:
    """Extract the final sandbox value in a JSON-safe form.

    Monty supports values plain JSON can't express (tuples, sets, bytes,
    dataclasses). The structured tool result eventually becomes a JSON
    ToolMessage, so a non-JSON-able value would blow up far from here, deep
    in message serialization. Strategy:

    - if ``json.dumps`` accepts the raw output, return it untouched (the
      overwhelmingly common case: None/bool/int/float/str/list/dict);
    - otherwise fall back to Monty's ``output_json()`` "natural form": rich
      types come back tagged, e.g. ``{"$tuple": [...]}``, ``{"$set": [...]}``
      — lossless, self-describing, and always serializable.
    """
    output = progress.output  # converted from VM repr on each access — grab once
    try:
        json.dumps(output)
        return output
    except (TypeError, ValueError):
        return json.loads(progress.output_json())


def _deserialize_return_value(value: Any) -> Any:
    """Attempt to JSON-deserialize string return values from host tools.

    LangChain tools typically return JSON-serialized strings. Monty needs
    native Python objects (lists, dicts) so that interpreter code like
    ``roster[0]`` indexes into a list rather than a string.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


# --------------------------------------------------------------------------- #
# Call-argument normalization                                                 #
# --------------------------------------------------------------------------- #

# Names corresponding to injected parameters on the host side. None of these
# may ever be forwarded from interpreter-supplied kwargs/args to the
# underlying tool — they are populated by LangChain itself (and re-added with
# the REAL values by _injected_args_for after stripping).
#
# ``ToolRuntime`` is the v1+ unified injection point; the earlier
# ``Injected*`` annotations are still present in many third-party tools.
# Note this is a *belt* on top of LangChain's own schema-stripping, which
# already excludes these from the visible tool schema. We re-strip here
# because the interpreter writes the call site, not the LLM, and we don't
# want an agent telling its interpreter code to pass ``tool_call_id=...``
# as if it were a normal kwarg.
_INJECTED_PARAM_NAMES: frozenset[str] = frozenset({
    "runtime",  # ToolRuntime
    "state",  # InjectedState (legacy)
    "store",  # InjectedStore
    "tool_call_id",  # InjectedToolCallId
    "config",  # RunnableConfig injection
    "context",  # langgraph Context (legacy)
    "stream_writer",  # injected from ToolRuntime
})


def _visible_schema_fields(tool: BaseTool) -> list[str]:
    """Return the host tool's *visible* arg names, excluding injected params.

    LangChain's own schema generation already hides injected parameters from
    ``tool.args``, but a few escape paths exist (custom ``args_schema``
    dicts, the open bugs cited in langchain-ai/langchain#34246 and #34293).
    We intersect ``tool.args`` with the not-injected set as a second line of
    defence so positional args from interpreter code can never land in an
    injected slot.
    """
    args_schema = getattr(tool, "args", None)
    if not isinstance(args_schema, dict):
        return []
    return [name for name in args_schema if name not in _INJECTED_PARAM_NAMES]


def _normalize_call_args(snapshot: FunctionSnapshot, tool: BaseTool) -> dict[str, Any]:
    """Map a Monty ``FunctionSnapshot`` to a LangChain tool kwargs dict.

    Monty surfaces both positional and keyword args. LangChain tools want a
    single kwargs dict matching ``tool.args_schema``. We:

    1. Strip any kwargs whose name matches a known injected parameter so the
       interpreter cannot forge a value for ``tool_call_id``, ``runtime``,
       ``state``, etc., even if it tries. (The real injected values are
       added back by ``_injected_args_for`` at invocation time.)
    2. Zip positional args against the tool's *visible* arg names — i.e.
       ``tool.args`` minus injected parameter names — in declaration order.
    3. Merge keyword args last, with kwargs winning on conflict (a conflict
       would itself indicate an interpreter-side bug).
    """
    args = snapshot.args or ()
    raw_kwargs = dict(snapshot.kwargs or {})

    # (1) Drop any injected names from interpreter-supplied kwargs.
    kwargs = {k: v for k, v in raw_kwargs.items() if k not in _INJECTED_PARAM_NAMES}

    if not args:
        return kwargs

    schema_fields = _visible_schema_fields(tool)

    out: dict[str, Any] = {}
    for i, v in enumerate(args):
        if i >= len(schema_fields):
            # Tool doesn't expose enough positional slots — fall back to
            # passing the rest as a single ``args`` list and let the tool
            # raise if it cares. Critically we do NOT fall through into
            # injected-param slots even if alphabetical ordering would put
            # them next.
            out.setdefault("args", []).append(v)
            continue
        out[schema_fields[i]] = v
    out.update(kwargs)
    return out
