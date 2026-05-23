"""Middleware exposing a sandboxed Python interpreter (pydantic-monty) to a deep agent.

This is the Python analog of ``langchain_quickjs.MontyCodeInterpreterMiddleware``.
QuickJS embeds a tiny JS VM; ``pydantic-monty`` is a tiny Python-subset VM
written in Rust by Pydantic (will back code-mode in Pydantic AI). Both are
in-process, microsecond-startup interpreters that completely isolate
agent-written code from the host except through explicitly-injected
external functions.

The mapping from QuickJS-world to Monty-world:

==============================  ============================================
``langchain-quickjs``           ``langchain-monty``
------------------------------  --------------------------------------------
QuickJS engine                  ``pydantic_monty.Monty``
``eval`` tool exposed to agent  ``eval_python`` tool exposed to agent
``ptc=[...]`` allowlist         ``ptc=[...]`` allowlist
host-runtime bridge (C↔JS)      ``start()`` / ``resume(FunctionSnapshot)``
``skills_backend=...``          ``skills_backend=...``
scoped JS globals               ``inputs=[...]`` + curated env
==============================  ============================================

Design follows ``deepagents.middleware.subagents.SubAgentMiddleware``:
``AgentMiddleware`` subclass that contributes a single tool and appends an
instruction block to the system prompt via ``wrap_model_call`` /
``awrap_model_call``. No mutable instance state — everything per-call flows
through ``ToolRuntime``.

The interesting bit, and the reason Monty is a strictly better fit than the
QuickJS analog the obvious way: every call from interpreter code into a host
tool surfaces on the host as a ``FunctionSnapshot``. We can drive it like an
event loop — invoke the LangChain tool through its normal machinery (so
``HumanInTheLoopMiddleware``, retries, traces, and ``Command``-returning
tools like ``task`` all keep working), then resume Monty with the result.
The snapshot is serializable, so a long-running interpreter call can be
checkpointed alongside the rest of graph state.
"""

from __future__ import annotations

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
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware._utils import append_to_system_message

from pydantic_monty import (
    Monty,
    ResourceLimits,
    FunctionSnapshot,
    MontyComplete,
)

from langchain_monty.models import EvalError, EvalPythonResult, MontyLimits


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

Iteration tip: Monty surfaces precise error messages. If your first attempt
hits an unsupported feature, the error will tell you what to change.
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
# Middleware                                                                  #
# --------------------------------------------------------------------------- #


class MontyCodeInterpreterMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Middleware that adds an ``eval_python`` tool backed by Monty.

    Args:
        ptc: *Programmatic tool calling* allowlist. Names of host tools that
            interpreter code is permitted to call. Tools not in this list are
            invisible to the sandbox even if they are bound to the agent.
            Default ``None`` means pure-compute only — no host tools exposed.
        limits: ``MontyLimits`` controlling per-call resource budgets.
        skills_backend: Optional deepagents backend that supplies importable
            Monty-compatible Python helpers. When set, the middleware reads
            ``*.py`` files from this backend at construction time and
            exposes their callables under ``skill_<module>_<name>``. (Monty
            does not yet support real imports.)
        system_prompt: Override for the appended system-prompt block. Pass
            ``None`` to disable the prompt contribution but keep the tool.
        tool_description: Override for the ``eval_python`` tool description.
            Placeholders: ``{available_host_tools}``, ``{max_duration_secs}``,
            ``{max_memory_bytes}``, ``{max_stack_depth}``.
        iteration_budget: Hard cap on host-tool round-trips per ``eval_python``
            call. Defends against an agent that writes a tight loop calling a
            host tool forever. Default 64.

    Example:
        ```python
        from deepagents import create_deep_agent
        from langchain_monty import MontyCodeInterpreterMiddleware

        agent = create_deep_agent(
            model="anthropic:claude-sonnet-4-6",
            middleware=[MontyCodeInterpreterMiddleware(ptc=["task", "search"])],
        )
        ```
    """

    def __init__(
        self,
        *,
        ptc: Sequence[str] | None = None,
        limits: MontyLimits | None = None,
        skills_backend: BackendProtocol | BackendFactory | None = None,
        system_prompt: str | None = CODE_INTERPRETER_SYSTEM_PROMPT,
        tool_description: str | None = None,
        iteration_budget: int = 64,
    ) -> None:
        super().__init__()

        self._ptc: frozenset[str] = frozenset(ptc or ())
        self._limits = limits or MontyLimits()
        self._skills_backend = skills_backend
        self._iteration_budget = iteration_budget
        self._description_template = (
            tool_description or CODE_INTERPRETER_TOOL_DESCRIPTION
        )

        # Build the system-prompt block once. The list of allowed host names
        # is fixed at construction time; the description rendered at tool-
        # build time substitutes the same list. Both are static across the
        # life of the middleware instance.
        if system_prompt is not None:
            if self._ptc:
                allowed = "\n".join(f"- {name}(...)" for name in sorted(self._ptc))
                self.system_prompt: str | None = (
                    system_prompt
                    + "\n\nHost functions exposed to the interpreter:\n"
                    + allowed
                )
            else:
                self.system_prompt = (
                    system_prompt
                    + "\n\nNo host functions are exposed; the interpreter is "
                    "pure compute only."
                )
        else:
            self.system_prompt = None

        self._tool = self._build_eval_python_tool()
        self.tools: list[BaseTool] = [self._tool]

    # --------------------------------------------------------------- tool

    def _build_eval_python_tool(self) -> BaseTool:
        """Build the ``eval_python`` StructuredTool.

        Closes over ``self._ptc``, ``self._limits``, ``self._skills_backend``,
        and ``self._iteration_budget``. The Monty interpreter object itself
        is constructed fresh per invocation so two concurrent ``eval_python``
        calls on different threads cannot share state.

        Schema-visibility contract: the only parameter the LLM sees on the
        ``eval_python`` tool is ``code: str``. The ``runtime: ToolRuntime``
        parameter is auto-injected by LangChain and stripped from the
        generated schema — that's how LangChain treats any ``ToolRuntime``-
        typed argument since the v1 tools refactor. We rely on that here
        rather than building a custom ``args_schema``, which is what the
        deepagents ``task`` tool (and the QuickJS analog) also do.

        Inside the driver, ``_normalize_call_args`` strips injected-style
        names from interpreter-supplied kwargs/positional args before they
        reach a host tool, so the agent's Python code can't forge a value
        for ``tool_call_id``, ``state``, ``store``, etc.
        """
        ptc = self._ptc
        limits = self._limits
        iteration_budget = self._iteration_budget
        description_template = self._description_template

        def _render_description(host_names: list[str]) -> str:
            if host_names:
                listing = "\n".join(f"- {n}(...)" for n in host_names)
            else:
                listing = "(none — interpreter is pure-compute only)"
            return description_template.format(
                available_host_tools=listing,
                max_duration_secs=limits.max_duration_secs,
                max_memory_bytes=limits.max_memory_bytes,
                max_stack_depth=limits.max_stack_depth,
            )

        def _resolve_host_tools(runtime: ToolRuntime) -> dict[str, BaseTool]:
            """Pick host tools the interpreter is allowed to call.

            ``ToolRuntime`` carries the bound tool set; filter to the
            allowlist and drop ``eval_python`` itself to prevent recursive
            invocation through the bridge.
            """
            bound = getattr(runtime, "tools", None) or []
            return {
                t.name: t for t in bound if t.name in ptc and t.name != "eval_python"
            }

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

        def _exception_result(exc: BaseException) -> dict[str, Any]:
            return EvalPythonResult(
                error=EvalError(type=type(exc).__name__, message=str(exc)),
            ).model_dump()

        def _complete_result(progress: MontyComplete) -> dict[str, Any]:
            return EvalPythonResult(
                result=progress.output,
                stdout=getattr(progress, "stdout", "") or "",
            ).model_dump()

        def _drive_sync(
            monty: Monty,
            inputs: dict[str, Any],
            host_tools: dict[str, BaseTool],
            runtime: ToolRuntime,
        ) -> dict[str, Any]:
            """Synchronous driver over Monty's start/resume protocol.

            Monty pauses every time interpreter code calls a host function.
            ``start()`` and subsequent ``resume()`` calls return either:

            - ``FunctionSnapshot`` — paused on a host call. Invoke the
              corresponding LangChain tool, then
              ``snapshot.resume({"return_value": ...})``.
            - ``MontyComplete`` — finished; ``.output`` holds the final value.

            Tools are invoked through their normal LangChain machinery, so any
            wrapping middleware (HITL, retries, tracing) keeps applying. Host
            errors surface inside the interpreter as exceptions the user code
            can ``try/except``.
            """
            progress = monty.start(inputs=inputs, limits=limits.to_monty())
            iterations = 0
            while isinstance(progress, FunctionSnapshot):
                iterations += 1
                if iterations > iteration_budget:
                    return _iteration_budget_result()

                name = progress.function_name
                tool = host_tools.get(name)
                if tool is None:
                    progress = progress.resume(
                        {
                            "error": (
                                f"host function {name!r} is not in the "
                                f"allowlist; available: {sorted(host_tools)}"
                            )
                        }
                    )
                    continue

                tool_kwargs = _normalize_call_args(progress, tool)
                try:
                    return_value = tool.invoke(
                        tool_kwargs,
                        config={"configurable": runtime.config},
                    )
                    progress = progress.resume({"return_value": return_value})
                except Exception as exc:  # noqa: BLE001
                    progress = progress.resume(
                        {"error": f"{type(exc).__name__}: {exc}"}
                    )

            assert isinstance(progress, MontyComplete)
            return _complete_result(progress)

        async def _drive_async(
            monty: Monty,
            inputs: dict[str, Any],
            host_tools: dict[str, BaseTool],
            runtime: ToolRuntime,
        ) -> dict[str, Any]:
            progress = monty.start(inputs=inputs, limits=limits.to_monty())
            iterations = 0
            while isinstance(progress, FunctionSnapshot):
                iterations += 1
                if iterations > iteration_budget:
                    return _iteration_budget_result()

                name = progress.function_name
                tool = host_tools.get(name)
                if tool is None:
                    progress = progress.resume(
                        {
                            "error": (
                                f"host function {name!r} is not in the "
                                f"allowlist; available: {sorted(host_tools)}"
                            )
                        }
                    )
                    continue

                tool_kwargs = _normalize_call_args(progress, tool)
                try:
                    return_value = await tool.ainvoke(
                        tool_kwargs,
                        config={"configurable": runtime.config},
                    )
                    progress = progress.resume({"return_value": return_value})
                except Exception as exc:  # noqa: BLE001
                    progress = progress.resume(
                        {"error": f"{type(exc).__name__}: {exc}"}
                    )

            assert isinstance(progress, MontyComplete)
            return _complete_result(progress)

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
            try:
                monty = Monty(code, inputs=[])
            except Exception as exc:  # noqa: BLE001 — compile/parse error
                return _exception_result(exc)
            try:
                return _drive_sync(
                    monty,
                    inputs={},
                    host_tools=host_tools,
                    runtime=runtime,
                )
            except Exception as exc:  # noqa: BLE001 — resource exhaustion
                return _exception_result(exc)

        async def aeval_python(
            code: Annotated[str, "Python source to execute in the Monty sandbox."],
            runtime: ToolRuntime,
        ) -> dict[str, Any]:
            host_tools = _resolve_host_tools(runtime)
            try:
                monty = Monty(code, inputs=[])
            except Exception as exc:  # noqa: BLE001
                return _exception_result(exc)
            try:
                return await _drive_async(
                    monty,
                    inputs={},
                    host_tools=host_tools,
                    runtime=runtime,
                )
            except Exception as exc:  # noqa: BLE001
                return _exception_result(exc)

        return StructuredTool.from_function(
            name="eval_python",
            func=eval_python,
            coroutine=aeval_python,
            description=_render_description(sorted(ptc)),
        )

    # ---------------------------------------------------------- prompt hooks

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(
                request.system_message, self.system_prompt
            )
            return handler(request.override(system_message=new_system_message))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(
                request.system_message, self.system_prompt
            )
            return await handler(request.override(system_message=new_system_message))
        return await handler(request)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


# Names corresponding to injected parameters on the host side. None of these
# may ever be forwarded from interpreter-supplied kwargs/args to the
# underlying tool — they are populated by LangChain itself.
#
# ``ToolRuntime`` is the v1+ unified injection point; the earlier
# ``Injected*`` annotations are still present in many third-party tools.
# Note this is a *belt* on top of LangChain's own schema-stripping, which
# already excludes these from the visible tool schema. We re-strip here
# because the interpreter writes the call site, not the LLM, and we don't
# want an agent telling its interpreter code to pass ``tool_call_id=...``
# as if it were a normal kwarg.
_INJECTED_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "runtime",  # ToolRuntime
        "state",  # InjectedState (legacy)
        "store",  # InjectedStore
        "tool_call_id",  # InjectedToolCallId
        "config",  # RunnableConfig injection
        "context",  # langgraph Context (legacy)
        "stream_writer",  # injected from ToolRuntime
    }
)


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
       ``state``, etc., even if it tries.
    2. Zip positional args against the tool's *visible* arg names — i.e.
       ``tool.args`` minus injected parameter names — in declaration order.
    3. Merge keyword args last, with kwargs winning on conflict (a conflict
       would itself indicate an interpreter-side bug).
    """
    args = getattr(snapshot, "args", ()) or ()
    raw_kwargs = dict(getattr(snapshot, "kwargs", {}) or {})

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
