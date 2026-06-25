"""LangChain middleware exposing pydantic-monty to an agent as ``eval_python``."""

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

from langchain_monty.middleware._driver import _MontyDriver
from langchain_monty.middleware._tool_rendering import (
    _append_to_system_message,
    _format_tool_schema,
    _render_type_stubs,
    _resolve_deferred_from_request,
)
from langchain_monty.models import MontyLimits

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

        def _make_driver(code: str, runtime: ToolRuntime) -> _MontyDriver:
            host_tools = _resolve_host_tools(runtime, ptc_tools, ptc)
            return _MontyDriver(
                code=code,
                host_tools=host_tools,
                runtime=runtime,
                limits=limits,
                iteration_budget=budget,
                compile_kwargs=_compile_kwargs(host_tools, type_check, ptc),
            )

        def eval_python(
            code: Annotated[
                str,
                "Python source to execute in the Monty sandbox. The value of "
                "the final expression is returned as ``result``. Stdout is "
                "captured. Exceptions return as a structured ``error``.",
            ],
            runtime: ToolRuntime,
        ) -> dict[str, Any]:
            return _make_driver(code, runtime).run_sync()

        async def aeval_python(
            code: Annotated[
                str,
                "Python source to execute in the Monty sandbox. The value of "
                "the final expression is returned as ``result``. Stdout is "
                "captured. Exceptions return as a structured ``error``.",
            ],
            runtime: ToolRuntime,
        ) -> dict[str, Any]:
            return await _make_driver(code, runtime).run_async()

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
