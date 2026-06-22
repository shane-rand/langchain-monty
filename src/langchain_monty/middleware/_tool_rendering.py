"""Prompt/stub rendering and call-argument normalization for host tools."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import ModelRequest
from langchain.tools import BaseTool
from langchain_core.messages import ContentBlock, SystemMessage
from pydantic_monty import FunctionSnapshot

from langchain_monty.middleware._bridge import INJECTED_PARAM_NAMES


def _visible_parameters(tool: BaseTool) -> list[tuple[str, str, bool]]:
    """Return (name, json_type, required) for each visible arg, in declaration order."""
    args: dict[str, Any] = getattr(tool, "args", None) or {}
    if not isinstance(args, dict):
        return []

    required: set[str] = set()
    schema = getattr(tool, "args_schema", None)
    if schema is not None:
        try:
            raw = getattr(schema, "model_json_schema", lambda: {})() or {}
            required = set(raw.get("required", []))
        except Exception:  # noqa: BLE001
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
    """Format a tool's signature and description for the system prompt."""
    params = _visible_parameters(tool)
    args: dict[str, Any] = getattr(tool, "args", None) or {}

    sig = ", ".join(
        f"{name}: {typ}" if required else f"{name}: {typ} = ..."
        for name, typ, required in params
    )

    header = f"{tool.name}({sig})"
    lines = [header]

    description = (getattr(tool, "description", None) or "").strip()
    if description:
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


# JSON Schema → Python annotation. Unmapped types degrade to Any so stubs
# only ever under-constrain, never reject valid calls.
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

    Resolved tools get real parameter types. Unresolved names (deferred tools
    not yet in the runtime) get a permissive ``(*args, **kwargs)`` stub.
    Non-identifier names are skipped — they're unspellable in sandbox code.
    """
    lines = ["from typing import Any", ""]
    for name in sorted(allowlist):
        if not name.isidentifier():
            continue
        tool = host_tools.get(name)
        if tool is None:
            lines.append(f"def {name}(*args: Any, **kwargs: Any) -> Any: ...")
            continue
        parts = []
        for pname, json_type, required in _visible_parameters(tool):
            if not pname.isidentifier():
                # One bad parameter name breaks the whole stub; degrade to permissive.
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
    """Find deferred-name tools among the tools bound to this model request."""
    tools = getattr(request, "tools", None) or []
    resolved: list[BaseTool] = []
    try:
        for t in tools:
            if getattr(t, "name", None) in deferred_names:
                resolved.append(t)
    except TypeError:
        return []
    return resolved


def _append_to_system_message(
    system_message: SystemMessage | None, text: str
) -> SystemMessage:
    """Append a text block to a (possibly absent) system message.

    Kept local to avoid a runtime dependency on deepagents' private modules.
    """
    new_content: list[ContentBlock] = (
        list(system_message.content_blocks) if system_message else []
    )
    if new_content:
        text = f"\n\n{text}"
    new_content.append({"type": "text", "text": text})
    return SystemMessage(content_blocks=new_content)


def _visible_schema_fields(tool: BaseTool) -> list[str]:
    """tool.args keys excluding injected params, in declaration order."""
    args_schema = getattr(tool, "args", None)
    if not isinstance(args_schema, dict):
        return []
    return [name for name in args_schema if name not in INJECTED_PARAM_NAMES]


def _normalize_call_args(snapshot: FunctionSnapshot, tool: BaseTool) -> dict[str, Any]:
    """Map a Monty ``FunctionSnapshot`` to a LangChain tool kwargs dict.

    1. Strip injected param names from interpreter-supplied kwargs.
    2. Zip positional args against visible schema fields in declaration order.
    3. Merge keyword args (kwargs win on conflict).
    """
    args = snapshot.args or ()
    raw_kwargs = dict(snapshot.kwargs or {})

    kwargs = {k: v for k, v in raw_kwargs.items() if k not in INJECTED_PARAM_NAMES}

    if not args:
        return kwargs

    schema_fields = _visible_schema_fields(tool)

    out: dict[str, Any] = {}
    for i, v in enumerate(args):
        if i >= len(schema_fields):
            # No more named slots; accumulate extras under "args" rather than
            # spilling into injected-param positions.
            out.setdefault("args", []).append(v)
            continue
        out[schema_fields[i]] = v
    out.update(kwargs)
    return out
