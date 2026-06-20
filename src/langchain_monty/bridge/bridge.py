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

import uuid
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime
from langgraph.types import Command
from langchain_core.tools import BaseTool
from typing import Any
from langgraph.errors import GraphInterrupt
from pydantic_monty import ExternalResult

from langchain_monty.helpers import deserialize_return_value, make_exception_result, make_return_value_result


INJECTED_PARAM_NAMES: frozenset[str] = frozenset({
    "runtime",  # ToolRuntime
    "state",  # InjectedState (legacy)
    "store",  # InjectedStore
    "tool_call_id",  # InjectedToolCallId
    "config",  # RunnableConfig injection
    "context",  # langgraph Context (legacy)
    "stream_writer",  # injected from ToolRuntime
})

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
        names |= (full - visible) & INJECTED_PARAM_NAMES
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
        return deserialize_return_value(text)
    return deserialize_return_value(raw)


def invoke_host_tool_sync(
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
        return make_return_value_result(_extract_tool_output(raw))
    except GraphInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 — surfaced in-sandbox
        return make_exception_result(exc)


async def invoke_host_tool_async(
    tool: BaseTool, kwargs: dict[str, Any], runtime: ToolRuntime
) -> ExternalResult:
    """Async twin of invoke_host_tool_sync (used by the gather batches)."""
    try:
        raw = await tool.ainvoke(
            _build_tool_call(tool, kwargs, runtime),
            config=getattr(runtime, "config", None) or None,
        )
        return make_return_value_result(_extract_tool_output(raw))
    except GraphInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 — surfaced in-sandbox
        return make_exception_result(exc)