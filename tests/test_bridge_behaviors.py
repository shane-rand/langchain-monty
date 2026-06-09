"""Behavioral tests for the host-call bridge, run against the REAL Monty VM.

Unlike test_middleware.py (which mocks Monty to test driver plumbing) and
test_integration.py (which spins up a whole deep agent), these tests invoke
the ``eval_python`` tool function directly with a real ``ToolRuntime``. That
exercises the genuine Monty pause/resume protocol, the static type checker,
and the ToolCall-based host-tool invocation path with minimal scaffolding.
"""

from __future__ import annotations

import json
from typing import Annotated

import pytest
from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from langchain_monty import MontyCodeInterpreterMiddleware

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _runtime(tools=None, state=None) -> ToolRuntime:
    """A real ToolRuntime, as the agent's ToolNode would inject it."""
    return ToolRuntime(
        state=state if state is not None else {},
        context=None,
        config={},
        stream_writer=lambda *_: None,
        tool_call_id="outer-eval-python-call",
        store=None,
        tools=list(tools or []),
    )


def _run(middleware: MontyCodeInterpreterMiddleware, code: str, **rt_kwargs):
    """Invoke the sync eval_python entrypoint directly."""
    return middleware._tool.func(code=code, runtime=_runtime(**rt_kwargs))


async def _arun(middleware: MontyCodeInterpreterMiddleware, code: str, **rt_kwargs):
    """Invoke the async eval_python entrypoint directly."""
    return await middleware._tool.coroutine(code=code, runtime=_runtime(**rt_kwargs))


@tool
def lookup(key: str) -> str:
    """Look up a value by key.

    Returns a JSON object with: key (str), value (int).
    """
    return json.dumps({"key": key, "value": len(key)})


# --------------------------------------------------------------------------- #
# Static type checking against generated stubs                                #
# --------------------------------------------------------------------------- #


class TestTypeChecking:
    def test_hallucinated_kwarg_caught_before_execution(self):
        """A kwarg the tool doesn't declare fails the static check; the tool
        is never invoked and the error carries line-precise diagnostics."""
        calls = []

        @tool
        def search(query: str) -> str:
            """Search."""
            calls.append(query)
            return "[]"

        m = MontyCodeInterpreterMiddleware(ptc=[search])
        r = _run(m, 'search(query="x", limit=5)')

        assert r["error"]["type"] == "TypeCheckError"
        assert "limit" in r["error"]["traceback"]
        assert calls == []  # nothing executed

    def test_wrong_argument_type_caught(self):
        @tool
        def repeat(text: str, times: int) -> str:
            """Repeat text."""
            return text * times

        m = MontyCodeInterpreterMiddleware(ptc=[repeat])
        r = _run(m, 'repeat("a", times="three")')
        assert r["error"]["type"] == "TypeCheckError"

    def test_valid_call_passes_check_and_runs(self):
        m = MontyCodeInterpreterMiddleware(ptc=[lookup])
        r = _run(m, 'lookup("abc")["value"]')
        assert r["error"] is None
        assert r["result"] == 3

    def test_type_check_can_be_disabled(self):
        """With type_check=False the bad kwarg reaches the tool at runtime
        instead (and fails there) — the pre-flight gate is gone."""

        @tool
        def search(query: str) -> str:
            """Search."""
            return "[]"

        m = MontyCodeInterpreterMiddleware(ptc=[search], type_check=False)
        r = _run(m, 'search(query="x", limit=5)')
        # No TypeCheckError; the failure (if any) is a runtime one.
        assert r["error"] is None or r["error"]["type"] != "TypeCheckError"

    def test_deferred_names_get_permissive_stubs(self):
        """Allowlisted-but-unresolved names must not fail the static check —
        the runtime allowlist governs execution, not the type checker."""
        m = MontyCodeInterpreterMiddleware(ptc=["read_file"])
        r = _run(m, 'read_file("/tmp/x")')
        # Passes the type check, then fails at runtime because the deferred
        # tool was never resolved from the (empty) runtime tool list.
        assert r["error"]["type"] != "TypeCheckError"


# --------------------------------------------------------------------------- #
# Injected-argument forwarding (ToolRuntime / InjectedState / tool_call_id)   #
# --------------------------------------------------------------------------- #


class TestInjectedArgForwarding:
    def test_runtime_requiring_tool_works_through_bridge(self):
        """Tools that declare ``runtime: ToolRuntime`` (e.g. deepagents
        filesystem tools) must receive the live runtime when bridged —
        a bare-args invoke would fail their validation outright."""

        @tool
        def read_state(key: str, runtime: ToolRuntime) -> str:
            """Read a key from agent state."""
            return json.dumps(runtime.state.get(key))

        m = MontyCodeInterpreterMiddleware(ptc=[read_state])
        r = _run(m, 'read_state("color")', state={"color": "teal"})
        assert r["error"] is None
        assert r["result"] == "teal"

    def test_sandbox_cannot_forge_injected_values(self):
        """Interpreter-supplied runtime/state kwargs are stripped and
        replaced with the real ones."""

        @tool
        def read_state(key: str, runtime: ToolRuntime) -> str:
            """Read a key from agent state."""
            return json.dumps(runtime.state.get(key))

        m = MontyCodeInterpreterMiddleware(ptc=[read_state], type_check=False)
        # The sandbox tries to smuggle its own "runtime"; the bridge strips
        # it (it would fail type checking too, hence type_check=False here).
        r = _run(
            m,
            'read_state("color", runtime="forged")',
            state={"color": "teal"},
        )
        assert r["error"] is None
        assert r["result"] == "teal"

    def test_tool_call_id_tool_works_through_bridge(self):
        """InjectedToolCallId tools require a full ToolCall invocation; the
        bridge's synthetic id satisfies that and marks bridged calls."""

        @tool
        def echo_id(x: int, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
            """Echo the tool call id."""
            return f"{x}:{tool_call_id}"

        m = MontyCodeInterpreterMiddleware(ptc=[echo_id])
        r = _run(m, "echo_id(7)")
        assert r["error"] is None
        assert r["result"].startswith("7:eval_python:")


# --------------------------------------------------------------------------- #
# GraphInterrupt propagation (human-in-the-loop)                              #
# --------------------------------------------------------------------------- #


class TestGraphInterrupt:
    def test_interrupt_propagates_out_of_eval_python_sync(self):
        """GraphInterrupt must NOT become a sandbox/structured error — it has
        to reach LangGraph so the graph checkpoints and pauses."""

        @tool
        def sensitive(action: str) -> str:
            """Do something needing approval."""
            raise GraphInterrupt(())

        m = MontyCodeInterpreterMiddleware(ptc=[sensitive])
        with pytest.raises(GraphInterrupt):
            _run(m, 'sensitive("delete")')

    @pytest.mark.asyncio
    async def test_interrupt_propagates_out_of_eval_python_async(self):
        @tool
        def sensitive(action: str) -> str:
            """Do something needing approval."""
            raise GraphInterrupt(())

        m = MontyCodeInterpreterMiddleware(ptc=[sensitive])
        with pytest.raises(GraphInterrupt):
            await _arun(m, 'sensitive("delete")')


# --------------------------------------------------------------------------- #
# Call styles: plain vs awaited, in both drivers                              #
# --------------------------------------------------------------------------- #

GATHER_CODE = """\
import asyncio

async def go():
    a, b = await asyncio.gather(lookup("ab"), lookup("cdef"))
    return [a["value"], b["value"]]

asyncio.run(go())
"""

MIXED_CODE = """\
import asyncio

async def go():
    a = await lookup("ab")
    lookup("never awaited")
    return a["value"]

asyncio.run(go())
"""


class TestCallStyles:
    def test_plain_style_sync_driver(self):
        m = MontyCodeInterpreterMiddleware(ptc=[lookup])
        r = _run(m, 'lookup("abcd")["value"]')
        assert r["error"] is None
        assert r["result"] == 4

    @pytest.mark.asyncio
    async def test_plain_style_async_driver(self):
        """The adaptive rerun: plain-style code must work under ainvoke too,
        not silently evaluate to an un-awaited coroutine."""
        m = MontyCodeInterpreterMiddleware(ptc=[lookup])
        r = await _arun(m, 'lookup("abcd")["value"]')
        assert r["error"] is None
        assert r["result"] == 4

    def test_gather_style_sync_driver(self):
        """Awaited style works in the sync driver too (sequential execution),
        so a snippet behaves the same under invoke and ainvoke."""
        m = MontyCodeInterpreterMiddleware(ptc=[lookup])
        r = _run(m, GATHER_CODE)
        assert r["error"] is None
        assert r["result"] == [2, 4]

    @pytest.mark.asyncio
    async def test_gather_style_async_driver(self):
        m = MontyCodeInterpreterMiddleware(ptc=[lookup])
        r = await _arun(m, GATHER_CODE)
        assert r["error"] is None
        assert r["result"] == [2, 4]

    def test_mixed_style_rejected_with_clear_error(self):
        """Awaiting some host calls while discarding others can't be fixed by
        an eager rerun (it would duplicate the executed calls) — the agent
        gets a structured error naming the rule instead."""
        m = MontyCodeInterpreterMiddleware(ptc=[lookup])
        r = _run(m, MIXED_CODE)
        assert r["error"]["type"] == "UnawaitedHostCallError"

    def test_iteration_budget_counts_calls_in_loop(self):
        m = MontyCodeInterpreterMiddleware(ptc=[lookup], iteration_budget=3)
        code = "\n".join(f'lookup("{c}")' for c in "abcde")  # 5 calls
        r = _run(m, code)
        assert r["error"]["type"] == "IterationBudgetExceeded"


# --------------------------------------------------------------------------- #
# Result/error shapes                                                         #
# --------------------------------------------------------------------------- #


class TestResultShapes:
    def test_non_json_native_result_uses_tagged_form(self):
        """A set can't pass through json.dumps; the fallback uses Monty's
        tagged natural-form mapping so the result stays serializable."""
        m = MontyCodeInterpreterMiddleware()
        r = _run(m, "{1, 2, 3}")
        assert r["error"] is None
        assert sorted(r["result"]["$set"]) == [1, 2, 3]

    def test_runtime_error_carries_real_type_and_traceback(self):
        m = MontyCodeInterpreterMiddleware()
        r = _run(m, "x = [1]\nx[5]")
        assert r["error"]["type"] == "IndexError"
        assert "Traceback" in r["error"]["traceback"]
        assert "x[5]" in r["error"]["traceback"]  # source preview line

    def test_command_returning_tool_yields_clear_error(self):
        """Command results mutate graph state and can't be honoured from
        inside a nested interpreter — the sandbox sees an explanatory
        exception, not an opaque object."""

        @tool
        def task(name: str) -> Command:
            """Launch a subagent task."""
            return Command(update={"messages": []})

        m = MontyCodeInterpreterMiddleware(ptc=[task])
        r = _run(m, 'task("research")')
        assert r["error"] is not None
        assert "Command" in r["error"]["message"]


# --------------------------------------------------------------------------- #
# Dynamic system-prompt rendering for deferred tools                          #
# --------------------------------------------------------------------------- #


class TestDeferredSchemaRendering:
    def test_deferred_schema_rendered_when_resolved_on_request(self):
        """ptc=["lookup"] starts as a bare name; once the request's tool list
        contains the real tool, its full signature must appear in the
        system-prompt block."""
        from unittest.mock import MagicMock

        m = MontyCodeInterpreterMiddleware(ptc=["lookup"])
        # Bare name (no signature) in the static block...
        assert "lookup" in m.system_prompt
        assert "lookup(key: string)" not in m.system_prompt

        request = MagicMock()
        request.system_message = None
        request.tools = [lookup]
        captured = {}

        def _override(**kwargs):
            captured.update(kwargs)
            return request

        request.override = _override
        m.wrap_model_call(request, lambda req: "resp")

        rendered = captured["system_message"].text
        # ...full signature + docstring once resolved per-request.
        assert "lookup(key: string)" in rendered
        assert "Look up a value by key." in rendered

    def test_schemas_not_duplicated_into_tool_description(self):
        """Full schemas live in the system prompt only; the tool description
        lists names (avoids paying for schema text twice per model call)."""

        @tool
        def search(query: str) -> str:
            """A searchable index of documents."""
            return "[]"

        m = MontyCodeInterpreterMiddleware(ptc=[search])
        assert "search" in m._tool.description
        assert "A searchable index" not in m._tool.description
        assert "A searchable index" in m.system_prompt

    def test_schemas_fall_back_to_description_without_system_prompt(self):
        """system_prompt=None removes the prompt block, so the description
        becomes the only home for the signatures."""

        @tool
        def search(query: str) -> str:
            """A searchable index of documents."""
            return "[]"

        m = MontyCodeInterpreterMiddleware(ptc=[search], system_prompt=None)
        assert m.system_prompt is None
        assert "A searchable index" in m._tool.description
