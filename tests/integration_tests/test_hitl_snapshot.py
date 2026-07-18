"""HITL snapshot-resume tests: no task replay when a store is configured.

Two layers:

- ``TestSnapshotResumeUnit`` drives the ``eval_python`` tool function
  directly (real Monty VM, real ``InMemoryStore``) and simulates LangGraph's
  interrupt/replay cycle by calling the tool twice with the same
  ``tool_call_id``. This pins the capture/resume mechanics without a graph.
- ``TestSnapshotResumeEndToEnd`` runs a real agent graph (checkpointer +
  store + genuine ``interrupt()`` calls inside a host tool), including two
  sequential interrupts in one snippet — the case that exercises the
  positional interrupt-counter burn (``_burn_answered_interrupts``).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt

from langchain_monty import MontyCodeInterpreterMiddleware

_NAMESPACE = ("langchain_monty", "snapshots", "thread-1")
_TOOL_CALL_ID = "eval-call-1"


def _sync_call(m: MontyCodeInterpreterMiddleware, **kwargs: Any) -> Any:
    """Call eval_python's sync function directly, bypassing ToolNode."""
    func = m._tool.func
    assert func is not None
    return func(**kwargs)


async def _async_call(m: MontyCodeInterpreterMiddleware, **kwargs: Any) -> Any:
    """Call eval_python's async function directly, bypassing ToolNode."""
    coroutine = m._tool.coroutine
    assert coroutine is not None
    return await coroutine(**kwargs)


def _hitl_runtime(store=None) -> ToolRuntime:
    """A ToolRuntime as the ToolNode would inject it, with thread + store.

    Each call builds a FRESH runtime with the SAME tool_call_id/thread_id —
    exactly what LangGraph's task replay hands the re-run tool call.
    """
    return ToolRuntime(
        state={},
        context=None,
        config={"configurable": {"thread_id": "thread-1"}},
        stream_writer=lambda *_: None,
        tool_call_id=_TOOL_CALL_ID,
        store=store,
        tools=[],
    )


def _tracking_tool():
    """A side-effecting tool plus its call log — the 'must not replay' probe."""
    calls: list[str] = []

    @tool
    def track(x: str) -> str:
        """Record a value.

        Returns a JSON object with: n (int).
        """
        calls.append(x)
        return json.dumps({"n": len(calls)})

    return track, calls


def _gated_tool(name: str = "gated"):
    """A tool that raises GraphInterrupt until approved (simulated HITL)."""
    state = {"approved": False, "calls": 0}

    @tool
    def gated(action: str) -> str:
        """Do something needing approval.

        Returns a JSON object with: ok (str).
        """
        state["calls"] += 1
        if not state["approved"]:
            raise GraphInterrupt(())
        return json.dumps({"ok": action})

    gated.name = name
    return gated, state


class TestSnapshotResumeUnit:
    CODE = 'track("a")\ngated("x")["ok"]'

    def test_prior_host_calls_not_replayed_with_store(self):
        """The headline behaviour: with a store, the replayed call resumes
        from the snapshot — the tool that already ran is NOT re-invoked."""
        track, calls = _tracking_tool()
        gated, gate = _gated_tool()
        m = MontyCodeInterpreterMiddleware(ptc=[track, gated])
        store = InMemoryStore()

        with pytest.raises(GraphInterrupt):
            _sync_call(m, code=self.CODE, runtime=_hitl_runtime(store))
        assert calls == ["a"]
        assert store.get(_NAMESPACE, _TOOL_CALL_ID) is not None

        gate["approved"] = True
        r = _sync_call(m, code=self.CODE, runtime=_hitl_runtime(store))

        assert r["error"] is None
        assert r["result"] == "x"
        assert calls == ["a"]  # ran exactly once across both executions
        assert gate["calls"] == 2  # interrupted once, re-invoked once
        # No orphaned record after completion.
        assert store.get(_NAMESPACE, _TOOL_CALL_ID) is None

    def test_record_contents(self):
        """The persisted sidecar carries what the VM snapshot cannot."""
        track, _ = _tracking_tool()
        gated, _ = _gated_tool()
        m = MontyCodeInterpreterMiddleware(ptc=[track, gated])
        store = InMemoryStore()

        code = 'print("before")\ntrack("a")\ngated("x")'
        with pytest.raises(GraphInterrupt):
            _sync_call(m, code=code, runtime=_hitl_runtime(store))

        rec = store.get(_NAMESPACE, _TOOL_CALL_ID).value
        assert rec["stdout"] == "before\n"
        # track counted; the interrupted gated call excluded (re-counted on
        # resume so the iteration budget stays exact).
        assert rec["host_calls"] == 1
        assert rec["interrupts_answered"] == 0
        assert rec["monty_snapshot"]  # non-empty b64 payload

    def test_stdout_preserved_across_resume(self):
        track, _ = _tracking_tool()
        gated, gate = _gated_tool()
        m = MontyCodeInterpreterMiddleware(ptc=[track, gated])
        store = InMemoryStore()

        code = 'print("before")\ntrack("a")\ngated("x")\nprint("after")\n42'
        with pytest.raises(GraphInterrupt):
            _sync_call(m, code=code, runtime=_hitl_runtime(store))

        gate["approved"] = True
        r = _sync_call(m, code=code, runtime=_hitl_runtime(store))
        assert r["error"] is None
        assert r["result"] == 42
        # "before" printed pre-interrupt exactly once, "after" post-resume.
        assert r["stdout"] == "before\nafter\n"

    def test_iteration_budget_spans_the_interrupt(self):
        """Budget counts the whole logical call, not each replay leg."""
        track, _ = _tracking_tool()
        gated, gate = _gated_tool()
        # 3 calls total (track, track, gated) with budget 3: must complete.
        m = MontyCodeInterpreterMiddleware(ptc=[track, gated], iteration_budget=3)
        store = InMemoryStore()

        code = 'track("a")\ntrack("b")\ngated("x")["ok"]'
        with pytest.raises(GraphInterrupt):
            _sync_call(m, code=code, runtime=_hitl_runtime(store))
        assert store.get(_NAMESPACE, _TOOL_CALL_ID).value["host_calls"] == 2

        gate["approved"] = True
        r = _sync_call(m, code=code, runtime=_hitl_runtime(store))
        assert r["error"] is None
        assert r["result"] == "x"

    def test_no_store_falls_back_to_replay(self):
        """Without a store the original replay model applies: prior host
        calls re-run on the replayed execution (idempotency caveat)."""
        track, calls = _tracking_tool()
        gated, gate = _gated_tool()
        m = MontyCodeInterpreterMiddleware(ptc=[track, gated])

        with pytest.raises(GraphInterrupt):
            _sync_call(m, code=self.CODE, runtime=_hitl_runtime(store=None))
        gate["approved"] = True
        r = _sync_call(m, code=self.CODE, runtime=_hitl_runtime(store=None))

        assert r["error"] is None
        assert r["result"] == "x"
        assert calls == ["a", "a"]  # replayed — the documented fallback

    def test_second_interrupt_overwrites_record(self):
        """A later interrupt within the same logical call overwrites the
        snapshot and advances the answered-interrupt bookkeeping.

        Stops after the second interrupt: resuming past it burns the
        positional counter via ``interrupt()``, which needs a real graph
        context — that final leg is covered by
        ``TestSnapshotResumeEndToEnd.test_two_interrupts_one_snippet_no_replay``.
        """
        track, calls = _tracking_tool()
        gate1_tool, gate1 = _gated_tool("gate1")
        gate2_tool, gate2 = _gated_tool("gate2")
        m = MontyCodeInterpreterMiddleware(ptc=[track, gate1_tool, gate2_tool])
        store = InMemoryStore()

        code = 'track("a")\ngate1("x")\ngate2("y")["ok"]'
        with pytest.raises(GraphInterrupt):
            _sync_call(m, code=code, runtime=_hitl_runtime(store))
        assert store.get(_NAMESPACE, _TOOL_CALL_ID).value["interrupts_answered"] == 0

        gate1["approved"] = True
        with pytest.raises(GraphInterrupt):
            _sync_call(m, code=code, runtime=_hitl_runtime(store))

        rec = store.get(_NAMESPACE, _TOOL_CALL_ID).value
        # gate1's answer was consumed by its successful re-invocation; the
        # next resume must burn one position before gate2's interrupt.
        assert rec["interrupts_answered"] == 1
        assert calls == ["a"]  # still never replayed
        assert gate1["calls"] == 2  # interrupted, then re-invoked once

    @pytest.mark.asyncio
    async def test_prior_host_calls_not_replayed_with_store_async(self):
        track, calls = _tracking_tool()
        gated, gate = _gated_tool()
        m = MontyCodeInterpreterMiddleware(ptc=[track, gated])
        store = InMemoryStore()

        with pytest.raises(GraphInterrupt):
            await _async_call(m, code=self.CODE, runtime=_hitl_runtime(store))
        assert calls == ["a"]
        assert store.get(_NAMESPACE, _TOOL_CALL_ID) is not None

        gate["approved"] = True
        r = await _async_call(m, code=self.CODE, runtime=_hitl_runtime(store))

        assert r["error"] is None
        assert r["result"] == "x"
        assert calls == ["a"]
        assert store.get(_NAMESPACE, _TOOL_CALL_ID) is None

    @pytest.mark.asyncio
    async def test_stdout_preserved_across_resume_async(self):
        track, _ = _tracking_tool()
        gated, gate = _gated_tool()
        m = MontyCodeInterpreterMiddleware(ptc=[track, gated])
        store = InMemoryStore()

        code = 'print("before")\ntrack("a")\ngated("x")\nprint("after")\n42'
        with pytest.raises(GraphInterrupt):
            await _async_call(m, code=code, runtime=_hitl_runtime(store))

        gate["approved"] = True
        r = await _async_call(m, code=code, runtime=_hitl_runtime(store))
        assert r["error"] is None
        assert r["stdout"] == "before\nafter\n"


class _FakeModel(BaseChatModel):
    """Minimal tool-call-capable fake model (mirrors test_integration.py)."""

    responses: list[AIMessage]
    i: int = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> _FakeModel:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.responses[self.i % len(self.responses)]
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-capable-model"


def _eval_call(code: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            ToolCall(id=str(uuid.uuid4()), name="eval_python", args={"code": code})
        ],
    )


def _last_eval_result(result: dict) -> dict:
    """Parse the eval_python ToolMessage out of a finished agent run."""
    for msg in reversed(result["messages"]):
        if isinstance(msg, ToolMessage) and msg.name == "eval_python":
            return json.loads(msg.content)
    raise AssertionError("no eval_python ToolMessage found")


class TestSnapshotResumeEndToEnd:
    def test_two_interrupts_one_snippet_no_replay(self):
        """Full graph flow with two sequential interrupt() calls: the second
        resume must burn one counter position so each interrupt receives its
        OWN answer, and the side-effecting tool runs exactly once."""
        track, calls = _tracking_tool()

        @tool
        def approve(action: str) -> str:
            """Ask the human to approve an action.

            Returns a JSON object with: answer (str).
            """
            answer = interrupt({"action": action})
            return json.dumps({"answer": answer})

        code = (
            'track("start")\n'
            'a = approve("one")["answer"]\n'
            'b = approve("two")["answer"]\n'
            "[a, b]"
        )
        agent = create_agent(
            model=_FakeModel(responses=[_eval_call(code), AIMessage(content="done")]),
            tools=[],
            middleware=[MontyCodeInterpreterMiddleware(ptc=[track, approve])],
            checkpointer=InMemorySaver(),
            store=InMemoryStore(),
        )
        config = {"configurable": {"thread_id": "t-e2e"}}

        res = agent.invoke({"messages": [HumanMessage("go")]}, config)
        assert "__interrupt__" in res
        assert res["__interrupt__"][0].value == {"action": "one"}

        res = agent.invoke(Command(resume="yes-one"), config)
        assert "__interrupt__" in res
        assert res["__interrupt__"][0].value == {"action": "two"}

        res = agent.invoke(Command(resume="yes-two"), config)
        assert "__interrupt__" not in res

        payload = _last_eval_result(res)
        assert payload["error"] is None
        # Each interrupt got its own answer — the counter burn worked.
        assert payload["result"] == ["yes-one", "yes-two"]
        # The pre-interrupt host call ran exactly once across three legs.
        assert calls == ["start"]

    def test_single_interrupt_end_to_end(self):
        track, calls = _tracking_tool()

        @tool
        def approve(action: str) -> str:
            """Ask the human to approve an action.

            Returns a JSON object with: answer (str).
            """
            answer = interrupt({"action": action})
            return json.dumps({"answer": answer})

        code = 'track("start")\napprove("only")["answer"]'
        agent = create_agent(
            model=_FakeModel(responses=[_eval_call(code), AIMessage(content="done")]),
            tools=[],
            middleware=[MontyCodeInterpreterMiddleware(ptc=[track, approve])],
            checkpointer=InMemorySaver(),
            store=InMemoryStore(),
        )
        config = {"configurable": {"thread_id": "t-e2e-single"}}

        res = agent.invoke({"messages": [HumanMessage("go")]}, config)
        assert "__interrupt__" in res

        res = agent.invoke(Command(resume="approved"), config)
        payload = _last_eval_result(res)
        assert payload["error"] is None
        assert payload["result"] == "approved"
        assert calls == ["start"]
