"""HITL snapshot persistence for eval_python replay across GraphInterrupt.

When a host tool raises GraphInterrupt, LangGraph replays the eval_python
call. These helpers persist the paused VM into the agent's BaseStore so the
replay resumes from the pause point instead of re-running every host call.
All helpers degrade silently — a failing backend must never mask the
GraphInterrupt that LangGraph needs to pause the graph.
"""

from __future__ import annotations

import base64

from langchain.tools import ToolRuntime
from langgraph.types import interrupt
from pydantic_monty import FunctionSnapshot

from langchain_monty.models import LangchainStoreMontySnapshot


def _snapshot_namespace(runtime: ToolRuntime) -> tuple[str, str, str]:
    """Store namespace scoped per conversation thread."""
    config = getattr(runtime, "config", None) or {}
    configurable = config.get("configurable") or {}
    thread_id = configurable.get("thread_id")
    return (
        "langchain_monty",
        "snapshots",
        str(thread_id) if thread_id is not None else "default",
    )


def _hitl_store_key(runtime: ToolRuntime) -> str | None:
    """Store key for this eval_python call; ``tool_call_id`` is stable across replay."""
    tool_call_id = getattr(runtime, "tool_call_id", None)
    return tool_call_id if isinstance(tool_call_id, str) and tool_call_id else None


def _make_hitl_record(
    progress: FunctionSnapshot,
    stdout_text: str,
    host_calls: int,
    interrupts_answered: int,
) -> LangchainStoreMontySnapshot:
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
    except Exception:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001
        return None


def _delete_hitl_record(runtime: ToolRuntime) -> None:
    """Remove the snapshot record once the call finishes."""
    store = getattr(runtime, "store", None)
    key = _hitl_store_key(runtime)
    if store is None or key is None:
        return
    try:
        store.delete(_snapshot_namespace(runtime), key)
    except Exception:  # noqa: BLE001
        pass


async def _adelete_hitl_record(runtime: ToolRuntime) -> None:
    """Async twin of _delete_hitl_record."""
    store = getattr(runtime, "store", None)
    key = _hitl_store_key(runtime)
    if store is None or key is None:
        return
    try:
        await store.adelete(_snapshot_namespace(runtime), key)
    except Exception:  # noqa: BLE001
        pass


def _burn_answered_interrupts(count: int) -> None:
    """Advance LangGraph's positional interrupt counter past answered interrupts.

    ``interrupt()`` matches resume values positionally. A snapshot-resumed run
    skips host tools that already ran, so without this correction the
    re-invoked call's ``interrupt()`` would land on index 0 and receive
    the first recorded answer instead of its own.
    """
    for _ in range(count):
        interrupt(
            "langchain-monty internal: interrupt-counter sync for snapshot "
            "resume — if you are seeing this, please report a bug"
        )
