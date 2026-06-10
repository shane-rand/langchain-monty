"""Persisted state for resuming an interrupted ``eval_python`` call.

When a host tool raises ``GraphInterrupt`` mid-execution (human-in-the-loop
approval), LangGraph checkpoints the graph and later *replays* the whole
``eval_python`` tool call. ``FunctionSnapshot.dump()`` lets us serialize the
paused Monty VM at the interrupt point so the replay can continue from the
interrupted host call instead of re-running the code (and re-invoking every
host tool that already ran).

The snapshot bytes capture everything VM-side. This record carries the bytes
plus the driver-side state that lives *outside* the VM — see the field
descriptions. It is stored in the agent's ``BaseStore`` (``runtime.store``)
keyed by the eval_python tool call id, which is the one identity stable
across LangGraph's interrupt/replay cycle.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LangchainStoreMontySnapshot(BaseModel):
    """One interrupted ``eval_python`` call, ready to be resumed.

    All fields must stay JSON-serializable: ``BaseStore`` backends persist
    record values as JSON (hence base64 for the snapshot bytes).
    """

    monty_snapshot: str = Field(
        description=(
            "Base64-encoded ``FunctionSnapshot.dump()`` bytes — the paused "
            "Monty VM, mid-way through the original run, waiting on the "
            "interrupted host call. Revived with "
            "``pydantic_monty.load_snapshot()``."
        ),
    )
    stdout: str = Field(
        default="",
        description=(
            "Stdout accumulated before the interrupt. The ``print_callback`` "
            "is host-side state and is not part of the snapshot bytes, so "
            "the text collected so far must travel here and be prepended to "
            "whatever the resumed run prints."
        ),
    )
    host_calls: int = Field(
        default=0,
        description=(
            "Host-call count to seed ``iteration_budget`` enforcement in the "
            "resumed run. Stored *excluding* the interrupted call, which the "
            "resumed drive loop counts again when it re-invokes it."
        ),
    )
    interrupts_answered: int = Field(
        default=0,
        description=(
            "How many ``interrupt()`` answers were already consumed within "
            "this eval_python call. LangGraph matches resume values to "
            "``interrupt()`` calls positionally per task execution; the "
            "resumed run skips previously-run host tools, so it must advance "
            "the counter by this many placeholder ``interrupt()`` calls "
            "before re-invoking the interrupted tool."
        ),
    )

    model_config = ConfigDict(extra="forbid")
