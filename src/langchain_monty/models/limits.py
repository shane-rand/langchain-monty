from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from pydantic_monty import ResourceLimits


class MontyLimits(BaseModel):
    """Resource ceilings forwarded to ``pydantic_monty.ResourceLimits``.

    Naming note: two field names deliberately differ from the upstream
    ``ResourceLimits`` TypedDict for clarity at this package's API surface —
    ``max_memory_bytes`` (upstream ``max_memory``: the unit is otherwise
    ambiguous) and ``max_stack_depth`` (upstream ``max_recursion_depth``).
    ``to_monty()`` performs the mapping, so the rename never leaks into
    Monty itself.

    Semantics: upstream, *omitting* a key disables that limit. We mirror
    that by treating ``None`` as "no limit" — a ``None`` field is simply not
    forwarded. Defaults are conservative; tune up for heavier code-mode
    workloads, or set a field to ``None`` to lift its cap entirely.
    """

    model_config = ConfigDict(frozen=True)

    max_duration_secs: float | None = 5.0
    """Wall-clock budget per eval_python call. ``None`` = unlimited."""

    max_memory_bytes: int | None = 64 * 1024 * 1024
    """Sandbox heap cap in bytes. ``None`` = unlimited."""

    max_stack_depth: int | None = 256
    """Sandbox recursion limit. ``None`` = unlimited."""

    max_allocations: int | None = 1_000_000
    """Total allocation count budget. ``None`` = unlimited."""

    gc_interval: int | None = None
    """Allocations between sandbox GC cycles; ``None`` keeps Monty's default.

    Exposed for parity with upstream ``ResourceLimits.gc_interval`` — only
    relevant when tuning long-running, allocation-heavy snippets.
    """

    def to_monty(self) -> ResourceLimits:
        """Translate to the upstream TypedDict, dropping disabled limits.

        ``ResourceLimits`` is ``total=False``: a key that is absent means
        "no limit", so ``None`` fields are filtered out rather than passed
        through (passing ``None`` explicitly would be a type error on the
        Rust side).
        """
        mapped: dict = {
            "max_duration_secs": self.max_duration_secs,
            "max_memory": self.max_memory_bytes,
            "max_recursion_depth": self.max_stack_depth,
            "max_allocations": self.max_allocations,
            "gc_interval": self.gc_interval,
        }
        return ResourceLimits(**{k: v for k, v in mapped.items() if v is not None})
