from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from pydantic_monty import ResourceLimits


class MontyLimits(BaseModel):
    """Resource ceilings forwarded to ``pydantic_monty.ResourceLimits``.

    Field names mirror ``ResourceLimits`` so callers can map 1:1. Defaults
    are conservative; tune up for heavier code-mode workloads.
    """

    model_config = ConfigDict(frozen=True)

    max_duration_secs: float = 5.0
    max_memory_bytes: int = 64 * 1024 * 1024
    max_stack_depth: int = 256
    max_allocations: int = 1_000_000

    def to_monty(self) -> ResourceLimits:
        return ResourceLimits(
            max_duration_secs=self.max_duration_secs,
            max_memory=self.max_memory_bytes,
            max_recursion_depth=self.max_stack_depth,
            max_allocations=self.max_allocations,
        )
