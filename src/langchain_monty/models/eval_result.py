from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvalError(BaseModel):
    """Structured error payload returned in ``EvalPythonResult.error``.

    Covers three distinct failure classes the LLM should be able to tell
    apart and respond to differently:

    - Parse/compile errors raised when constructing the ``Monty(...)``
      object — typically syntax or Monty-unsupported-feature errors
      (e.g. classes, currently). The agent should fix the code.
    - Resource-exhaustion errors raised by Monty during execution
      (duration, memory, stack, allocations). The agent should reduce
      scope.
    - The interpreter-budget overflow we enforce ourselves
      (``IterationBudgetExceeded``). The agent is in a host-call loop and
      should restructure.
    """

    type: str = Field(description="Exception/error class name.")
    message: str = Field(description="Human-readable error message.")

    model_config = ConfigDict(extra="forbid")


class EvalPythonResult(BaseModel):
    """Structured return shape for the ``eval_python`` tool.

    Returned as a plain dict to the agent via ``.model_dump()`` so the
    LLM sees regular JSON — but the host code paths build and validate
    these models explicitly. The dict shape the LLM sees is identical to
    the previous version; the change is purely a host-side
    structural-validation tightening.
    """

    result: Any = Field(
        default=None,
        description=(
            "Value of the final expression in the submitted code, if any. "
            "``None`` when execution finished without a trailing expression "
            "or when ``error`` is set."
        ),
    )
    stdout: str = Field(
        default="",
        description="Captured stdout from the sandboxed run.",
    )
    error: EvalError | None = Field(
        default=None,
        description="Structured error payload, or ``None`` on success.",
    )

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
