from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvalError(BaseModel):
    """Structured error payload returned in ``EvalPythonResult.error``.

    Covers four distinct failure classes the LLM should be able to tell
    apart and respond to differently:

    - Parse/compile errors (``type="SyntaxError"``) raised when Monty
      parses the code — typically syntax or Monty-unsupported-feature
      errors (e.g. classes, currently). The agent should fix the code.
    - Pre-execution static type-check failures
      (``type="TypeCheckError"``) — the code referenced a host function
      with a hallucinated keyword argument or a wrong argument type.
      Nothing was executed; ``traceback`` carries per-line diagnostics.
    - Runtime errors inside the sandbox, including resource exhaustion
      (duration, memory, stack, allocations). ``type`` is the *real*
      exception class the sandbox raised (e.g. ``ZeroDivisionError``)
      and ``traceback`` is a full CPython-style traceback with line
      numbers. The agent should fix the logic or reduce scope.
    - The interpreter-budget overflow we enforce ourselves
      (``IterationBudgetExceeded``). The agent is in a host-call loop and
      should restructure.
    """

    type: str = Field(description="Exception/error class name.")
    message: str = Field(description="Human-readable error message.")
    traceback: str | None = Field(
        default=None,
        description=(
            "Formatted diagnostics when Monty can provide them: a full "
            "CPython-style traceback (line numbers + source preview) for "
            "sandbox runtime errors, or per-line ``file:line:col`` "
            "diagnostics for static type-check failures. ``None`` when no "
            "richer rendering exists (e.g. host-side errors)."
        ),
    )

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
    attempted_code: str | None = Field(
        default=None,
        description=(
            "The code that was submitted when the error occurred. "
            "Populated only when ``error`` is set, to aid debugging."
        ),
    )

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
