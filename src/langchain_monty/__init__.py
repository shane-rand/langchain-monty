"""langchain-monty: a Monty-backed code-interpreter middleware for deepagents.

Python sibling of ``langchain-quickjs``. Provides ``MontyCodeInterpreterMiddleware``,
which contributes an ``eval_python`` tool backed by ``pydantic_monty`` — the
Rust-implemented sandboxed Python interpreter from Pydantic.
"""

from langchain_monty.middleware import (
    CODE_INTERPRETER_SYSTEM_PROMPT,
    CODE_INTERPRETER_TOOL_DESCRIPTION,
    MontyCodeInterpreterMiddleware,
    MontyLimits,
)
from langchain_monty.models import EvalError, EvalPythonResult

__all__ = [
    "MontyCodeInterpreterMiddleware",
    "EvalError",
    "EvalPythonResult",
    "MontyLimits",
    "CODE_INTERPRETER_SYSTEM_PROMPT",
    "CODE_INTERPRETER_TOOL_DESCRIPTION",
]


def main() -> None:
    print("Hello from langchain-monty!")
