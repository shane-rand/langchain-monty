# langchain-monty

LangChain middleware that gives a [deepagents](https://github.com/pydantic/deepagents) agent an `eval_python` tool backed by [pydantic-monty](https://github.com/pydantic/monty) — Pydantic's Rust-implemented, sandboxed Python interpreter.

The interpreter starts in microseconds, runs in-process, and has zero access to the host filesystem, network, or environment. The only way code running inside the sandbox can reach the outside world is through host tools you explicitly allowlist via the `ptc=` parameter.

This is the Python analog of `langchain-quickjs`, which does the same thing with a QuickJS JavaScript VM.

## Installation

```bash
pip install langchain-monty
```

Requires Python 3.12+.

## Quick start

```python
from deepagents import create_deep_agent
from langchain_monty import MontyCodeInterpreterMiddleware

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    middleware=[MontyCodeInterpreterMiddleware()],
)

result = agent.invoke({"messages": [{"role": "user", "content": "What is 2 ** 32?"}]})
```

The middleware adds an `eval_python` tool to the agent and appends a usage guide to the system prompt. The agent can call `eval_python` with any Python code; the result of the final expression is returned, along with any captured stdout.

## Programmatic tool calling (ptc)

By default the interpreter is pure-compute: it has no access to host tools. Pass `ptc=` with a list of tool names to expose those tools inside the sandbox:

```python
from langchain_core.tools import tool
from deepagents import create_deep_agent
from langchain_monty import MontyCodeInterpreterMiddleware

@tool
def search(query: str) -> list[dict]:
    """Search the web."""
    ...

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search],
    middleware=[MontyCodeInterpreterMiddleware(ptc=["search"])],
)
```

Inside the sandbox, the agent can now write:

```python
results = search("LangGraph 0.6 release notes")
[r["title"] for r in results if "breaking" in r["title"].lower()]
```

Each host-tool call surfaces on the Python side as a `FunctionSnapshot`. The middleware drives an event loop — invoking the LangChain tool through its normal machinery (so `HumanInTheLoopMiddleware`, retries, traces, and `Command`-returning tools all keep working), then resuming Monty with the result. Tools not in the allowlist return an error to the interpreter rather than executing.

## Resource limits

Use `MontyLimits` to control per-call resource budgets:

```python
from langchain_monty import MontyCodeInterpreterMiddleware, MontyLimits

limits = MontyLimits(
    max_duration_secs=10.0,      # wall-clock time (default 5.0)
    max_memory_bytes=128_000_000, # heap cap (default 64 MB)
    max_stack_depth=512,          # recursion limit (default 256)
    max_allocations=2_000_000,    # allocation count (default 1 000 000)
)

middleware = MontyCodeInterpreterMiddleware(limits=limits)
```

## Constructor reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ptc` | `Sequence[str] \| None` | `None` | Tool names the interpreter may call. `None` means pure-compute only. |
| `limits` | `MontyLimits \| None` | `None` | Per-call resource budgets. Uses defaults when `None`. |
| `skills_backend` | `BackendProtocol \| BackendFactory \| None` | `None` | Deepagents backend that supplies Monty-compatible Python helpers. Callables are exposed as `skill_<module>_<name>` inside the interpreter. |
| `system_prompt` | `str \| None` | Built-in block | System-prompt block appended to every model call. Pass `None` to keep the tool but add no prompt text. |
| `tool_description` | `str \| None` | Built-in template | Description rendered on the `eval_python` tool. Supports `{available_host_tools}`, `{max_duration_secs}`, `{max_memory_bytes}`, `{max_stack_depth}` placeholders. |
| `iteration_budget` | `int` | `64` | Hard cap on host-tool round-trips per `eval_python` call. Exceeding it returns an `IterationBudgetExceeded` error. |

## Return shape

`eval_python` always returns a JSON object with three fields:

```json
{
  "result": <value of final expression, or null>,
  "stdout": "<captured stdout>",
  "error": null
}
```

On failure:

```json
{
  "result": null,
  "stdout": "",
  "error": {
    "type": "ZeroDivisionError",
    "message": "division by zero"
  }
}
```

Three error classes the agent can act on differently:

- **Parse/compile errors** — syntax or unsupported-feature errors (e.g. classes). The agent should fix the code.
- **Resource-exhaustion errors** — duration, memory, stack, or allocation limits exceeded. The agent should reduce scope.
- **`IterationBudgetExceeded`** — the interpreter made too many host-tool calls in one invocation. The agent should restructure its code.

## Sandbox capabilities

Monty implements a Python subset. Currently supported stdlib modules:

`sys`, `os`, `typing`, `asyncio`, `re`, `datetime`, `json`, `dataclasses`

Not supported (yet): class definitions, real imports beyond the listed modules.

The sandbox has no access to the host filesystem, network, subprocesses, or environment variables. All communication with the outside world goes through explicitly allowlisted host tools.

## Async support

The middleware exposes both sync (`eval_python`) and async (`aeval_python`) variants. Use `agent.ainvoke(...)` to run the async path end-to-end:

```python
result = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src tests
```

## License

See [LICENSE](LICENSE).
