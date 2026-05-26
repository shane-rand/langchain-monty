# langchain-monty

LangChain middleware that gives a [deepagents](https://github.com/langchain-ai/deepagents) agent an `eval_python` tool backed by [pydantic-monty](https://github.com/pydantic/monty) — Pydantic's Rust-implemented, sandboxed Python interpreter.

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

By default the interpreter is pure-compute: it has no access to host tools. Pass `ptc=` with a list of `BaseTool` objects to expose those tools inside the sandbox:

```python
from langchain_core.tools import tool
from deepagents import create_deep_agent
from langchain_monty import MontyCodeInterpreterMiddleware

@tool
async def search(query: str) -> str:
    """Search the document index.

    Returns a JSON array of results. Each result is a dict with:
      - title (str): document title
      - url (str): source URL
      - snippet (str): matching excerpt
    """
    ...

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search],
    middleware=[MontyCodeInterpreterMiddleware(ptc=[search])],
)
```

Inside the sandbox, the agent can now write:

```python
results = search("LangGraph 0.6 release notes")
[r["title"] for r in results if "breaking" in r["title"].lower()]
```

Each host-tool call surfaces on the Python side as a `FunctionSnapshot`. The middleware drives an event loop — invoking the LangChain tool through its normal machinery (so `HumanInTheLoopMiddleware`, retries, traces, and `Command`-returning tools all keep working), then resuming Monty with the result. Tools not in the allowlist return an error to the interpreter rather than executing.

## Building tools for the sandbox

Monty has no type introspection and the LLM writes code before it has seen any data. The **only** signal it has about what a host function returns is the tool's docstring, which the middleware surfaces verbatim in both the system prompt and the `eval_python` tool description. Following these conventions keeps generated code correct on the first attempt.

### 1. Document the return shape precisely

Name every field, give its type, and note optional or nullable fields. Vague descriptions produce hallucinated field names and silent empty results.

```python
# Bad — the LLM will guess field names and get them wrong
@tool
async def get_compensation_history() -> str:
    """Retrieve salary history records."""
    ...

# Good — the LLM knows exactly what to expect
@tool
async def get_compensation_history() -> str:
    """
    Retrieve salary change history for all employees.

    Returns a JSON array. Each record contains:
      - employee_id (str): matches employee_id in the roster
      - effective_year (int): year the change took effect
      - previous_salary (float): salary before the change
      - new_salary (float): salary after the change
      - raise_pct (float): percentage change (can be negative)
      - rating_at_time (float | null): performance rating that drove the raise
    """
    ...
```

### 2. Return JSON-serializable data

Return `str` (a JSON-encoded payload) or a plain Python type (`list`, `dict`, `int`, `float`, `bool`, `None`). Pydantic models, dataclasses, and other objects will be passed through `json.dumps` / `json.loads` before Monty receives them, which may lose information or raise if the object is not serializable.

```python
# Preferred — explicit JSON encoding, no surprises
@tool
async def get_employee_roster() -> str:
    records = fetch_employees()
    return json.dumps([r.model_dump() for r in records])
```

### 3. Name join keys explicitly

When multiple tools return related datasets, call out the join key in every docstring. The LLM needs to know which field to use without inspecting actual data.

```python
"""...
Join with get_compensation_history() on employee_id.
"""
```

### 4. Document edge cases

Note nulls, mixed currencies, date formats, and any filtering the tool applies (e.g. active-only). Silent nulls in generated code produce `population_n: 0` results with no error.

```python
"""...
- currency (str): ISO 4217 code; records may mix currencies — normalize
  before computing ratios across the full population.
- is_active (bool): False records are included; filter with
  `[e for e in roster if e['is_active']]` if you only want current employees.
"""
```

### 5. Keep field names stable

The LLM hard-codes field names in generated code. Renaming a field is a silent, undetectable breakage — code runs without error but produces empty or wrong results because `.get('old_name')` returns `None`.

### Full example

```python
import json
from langchain_core.tools import tool
from langchain_monty import MontyCodeInterpreterMiddleware

@tool
async def get_employee_roster() -> str:
    """
    Retrieve the full employee roster.

    Returns a JSON array. Each record contains:
      - employee_id (str): unique identifier, join key for all other datasets
      - department (str): e.g. "Engineering", "Sales"
      - title (str): job title
      - seniority_level (int): 0 (IC) – 3 (VP)
      - hire_date (str): ISO 8601 date
      - location (str): office city
      - gender (str | null): self-reported; null if not disclosed
      - age (int): age in years at last review cycle
      - current_salary (float): USD annual base salary
      - manager_id (str | null): employee_id of direct manager
      - is_active (bool): False for departed employees
    """
    return json.dumps(fetch_roster())

middleware = MontyCodeInterpreterMiddleware(ptc=[get_employee_roster])
```

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
| `ptc` | `Sequence[BaseTool] \| None` | `None` | Tool objects the interpreter may call. The middleware extracts names for the allowlist and stores the objects for direct invocation. `None` means pure-compute only. |
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
