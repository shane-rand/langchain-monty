# langchain-monty

LangChain agent middleware that adds an `eval_python` tool backed by [pydantic-monty](https://github.com/pydantic/monty) — Pydantic's Rust-implemented, sandboxed Python interpreter.

The interpreter runs in a pooled `monty` worker subprocess and has zero access to the host filesystem, network, or environment. The only way code running inside the sandbox can reach the outside world is through host tools you explicitly allowlist via the `ptc=` parameter.

Running out of process is what makes the sandbox safe to point at model-written code: a crash no memory-safe interpreter can fully rule out — stack overflow, allocator abort — kills a replaceable worker, and your process is never at risk.

Works with any LangChain v1 agent (`langchain.agents.create_agent`) and with [deepagents](https://github.com/langchain-ai/deepagents) (`create_deep_agent`) — there is no runtime dependency on deepagents. This is the Python analog of `langchain-quickjs`, which does the same thing with a QuickJS JavaScript VM.


## Installation

```bash
uv add langchain-monty
```

Requires Python 3.12+.

`pydantic-monty` pulls in `pydantic-monty-runtime`, which installs the `monty`
binary into your environment's scripts directory. The worker pool finds it there,
on `PATH`, or at `$MONTY_BIN` — worth knowing if you vendor dependencies, build a
slim container image, or install with `--no-binary`.

## Quick start

```python
from langchain.agents import create_agent
from langchain_monty import MontyCodeInterpreterMiddleware

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    middleware=[MontyCodeInterpreterMiddleware()],
)

result = agent.invoke({"messages": [{"role": "user", "content": "What is 2 ** 32?"}]})
```

The middleware adds an `eval_python` tool to the agent and appends a usage guide to the system prompt. The agent can call `eval_python` with any Python code; the result of the final expression is returned, along with any captured stdout.

With deepagents, pass the middleware to `create_deep_agent` the same way:

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    middleware=[MontyCodeInterpreterMiddleware()],
)
```

## Programmatic tool calling (ptc)

By default the interpreter is pure-compute: it has no access to host tools. Pass `ptc=` with a list of `BaseTool` objects and/or `str` tool names to expose those tools inside the sandbox:

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

### Deferred tool names

`ptc` entries can also be plain strings. String entries register the name in the allowlist but are resolved at runtime from `runtime.tools` — useful for tools injected by other middleware (e.g. `FilesystemMiddleware` contributes `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`):

```python
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    middleware=[
        MontyCodeInterpreterMiddleware(
            ptc=[my_api_tool, "read_file", "ls", "grep"],
        ),
    ],
)
```

`BaseTool` entries have their schemas shown in the system prompt immediately. `str` entries are listed as runtime-resolved; on every model call the middleware checks the tools bound to the request and, once a deferred name resolves to a real tool, renders its full signature and docstring into the system prompt dynamically.

Inside the sandbox, the agent can now write:

```python
results = search("LangGraph 0.6 release notes")
[r["title"] for r in results if "breaking" in r["title"].lower()]
```

Each host-tool call surfaces on the Python side as a `FunctionSnapshot`. The middleware drives an event loop — invoking the LangChain tool through its normal machinery as a full `ToolCall` (so tracing, retries, and injected parameters all work), then resuming Monty with the result. Tools not in the allowlist return an error to the interpreter rather than executing.

Tools that declare injected parameters work through the bridge: the live `ToolRuntime` (and its `state`/`store`) is forwarded into any `runtime: ToolRuntime`, `InjectedState`, or `InjectedStore` slot the tool declares, and `InjectedToolCallId` parameters receive a synthetic id prefixed `eval_python:` so bridged calls are recognizable in traces. Sandbox code can never forge these values — interpreter-supplied kwargs matching injected names are stripped before the real ones are added. The one unsupported shape is `Command`-returning tools (e.g. deepagents' `task`): a `Command` mutates graph state and can only be applied by the agent's own tool node, so calling one from inside `eval_python` raises a clear error telling the agent to call that tool directly instead.

### Call styles: plain vs concurrent

Host functions support two call styles inside the sandbox, and both behave identically under `invoke` and `ainvoke`:

```python
# Plain — calls resolve one at a time
hits = search("a")

# Concurrent — independent calls run in parallel (under ainvoke)
import asyncio

async def go():
    return await asyncio.gather(search("a"), search("b"))

asyncio.run(go())
```

The two styles cannot be mixed in one snippet (Monty's pause/resume protocol forces the host to answer each call as either a value or a future before knowing whether the sandbox will await it). The middleware handles this adaptively: it first runs the code in deferred mode, and if the code turns out to use plain calls it transparently restarts in eager mode — safe, because deferred mode executes no host tools until the sandbox awaits. Code that awaits some calls but discards others gets a structured `UnawaitedHostCallError` telling the agent to pick one style.

### Static type checking against tool schemas

Before executing anything, the submitted code is type-checked by Monty's built-in static checker against stub signatures generated from the allowlisted tools' JSON schemas. A hallucinated keyword argument, a wrong argument type, or a misspelled parameter comes back instantly as a structured `TypeCheckError` with `file:line:col` diagnostics — no execution, no wasted host-tool calls:

```json
{
  "result": null,
  "stdout": "",
  "error": {
    "type": "TypeCheckError",
    "message": "static type check failed before execution; no code was run",
    "traceback": "main.py:1:18: error[unknown-argument] Argument `limit` does not match any known parameter of function `search`"
  },
  "attempted_code": "search(query=\"x\", limit=5)"
}
```

Disable with `MontyCodeInterpreterMiddleware(type_check=False)` if Monty's checker (a strict subset of Python's type system) rejects code you need to run. Deferred tool names that haven't resolved yet get permissive `(*args, **kwargs)` stubs, so they never fail the static check.

### Human-in-the-loop and interrupts

When a bridged host tool raises `GraphInterrupt` (e.g. `HumanInTheLoopMiddleware` asking for approval), the middleware re-raises it instead of feeding it into the sandbox, so LangGraph checkpoints and pauses normally. What happens on resume depends on whether the agent has a [LangGraph store](https://langchain-ai.github.io/langgraph/concepts/persistence/#memory-store):

**With a store** (`create_agent(..., store=...)`), the paused Monty VM is serialized (`FunctionSnapshot.dump()`) into the store at interrupt time, keyed by the tool call id. When LangGraph replays the `eval_python` call, the snapshot is revived (`MontySession.load_snapshot()`, on a freshly checked-out session) and execution **continues from the interrupted host call**: host tools that already ran are *not* re-invoked, stdout printed before the pause is preserved, and the iteration budget keeps counting across the pause. Only the interrupted tool itself is re-invoked — its `interrupt()` then returns the recorded human answer. The snapshot record is deleted from the store when the call finishes. Multiple sequential interrupts within one snippet are supported.

**Without a store**, LangGraph's plain replay model applies: on resume the whole `eval_python` call re-runs from the top, so host tools called *before* the interrupt point are re-invoked — combine HITL with idempotent tools in this mode.

Scope notes for snapshot-resume: it covers the plain-call execution path; an interrupt escaping an awaited `asyncio.gather` batch falls back to full replay. A single host tool that calls `interrupt()` more than once per invocation is not supported by the resume bookkeeping (one interrupt per tool call — the `HumanInTheLoopMiddleware` shape — is fully supported). Persistence failures degrade silently to the replay model, never to a broken run.

## Building tools for the sandbox

The LLM writes code before it has seen any data. Argument names and types are enforced by the static type check, but the **only** signal the model has about what a host function *returns* is the tool's docstring, which the middleware surfaces verbatim in the system prompt. Following these conventions keeps generated code correct on the first attempt.

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

Use `MontyLimits` to control per-call resource budgets. Setting any field to `None` disables that limit (mirroring upstream `ResourceLimits`, where an omitted key means "no limit"):

```python
from langchain_monty import MontyCodeInterpreterMiddleware, MontyLimits

limits = MontyLimits(
    max_duration_secs=10.0,       # wall-clock time (default 5.0)
    max_memory_bytes=128_000_000, # heap cap (default 64 MB)
    max_stack_depth=512,          # recursion limit (default: Monty's 1000)
    max_suspensions=2_000,        # pauses per call (default: Monty's 1000)
    gc_interval=None,             # allocations between GCs (default: Monty's)
)

middleware = MontyCodeInterpreterMiddleware(limits=limits)
```

`max_suspensions` counts every pause the sandbox makes the host answer: host-tool
calls, OS callbacks (`pathlib`, the clock), undefined-name lookups, and future
resolutions. It is strictly wider than `iteration_budget`, which caps host-tool
calls alone — filesystem-heavy code can exhaust it without calling a single host
tool. Monty's pool enforces it and aborts an over-budget run.

Naming note: `max_memory_bytes` and `max_stack_depth` map to upstream `ResourceLimits.max_memory` and `.max_recursion_depth`; `MontyLimits.to_monty()` performs the translation. `max_stack_depth` and `max_suspensions` are the two fields `None` does not make unlimited — upstream refuses to disable them, so `None` leaves Monty's default of 1000 in place.

## Constructor reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ptc` | `Sequence[BaseTool \| str] \| None` | `None` | Tools the interpreter may call. `BaseTool` entries are available immediately — their schemas appear in the system prompt. `str` entries are deferred: the name is registered in the allowlist and resolved at runtime from the agent's bound tools (useful for tools injected by other middleware); their schemas are rendered into the system prompt dynamically once resolved. `None` means pure-compute only. |
| `limits` | `MontyLimits \| None` | `None` | Per-call resource budgets. Uses defaults when `None`. |
| `system_prompt` | `str \| None` | Built-in block | System-prompt block appended to every model call. Pass `None` to keep the tool but add no prompt text — host-function schemas then move into the tool description so the model still sees them. |
| `tool_description` | `str \| None` | Built-in template | Description rendered on the `eval_python` tool. Supports `{available_host_tools}`, `{max_duration_secs}`, `{max_memory_bytes}`, `{max_stack_depth}` placeholders. |
| `iteration_budget` | `int` | `64` | Hard cap on host-tool **calls** per `eval_python` call (a `gather` fan-out of N counts N). Exceeding it returns an `IterationBudgetExceeded` error. |
| `type_check` | `bool` | `True` | Statically type-check submitted code against stubs generated from the allowlisted tools' schemas before executing. Failures return a `TypeCheckError` with line-precise diagnostics. |

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
    "message": "division by zero",
    "traceback": "Traceback (most recent call last):\n  File \"main.py\", line 1, in <module>\n    1 / 0\n    ~~~\nZeroDivisionError: division by zero"
  },
  "attempted_code": "1 / 0"
}
```

`error.type` is the real exception class the sandbox raised (unwrapped from Monty's wrapper), `error.traceback` carries a CPython-style traceback with line numbers and source previews when available, and `attempted_code` is populated only when `error` is set.

If the final expression's value can't be expressed in plain JSON (tuples serialize as arrays, but e.g. sets and dataclasses can't), the result falls back to Monty's tagged natural form — `{"$set": [1, 2, 3]}`, `{"$dataclass": {...}, "name": "..."}` — so it always survives message serialization losslessly.

Error classes the agent can act on differently:

- **`SyntaxError`** — parse or unsupported-feature errors (e.g. classes). The agent should fix the code; nothing was executed.
- **`TypeCheckError`** — the static pre-flight check failed (bad host-function arguments). Nothing was executed; `traceback` has per-line diagnostics.
- **Runtime errors** — the real sandbox exception class (`KeyError`, `ZeroDivisionError`, ...) including resource exhaustion. The agent should fix the logic or reduce scope.
- **`IterationBudgetExceeded`** — too many host-tool calls in one invocation. The agent should restructure its code.
- **`UnawaitedHostCallError`** — the code mixed awaited and plain host-call styles. The agent should pick one style.

## Sandbox capabilities

Monty implements a Python subset. Currently supported stdlib modules:

`sys`, `os`, `typing`, `asyncio`, `re`, `datetime`, `json`, `dataclasses`

Not supported (yet): class definitions, real imports beyond the listed modules.

The sandbox has no access to the host filesystem, network, subprocesses, or environment variables. All communication with the outside world goes through explicitly allowlisted host tools.

(The middleware itself runs the VM in a worker subprocess, which is a separate matter: sandbox code can neither see nor spawn processes. Its `pathlib` calls land in an empty in-memory filesystem, and `os.getenv` in an empty environment.)

## Async support

The tool is always called `eval_python`. Internally the middleware registers both a sync and an async implementation; LangChain dispatches to the async path automatically when you use `agent.ainvoke(...)`:

```python
result = await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})
```

The async path is event-loop friendly: it drives `AsyncMonty` directly, so every VM step is async IPC with the worker rather than a blocking call parked on a thread, and a compute-heavy snippet never stalls other coroutines in your server. Sandbox code using `asyncio.gather` over host calls gets true host-side concurrency under `ainvoke` (and falls back to sequential execution under `invoke`).

## Upgrading to 3.0

3.0 moves the package onto `pydantic-monty` 0.0.23, which restructured the interpreter
from a single in-process extension into a client package plus a `monty` runtime binary
that executes code in worker subprocesses. Three things change for callers:

**`MontyLimits.max_allocations` is gone.** Upstream `ResourceLimits` dropped the key and
now rejects it outright, so it could not be kept as a no-op without silently ignoring a
limit you asked for. Use `max_suspensions` to bound how much work a single `eval_python`
call can push back at the host, and `max_duration_secs` / `max_memory_bytes` for the caps
it used to stand in for. Every other field is unchanged.

**The `monty` binary is a real install-time artifact.** It arrives with
`pydantic-monty-runtime` and is found in your environment's scripts directory, on `PATH`,
or at `$MONTY_BIN`. A normal `uv add` / `pip install` needs no action; vendored
dependencies, `--no-binary` installs and hand-built container images do.

**Execution is out of process.** Startup costs a few milliseconds per call rather than
microseconds — noise next to a model round trip — and in exchange a sandbox crash takes
a replaceable worker with it instead of your process. The middleware's own API, the
`eval_python` tool, its return shape, and the HITL snapshot format are unchanged; code
that only touches `MontyCodeInterpreterMiddleware` needs no edits beyond the limits field.

## Development

```bash
# Install with dev dependencies (deepagents is dev-only, used by the
# integration tests; the library itself does not depend on it)
uv sync

# Run tests
uv run pytest

# Lint
uv run ruff check src tests
```

## License

See [LICENSE](LICENSE).
