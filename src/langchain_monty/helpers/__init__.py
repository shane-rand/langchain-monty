"""Monty payload helpers."""

import base64
import json
from typing import Any

from pydantic_monty import ExternalResult, MontyComplete


def make_return_value_result(value: Any) -> ExternalResult:
    """Build a successful ``ExternalResult`` dict for ``snapshot.resume()``.

    Monty's ``resume()`` accepts ``ExternalResult``, a union of TypedDict
    variants. ``{"return_value": ...}`` is the success form: Monty unwraps
    the value on the interpreter side and the sandbox call returns it.
    """
    return {"return_value": value}


def make_exception_result(exc: Exception) -> ExternalResult:
    """Build an exception ``ExternalResult`` dict for ``snapshot.resume()``.

    Two ways exist to signal an exception to Monty:

    - ``{"exc_type": "...", "message": "..."}`` only accepts a fixed set of
      builtin exception names; framework types (``ToolException``, ...) are
      rejected with ``TypeError: Unknown exception type``.
    - ``{"exception": exc}`` passes the real Python exception object and
      lets Monty map it internally — works for *any* class, so it's what we
      use for tool failures.

    The sandbox sees the exception as if the host function raised it, so
    interpreter code can ``try/except`` around host calls normally.
    """
    return {"exception": exc}


def jsonable_output(progress: MontyComplete) -> Any:
    """Extract the final sandbox value in a JSON-safe form.

    Monty supports values plain JSON can't express (tuples, sets, bytes,
    dataclasses). The structured tool result eventually becomes a JSON
    ToolMessage, so a non-JSON-able value would blow up far from here, deep
    in message serialization. Strategy:

    - if ``json.dumps`` accepts the raw output, return it untouched (the
      overwhelmingly common case: None/bool/int/float/str/list/dict);
    - otherwise re-encode into a tagged "natural form": rich types come back
      tagged, e.g. ``{"$tuple": [...]}``, ``{"$set": [...]}`` — lossless,
      self-describing, and always serializable.
    """
    output = progress.output  # converted from VM repr on each access — grab once
    try:
        json.dumps(output)
        return output
    except (TypeError, ValueError):
        return _tagged(output)


def _tagged(value: Any) -> Any:
    """Recursively re-encode a value JSON can't express into tagged form.

    Monty hands back native Python objects, so the tagging happens host-side.
    Containers are walked because the offending type is often nested (a list of
    tuples, a dict whose values are sets). Anything unrecognized degrades to its
    ``repr`` rather than raising — a weird result is worth more to the model than
    a serialization failure surfacing far from here, in message encoding.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return {"$tuple": [_tagged(v) for v in value]}
    if isinstance(value, frozenset):
        return {"$frozenset": [_tagged(v) for v in value]}
    if isinstance(value, set):
        return {"$set": [_tagged(v) for v in value]}
    if isinstance(value, (bytes, bytearray)):
        return {"$bytes": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, list):
        return [_tagged(v) for v in value]
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: _tagged(val) for key, val in value.items()}
        # Non-string keys can't be dict keys in JSON; degrade to pairs.
        return {"$dict": [[_tagged(k), _tagged(v)] for k, v in value.items()]}

    # Dataclass instances arrive as ClassInstance / MontyClassProxy, both of
    # which expose the same name/attributes pair.
    name = getattr(value, "name", None)
    attributes = getattr(value, "attributes", None)
    if isinstance(name, str) and isinstance(attributes, dict):
        return {
            "$object": {
                "name": name,
                "attributes": {k: _tagged(v) for k, v in attributes.items()},
            }
        }
    return repr(value)


def deserialize_return_value(value: Any) -> Any:
    """Attempt to JSON-deserialize string return values from host tools.

    LangChain tools typically return JSON-serialized strings. Monty needs
    native Python objects (lists, dicts) so that interpreter code like
    ``roster[0]`` indexes into a list rather than a string.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value
