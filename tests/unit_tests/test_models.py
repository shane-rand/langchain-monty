import pytest
from pydantic import ValidationError

from langchain_monty.models import EvalError, EvalPythonResult, MontyLimits


class TestEvalError:
    def test_construction(self):
        err = EvalError(type="SyntaxError", message="invalid syntax")
        assert err.type == "SyntaxError"
        assert err.message == "invalid syntax"

    def test_model_dump(self):
        err = EvalError(type="RuntimeError", message="boom")
        assert err.model_dump() == {
            "type": "RuntimeError",
            "message": "boom",
            "traceback": None,
        }

    def test_traceback_field(self):
        # Monty runtime errors and type-check failures populate traceback
        # with formatted diagnostics; default is None.
        err = EvalError(
            type="ZeroDivisionError",
            message="division by zero",
            traceback="Traceback (most recent call last):\n ...",
        )
        assert err.traceback.startswith("Traceback")

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            EvalError(type="Foo", message="bar", extra="oops")


class TestEvalPythonResult:
    def test_defaults(self):
        r = EvalPythonResult()
        assert r.result is None
        assert r.stdout == ""
        assert r.error is None

    def test_with_result_and_stdout(self):
        r = EvalPythonResult(result=42, stdout="hello\n")
        assert r.result == 42
        assert r.stdout == "hello\n"
        assert r.error is None

    def test_with_error(self):
        err = EvalError(type="SyntaxError", message="oops")
        r = EvalPythonResult(error=err)
        assert r.error == err
        assert r.result is None

    def test_model_dump_shape(self):
        r = EvalPythonResult(result="val", stdout="out")
        d = r.model_dump()
        assert set(d.keys()) == {"result", "stdout", "error", "attempted_code"}
        assert d["result"] == "val"
        assert d["stdout"] == "out"
        assert d["error"] is None
        assert d["attempted_code"] is None

    def test_error_nested_in_dump(self):
        err = EvalError(type="E", message="m")
        r = EvalPythonResult(error=err)
        d = r.model_dump()
        assert d["error"] == {"type": "E", "message": "m", "traceback": None}

    def test_with_attempted_code(self):
        err = EvalError(type="SyntaxError", message="oops")
        r = EvalPythonResult(error=err, attempted_code="bad code")
        d = r.model_dump()
        assert d["attempted_code"] == "bad code"

    def test_attempted_code_none_on_success(self):
        r = EvalPythonResult(result=42)
        assert r.attempted_code is None

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            EvalPythonResult(result=1, unknown="x")

    def test_arbitrary_result_type(self):
        r = EvalPythonResult(result={"nested": [1, 2, 3]})
        assert r.result == {"nested": [1, 2, 3]}


class TestMontyLimits:
    def test_defaults(self):
        lim = MontyLimits()
        assert lim.max_duration_secs == 5.0
        assert lim.max_memory_bytes == 64 * 1024 * 1024
        assert lim.max_stack_depth == 256
        assert lim.max_allocations == 1_000_000

    def test_custom_values(self):
        lim = MontyLimits(
            max_duration_secs=10.0,
            max_memory_bytes=128,
            max_stack_depth=64,
            max_allocations=500,
        )
        assert lim.max_duration_secs == 10.0
        assert lim.max_memory_bytes == 128
        assert lim.max_stack_depth == 64
        assert lim.max_allocations == 500

    def test_to_monty_returns_dict(self):
        lim = MontyLimits()
        rl = lim.to_monty()
        assert isinstance(rl, dict)

    def test_to_monty_passes_values(self):
        lim = MontyLimits(
            max_duration_secs=3.0,
            max_memory_bytes=1024,
            max_stack_depth=128,
            max_allocations=9999,
        )
        rl = lim.to_monty()
        assert rl["max_duration_secs"] == 3.0
        assert rl["max_memory"] == 1024
        assert rl["max_recursion_depth"] == 128
        assert rl["max_allocations"] == 9999

    def test_frozen(self):
        lim = MontyLimits()
        with pytest.raises(ValidationError):
            lim.max_duration_secs = 99.0  # type: ignore[misc]

    def test_none_means_unlimited_and_is_omitted(self):
        # Upstream ResourceLimits is a total=False TypedDict where an absent
        # key means "no limit"; None fields must therefore be dropped from
        # the dict rather than forwarded.
        lim = MontyLimits(max_duration_secs=None, max_memory_bytes=None)
        rl = lim.to_monty()
        assert "max_duration_secs" not in rl
        assert "max_memory" not in rl
        assert rl["max_recursion_depth"] == 256  # defaults still forwarded

    def test_gc_interval_default_omitted(self):
        # gc_interval defaults to None (keep Monty's internal default) and
        # is only forwarded when explicitly set.
        assert "gc_interval" not in MontyLimits().to_monty()
        assert MontyLimits(gc_interval=5000).to_monty()["gc_interval"] == 5000
