from unittest.mock import MagicMock

from langchain_monty.middleware.monty_code_interpreter_middleware import (
    _INJECTED_PARAM_NAMES,  # noqa: PLC2701
    _normalize_call_args,  # noqa: PLC2701
    _visible_schema_fields,  # noqa: PLC2701
)


def _make_snapshot(function_name="search", args=(), kwargs=None):
    snap = MagicMock()
    snap.function_name = function_name
    snap.args = args
    snap.kwargs = kwargs or {}
    return snap


def _make_tool(args_dict):
    tool = MagicMock()
    tool.args = args_dict
    return tool


class TestVisibleSchemaFields:
    def test_returns_non_injected_fields(self):
        tool = _make_tool({"query": {}, "max_results": {}})
        assert _visible_schema_fields(tool) == ["query", "max_results"]

    def test_filters_injected_names(self):
        tool = _make_tool({"query": {}, "runtime": {}, "state": {}})
        result = _visible_schema_fields(tool)
        assert "runtime" not in result
        assert "state" not in result
        assert "query" in result

    def test_all_injected_names_are_filtered(self):
        tool = _make_tool({name: {} for name in _INJECTED_PARAM_NAMES})
        assert _visible_schema_fields(tool) == []

    def test_no_args_attribute(self):
        tool = MagicMock(spec=[])
        assert _visible_schema_fields(tool) == []

    def test_non_dict_args(self):
        tool = MagicMock()
        tool.args = "not a dict"
        assert _visible_schema_fields(tool) == []


class TestNormalizeCallArgs:
    def test_pure_kwargs(self):
        snap = _make_snapshot(kwargs={"query": "foo", "n": 5})
        tool = _make_tool({"query": {}, "n": {}})
        result = _normalize_call_args(snap, tool)
        assert result == {"query": "foo", "n": 5}

    def test_positional_args_mapped_to_field_names(self):
        snap = _make_snapshot(args=("hello", 3))
        tool = _make_tool({"query": {}, "max_results": {}})
        result = _normalize_call_args(snap, tool)
        assert result == {"query": "hello", "max_results": 3}

    def test_positional_and_kwargs_merged(self):
        snap = _make_snapshot(args=("hello",), kwargs={"max_results": 5})
        tool = _make_tool({"query": {}, "max_results": {}})
        result = _normalize_call_args(snap, tool)
        assert result == {"query": "hello", "max_results": 5}

    def test_injected_kwargs_stripped(self):
        snap = _make_snapshot(
            kwargs={"query": "foo", "runtime": "injected", "state": "bad"}
        )
        tool = _make_tool({"query": {}})
        result = _normalize_call_args(snap, tool)
        assert "runtime" not in result
        assert "state" not in result
        assert result["query"] == "foo"

    def test_excess_positional_args_fall_back_to_args_key(self):
        snap = _make_snapshot(args=("a", "b", "c"))
        tool = _make_tool({"x": {}})
        result = _normalize_call_args(snap, tool)
        assert result["x"] == "a"
        assert result["args"] == ["b", "c"]

    def test_no_args_returns_kwargs_only(self):
        snap = _make_snapshot(args=(), kwargs={"q": "test"})
        tool = _make_tool({"q": {}})
        result = _normalize_call_args(snap, tool)
        assert result == {"q": "test"}

    def test_kwargs_win_on_conflict_with_positional(self):
        snap = _make_snapshot(args=("positional",), kwargs={"query": "kwarg"})
        tool = _make_tool({"query": {}})
        result = _normalize_call_args(snap, tool)
        assert result["query"] == "kwarg"

    def test_empty_snapshot(self):
        snap = _make_snapshot(args=(), kwargs={})
        tool = _make_tool({"query": {}})
        result = _normalize_call_args(snap, tool)
        assert result == {}
