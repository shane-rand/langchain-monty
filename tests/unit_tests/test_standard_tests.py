"""Standard langchain-tests unit suite for the ``eval_python`` tool.

https://docs.langchain.com/oss/python/contributing/standard-tests-langchain
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_monty import MontyCodeInterpreterMiddleware
from tests._standard_test_support import make_tool_runtime


class TestEvalPythonToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return MontyCodeInterpreterMiddleware().tools[0]

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"code": "1 + 1", "runtime": make_tool_runtime()}
