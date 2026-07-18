"""Standard langchain-tests integration suite for the ``eval_python`` tool.

Runs the tool's ``invoke``/``ainvoke`` for real (no external service — Monty
executes locally), with a standalone ``ToolRuntime`` standing in for the one
LangGraph's ``ToolNode`` would normally inject.

https://docs.langchain.com/oss/python/contributing/standard-tests-langchain
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_monty import MontyCodeInterpreterMiddleware
from tests._standard_test_support import make_tool_runtime


class TestEvalPythonToolStandard(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return MontyCodeInterpreterMiddleware().tools[0]

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"code": "1 + 1", "runtime": make_tool_runtime()}
