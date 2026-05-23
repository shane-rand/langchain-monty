import pytest
from unittest.mock import MagicMock
from langchain.tools import ToolRuntime


@pytest.fixture
def mock_runtime():
    runtime = MagicMock(spec=ToolRuntime)
    runtime.tools = []
    runtime.config = {}
    return runtime


@pytest.fixture
def mock_tool():
    tool = MagicMock()
    tool.name = "search"
    tool.args = {"query": {"type": "string"}}
    return tool
