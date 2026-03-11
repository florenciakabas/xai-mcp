import pytest
from xai_toolkit.server import init_server


@pytest.fixture(scope="session", autouse=True)
def _init_xai_server():
    """Initialize server models/knowledge once for the test session."""
    init_server()
