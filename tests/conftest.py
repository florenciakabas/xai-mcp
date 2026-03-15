import pytest

from xai_toolkit.server import init_server

SLOW_TEST_FILES = {
    "test_background_data.py",
    "test_batch_shap.py",
    "test_cli.py",
    "test_compare_predictions.py",
    "test_drift_integration.py",
    "test_intrinsic.py",
    "test_knowledge.py",
    "test_pipeline_bridge.py",
    "test_plots.py",
    "test_second_model.py",
    "test_server_errors.py",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom flags for local developer test loops."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow/integration tests in addition to fast tests.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Declare custom markers for better discoverability."""
    config.addinivalue_line(
        "markers",
        "slow: expensive integration-style tests excluded from default local runs",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip slow test modules unless explicitly requested."""
    if config.getoption("--run-slow"):
        return

    skip_slow = pytest.mark.skip(reason="slow test: use --run-slow to include")
    slow_marker = pytest.mark.slow
    for item in items:
        file_name = item.location[0].split("/")[-1].split("\\")[-1]
        if file_name in SLOW_TEST_FILES:
            item.add_marker(slow_marker)
            item.add_marker(skip_slow)


@pytest.fixture(scope="session", autouse=True)
def _init_xai_server():
    """Initialize server models/knowledge once for the test session."""
    init_server()
