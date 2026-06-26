from __future__ import annotations

from pathlib import Path

EXPERIMENTAL_TEST_FILES = {
    "test_llm_control.py",
    "test_policy_compare.py",
    "test_v6_to_v10.py",
}


def pytest_addoption(parser):
    parser.addoption(
        "--run-experimental",
        action="store_true",
        default=False,
        help="collect tests for unverified experimental ML/LLM research tools",
    )


def pytest_ignore_collect(collection_path, config):
    path = Path(str(collection_path))
    return path.name in EXPERIMENTAL_TEST_FILES and not config.getoption("--run-experimental")
