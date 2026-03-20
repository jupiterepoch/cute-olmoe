import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that download/load the full HuggingFace model weights (~14 GB)",
    )
