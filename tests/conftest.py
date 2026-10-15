import pytest

_REPORT = pytest.StashKey[list]()


def pytest_configure(config):
    config.stash[_REPORT] = []


@pytest.fixture(scope="session")
def segmentor():
    """One LLMSegmentorWithShape (first target) for all tests: the shape, examples and vocabularies are loaded once."""
    from tests.support import create_segmentor
    return create_segmentor()


@pytest.fixture(scope="session")
def report_lines(pytestconfig) -> list[str]:
    """Lines to print in an "LLM usage" section at the end of the test run."""
    return pytestconfig.stash[_REPORT]


def pytest_terminal_summary(terminalreporter, config):
    lines = config.stash.get(_REPORT, [])
    if lines:
        terminalreporter.section("LLM usage")
        for line in lines:
            terminalreporter.write_line(line)
