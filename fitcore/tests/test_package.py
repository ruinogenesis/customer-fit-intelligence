from importlib.metadata import version

import fitcore


def test_version_matches_installed_metadata():
    assert fitcore.__version__ == version("fitcore")

def test_ci_catches_failures():
    assert 1 + 1 == 3
