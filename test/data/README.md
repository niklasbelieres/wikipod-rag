# Test data

Place `climate-change-mini.zim` here to run the tests/scripts that depend on
a real ZIM archive. Tests that need it are skipped automatically if it's
absent (see `pytestmark` in `test/analysis/test_reader.py` and
`test/analysis/test_metadata.py`).

A small demo ZIM can be built with `zimwriterfs`, or downloaded from the
[Kiwix library](https://library.kiwix.org/) and trimmed with `zimdump`.
