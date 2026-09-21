import weakref
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from wikipod.analysis import reader
from wikipod.analysis.models import ArticleMetadata


class Batch(list):
    """Weak-referenceable result list for checking release before resubmission."""


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    path = tmp_path / "test.zim"
    path.touch()
    counter = SimpleNamespace(value=0)
    manager = SimpleNamespace(Value=lambda *args: counter, Lock=lambda: None)
    monkeypatch.setattr(reader.multiprocessing, "Manager", lambda: nullcontext(manager))
    monkeypatch.setattr(reader, "Archive", lambda path: SimpleNamespace(article_count=23))
    return path, counter


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_slow_consumer_bounds_batches_and_releases_results(fake_runtime, monkeypatch, workers):
    path, counter = fake_runtime
    references = []
    submitted = []

    class Pool:
        def __init__(self, max_workers):
            assert max_workers == workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, path, start, end, count, lock, *, include_sections):
            # The consumed result list must be gone before a replacement is submitted.
            assert sum(ref() is not None for ref in references) < workers
            assert include_sections is False
            submitted.append((start, end))
            batch = Batch(
                ArticleMetadata(
                    article_id=i, title=str(i), html_size_bytes=1, word_count=1,
                    link_count=0, links=[], section_count=0, sections=[], categories=[],
                )
                for i in range(start, end)
            )
            references.append(weakref.ref(batch))
            counter.value += end - start
            future = Future()
            future.set_result(batch)
            return future

    monkeypatch.setattr(reader, "ProcessPoolExecutor", Pool)
    progress = Mock()
    iterator = reader.iter_articles_metadata_parallel(
        path, workers=workers, batch_size=3, include_sections=False, on_progress=progress
    )
    seen = []
    for article in iterator:
        seen.append(article.article_id)
        # Consume one article at a time while all submitted work finishes immediately.
        assert len(submitted) <= workers + (len(seen) - 1) // 3
        assert sum(ref() is not None for ref in references) <= workers
    assert sorted(seen) == list(range(23))
    assert submitted == [(i, min(i + 3, 23)) for i in range(0, 23, 3)]
    assert all(ref() is None for ref in references)
    progress.assert_called_with(23, 23)


def test_empty_batches_and_batch_failures_are_handled(fake_runtime, monkeypatch):
    path, _ = fake_runtime
    pool = Mock()
    futures = []
    for result in [[], [], RuntimeError("worker failed")]:
        future = Future()
        if isinstance(result, Exception):
            future.set_exception(result)
        else:
            future.set_result(result)
        futures.append(future)
    pool.submit.side_effect = futures
    monkeypatch.setattr(reader, "ProcessPoolExecutor", lambda **kwargs: nullcontext(pool))
    with pytest.raises(RuntimeError, match="worker failed"):
        list(reader.iter_articles_metadata_parallel(path, workers=1, batch_size=3))
    assert pool.submit.call_count == 3


def test_empty_archive_reports_completion_without_submitting(fake_runtime, monkeypatch):
    path, _ = fake_runtime
    monkeypatch.setattr(reader, "Archive", lambda path: SimpleNamespace(article_count=0))
    pool = Mock()
    monkeypatch.setattr(reader, "ProcessPoolExecutor", lambda **kwargs: nullcontext(pool))
    progress = Mock()
    assert list(reader.iter_articles_metadata_parallel(path, on_progress=progress)) == []
    pool.submit.assert_not_called()
    progress.assert_called_once_with(0, 0)
