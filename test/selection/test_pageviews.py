import gzip
from datetime import date
from unittest.mock import MagicMock, patch

from wikipod.selection.pageviews import (
    download_pageviews_day,
    download_pageviews_hour,
    get_views,
    load_pageviews,
    normalize_title,
    save_pageviews,
)


# -- normalize_title / get_views ----------------------------------------------


def test_normalize_title_converts_spaces_to_underscores():
    assert normalize_title("Climate change") == "Climate_change"


def test_normalize_title_is_idempotent_on_already_underscored_titles():
    assert normalize_title("Climate_change") == "Climate_change"


def test_get_views_normalizes_before_lookup():
    pageviews = {"Climate_change": 4821}
    assert get_views(pageviews, "Climate change") == 4821


def test_get_views_returns_zero_for_unknown_title():
    assert get_views({}, "Some unknown article") == 0


# -- load_pageviews -------------------------------------------------------------


def test_load_pageviews_parses_and_filters_by_domain(tmp_path):
    path = tmp_path / "pageviews-20260601-000000"
    path.write_text(
        "en Climate_change 4821 0\n"
        "en.m Climate_change 9110 0\n"
        "de Klimawandel 300 0\n"
    )
    assert load_pageviews(path) == {"Climate_change": 4821}


def test_load_pageviews_returns_empty_dict_for_missing_file(tmp_path):
    assert load_pageviews(tmp_path / "does_not_exist") == {}


def test_load_pageviews_aggregates_duplicate_titles(tmp_path):
    path = tmp_path / "pageviews"
    path.write_text("en Climate_change 100 0\nen Climate_change 50 0\n")
    assert load_pageviews(path) == {"Climate_change": 150}


# -- download_pageviews_hour ---------------------------------------------------


def _gzip_response(text: str) -> MagicMock:
    response = MagicMock()
    response.content = gzip.compress(text.encode("utf-8"))
    response.raise_for_status.return_value = None
    return response


def test_download_pageviews_hour_parses_gzipped_response():
    raw = "en Climate_change 100 0\nen.m Climate_change 50 0\nde Klimawandel 30 0\n"

    with patch("wikipod.selection.pageviews.requests.get", return_value=_gzip_response(raw)):
        views = download_pageviews_hour(date(2026, 6, 1), 0)

    assert views == {"Climate_change": 100}


def test_download_pageviews_hour_returns_empty_dict_on_request_failure():
    with patch("wikipod.selection.pageviews.requests.get", side_effect=Exception("network error")):
        views = download_pageviews_hour(date(2026, 6, 1), 0)

    assert views == {}


# -- download_pageviews_day ------------------------------------------------------


def test_download_pageviews_day_sums_counts_across_hours():
    hourly_results = [{"Climate_change": 10}, {"Climate_change": 5, "Einstein": 20}] + [{}] * 22

    with patch(
        "wikipod.selection.pageviews.download_pageviews_hour", side_effect=hourly_results
    ) as mock_hour:
        views = download_pageviews_day(date(2026, 6, 1))

    assert views == {"Climate_change": 15, "Einstein": 20}
    assert mock_hour.call_count == 24


def test_download_pageviews_day_reports_progress_for_all_24_hours():
    calls = []

    with patch("wikipod.selection.pageviews.download_pageviews_hour", return_value={}):
        download_pageviews_day(date(2026, 6, 1), on_progress=lambda done, total: calls.append((done, total)))

    assert calls[-1] == (24, 24)
    assert len(calls) == 24


# -- save_pageviews ---------------------------------------------------------------


def test_save_pageviews_roundtrips_through_load_pageviews(tmp_path):
    views = {"Climate_change": 4821, "Albert_Einstein": 1200}
    path = tmp_path / "aggregated.txt"

    save_pageviews(views, path)

    assert load_pageviews(path) == views
