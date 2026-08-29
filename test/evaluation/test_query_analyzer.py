from wikipod.evaluation.query_analyzer import QueryAnalysis, QueryAnalyzer


def test_query_analysis_defaults_to_in_scope():
    analysis = QueryAnalysis(
        original_query="Who was George III?",
        normalized_query="Who was George III?",
        query_type="normal",
    )

    assert analysis.original_query == "Who was George III?"
    assert analysis.normalized_query == "Who was George III?"
    assert analysis.query_type == "normal"
    assert analysis.is_out_of_scope is False


def test_query_analyzer_keeps_normal_query_unchanged():
    analyzer = QueryAnalyzer()

    analysis = analyzer.analyze("Who was George III?")

    assert analysis.original_query == "Who was George III?"
    assert analysis.normalized_query == "Who was George III?"
    assert analysis.query_type == "normal"
    assert analysis.is_out_of_scope is False


def test_query_analyzer_normalizes_whitespace():
    analyzer = QueryAnalyzer()

    analysis = analyzer.analyze("  Who   was George III?  ")

    assert analysis.original_query == "  Who   was George III?  "
    assert analysis.normalized_query == "Who was George III?"
    assert analysis.query_type == "normal"


def test_query_analyzer_detects_known_typo():
    analyzer = QueryAnalyzer(
        corrections={
            "Geogre": "George",
        }
    )

    analysis = analyzer.analyze("Who was Geogre III?")

    assert analysis.original_query == "Who was Geogre III?"
    assert analysis.normalized_query == "Who was George III?"
    assert analysis.query_type == "typo"


def test_query_analyzer_does_not_change_unmatched_query():
    analyzer = QueryAnalyzer(
        corrections={
            "Geogre": "George",
        }
    )

    analysis = analyzer.analyze("Who was George IV?")

    assert analysis.normalized_query == "Who was George IV?"
    assert analysis.query_type == "normal"


def test_query_analyzer_marks_known_out_of_scope_query():
    analyzer = QueryAnalyzer(
        out_of_scope_queries={
            "Tell me about Albert Einstein",
        }
    )

    analysis = analyzer.analyze("Tell me about Albert Einstein")

    assert analysis.is_out_of_scope is True
    assert analysis.query_type == "out_of_scope"


def test_query_analyzer_keeps_in_scope_query_in_scope():
    analyzer = QueryAnalyzer(
        out_of_scope_queries={
            "Tell me about Albert Einstein",
        }
    )

    analysis = analyzer.analyze("Who was George III?")

    assert analysis.is_out_of_scope is False
    assert analysis.query_type == "normal"





