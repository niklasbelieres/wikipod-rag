from unittest.mock import MagicMock

from click.testing import CliRunner

from wikipod.chunking.models import Chunk
from wikipod.evaluation.query_analyzer import QueryAnalyzer
from wikipod.evaluation.run_eval import (
    load_eval_dataset,
    main,
    run_eval,
    run_single_query,
)


def _chunk(article_title: str) -> Chunk:
    return Chunk(
        article_id=1,
        article_title=article_title,
        section_title="Lead",
        chunk_index=0,
        word_count=1,
        text="x",
    )


# -- load_eval_dataset ----------------------------------------------------------
def test_load_eval_dataset_reads_valid_file(tmp_path):
    path = tmp_path / "test_file.yaml"
    path.write_text("""
    - query: "who was Albert Einstein"
      relevant_titles:
        - "Albert Einstein"

    - query: "what causes climate change"
      relevant_titles:
        - "Causes of climate change"
        - "Climate change"
    """)
    
    result = load_eval_dataset(path)
    assert len(result) == 2
    assert result[0]["query"] == "who was Albert Einstein"
    


def test_load_eval_dataset_handles_empty_file(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    
    result = load_eval_dataset(path)
    assert len(result) == 0


def test_load_eval_dataset_handles_missing_file(tmp_path):
    path = tmp_path / "does_not_exist.yaml"
    
    result = load_eval_dataset(path)
    assert len(result) == 0


# -- run_single_query -------------------------------------------------------------
def test_run_single_query_returns_article_titles():
    retriever = MagicMock()
    retriever.retrieve.return_value = [_chunk("Europe"), _chunk("France")]
    
    result = run_single_query(retriever, "Whats the capital of France?", 2)
    assert len(result) == 2
    assert result[0] == "Europe"


def test_run_single_query_removes_duplicate_titles():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        _chunk("Europe"),
        _chunk("France"),
        _chunk("France"),
    ]

    result = run_single_query(
        retriever,
        "Whats the capital of France?",
        3,
    )

    assert result == ["Europe", "France"]

def test_run_single_query_handles_empty_retrieval_result():
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    
    result = run_single_query(retriever, "Whats the capital of France?", 3)
    assert len(result) == 0


def test_run_single_query_requests_more_chunks_for_deduplication():
    retriever = MagicMock()
    query = "Whats the capital of France?"
    k = 3

    run_single_query(retriever, query, k)

    retriever.retrieve.assert_called_once_with(
        query,
        k=k * 4,
    )


# -- run_eval ------------------------------------------------------------------
def test_run_eval_matches_known_example():
    retriever = MagicMock()
    retriever.retrieve.side_effect = [
        [_chunk("Europe"), _chunk("France")],   # Query 1: France an Position 2
        [_chunk("Foo"), _chunk("Bar")],          # Query 2: Foo an Position 1
        [_chunk("X"), _chunk("Y")],              # Query 3: kein Treffer
    ]

    dataset = [
        {"query": "q1", "relevant_titles": ["France"]},
        {"query": "q2", "relevant_titles": ["Foo"]},
        {"query": "q3", "relevant_titles": ["Baz"]},
    ]

    result = run_eval(retriever, dataset, k=5)
    assert result["mean_reciprocal_rank"] == 0.5


def test_run_eval_handles_empty_dataset():
    retriever = MagicMock()
    retriever.retrieve.side_effect = [
        [_chunk("Europe"), _chunk("France")],
        [_chunk("Foo"), _chunk("Bar")],
        [_chunk("X"), _chunk("Y")], 
    ]
    k = 5

    dataset = []

    result = run_eval(retriever, dataset, k=k)
    assert result["mean_reciprocal_rank"] == 0.0
    assert result["k"] == k
    assert result["mean_recall_at_k"] == 0.0
    assert result["mean_precision_at_k"] == 0.0
    assert result["mean_ndcg_at_k"] == 0.0
    assert result["mean_reciprocal_rank"] == 0.0
    assert result["by_category"] == {}
    assert result["per_query"] == []
    assert result["k"] == k
    


def test_run_eval_per_query_entries_have_correct_values():
    retriever = MagicMock()
    retriever.retrieve.side_effect = [
        [_chunk("Europe"), _chunk("France")],
        [_chunk("Foo"), _chunk("Bar")],
        [_chunk("X"), _chunk("Y")],
    ]

    dataset = [
        {"query": "q1", "category": "short", "relevant_titles": ["France"]},
        {"query": "q2", "relevant_titles": ["Foo"]},
        {"query": "q3", "relevant_titles": ["Baz"]},
    ]

    result = run_eval(retriever, dataset, k=5)
    per_query = result["per_query"]
    assert len(per_query) == 3

    assert per_query[0]["query"] == "q1"
    assert per_query[0]["category"] == "short"
    assert per_query[0]["relevant_titles"] == ["France"]
    assert per_query[0]["retrieved_titles"] == ["Europe", "France"]
    assert per_query[0]["reciprocal_rank"] == 0.5

    assert per_query[1]["query"] == "q2"
    assert per_query[1]["relevant_titles"] == ["Foo"]
    assert per_query[1]["retrieved_titles"] == ["Foo", "Bar"]
    assert per_query[1]["reciprocal_rank"] == 1.0

    assert per_query[2]["query"] == "q3"
    assert per_query[2]["relevant_titles"] == ["Baz"]
    assert per_query[2]["retrieved_titles"] == ["X", "Y"]
    assert per_query[2]["reciprocal_rank"] == 0.0

def test_run_eval_aggregates_metrics_by_category():
    retriever = MagicMock()
    retriever.retrieve.side_effect = [
        [_chunk("France")],
        [_chunk("Wrong")],
    ]

    dataset = [
        {"query": "q1", "category": "short", "relevant_titles": ["France"]},
        {"query": "q2", "category": "short", "relevant_titles": ["Germany"]},
    ]

    result = run_eval(retriever, dataset, k=5)

    assert result["by_category"]["short"]["query_count"] == 2
    assert result["by_category"]["short"]["mean_recall_at_k"] == 0.5
    assert result["by_category"]["short"]["mean_precision_at_k"] == 0.1
    assert result["by_category"]["short"]["mean_ndcg_at_k"] == 0.5
    assert result["by_category"]["short"]["mean_reciprocal_rank"] == 0.5

def test_main_writes_json_output(tmp_path, monkeypatch):
    runner = CliRunner()

    dataset_path = tmp_path / "eval.yaml"
    dataset_path.write_text(
        """
- query: "q1"
  relevant_titles:
    - "France"
"""
    )

    output_path = tmp_path / "results.json"

    fake_results = {
        "k": 5,
        "mean_recall_at_k": 1.0,
        "mean_precision_at_k": 0.2,
        "mean_ndcg_at_k": 1.0,
        "mean_reciprocal_rank": 1.0,
        "per_query": [],
    }

    monkeypatch.setattr(
        "wikipod.evaluation.run_eval.run_eval",
        lambda retriever, dataset, k: fake_results,
    )

    monkeypatch.setattr(
        "wikipod.evaluation.run_eval.build_client",
        lambda config: MagicMock(),
    )

    monkeypatch.setattr(
        "wikipod.evaluation.run_eval.Embedder",
        MagicMock(),
    )

    monkeypatch.setattr(
        "wikipod.evaluation.run_eval.Retriever",
        MagicMock(),
    )

    result = runner.invoke(
        main,
        [
            "--dataset",
            str(dataset_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert '"mean_recall_at_k": 1.0' in output_path.read_text()

def test_main_writes_csv_output(tmp_path, monkeypatch):
    runner = CliRunner()

    dataset_path = tmp_path / "eval.yaml"
    dataset_path.write_text(
        """
- query: "q1"
  relevant_titles:
    - "France"
"""
    )

    output_path = tmp_path / "results.csv"

    fake_results = {
        "k": 5,
        "mean_recall_at_k": 1.0,
        "mean_precision_at_k": 0.2,
        "mean_ndcg_at_k": 1.0,
        "mean_reciprocal_rank": 1.0,
        "per_query": [
            {
                "query": "q1",
                "category": "normal",
                "is_out_of_scope": False,
                "relevant_titles": ["France"],
                "retrieved_titles": ["Europe", "France"],
                "recall_at_k": 1.0,
                "precision_at_k": 0.2,
                "ndcg_at_k": 1.0,
                "reciprocal_rank": 1.0,
            }
        ],
    }

    monkeypatch.setattr(
        "wikipod.evaluation.run_eval.run_eval",
        lambda retriever, dataset, k: fake_results,
    )

    monkeypatch.setattr(
        "wikipod.evaluation.run_eval.build_client",
        lambda config: MagicMock(),
    )

    monkeypatch.setattr(
        "wikipod.evaluation.run_eval.Embedder",
        MagicMock(),
    )

    monkeypatch.setattr(
        "wikipod.evaluation.run_eval.Retriever",
        MagicMock(),
    )

    result = runner.invoke(
        main,
        [
            "--dataset",
            str(dataset_path),
            "--csv-output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()

    content = output_path.read_text()

    assert (
            "query,category,is_out_of_scope,relevant_titles,retrieved_titles"
            in content
    )
    assert "France" in content
    assert "Europe; France" in content



def test_main_writes_output_directory(tmp_path, monkeypatch):
        runner = CliRunner()

        dataset_path = tmp_path / "eval.yaml"
        dataset_path.write_text(
            """
        - query: "q1"
          category: "normal"
          relevant_titles:
            - "France"
        """
        )

        output_dir = tmp_path / "eval_results"

        fake_results = {
            "k": 5,
            "mean_recall_at_k": 1.0,
            "mean_precision_at_k": 0.2,
            "mean_ndcg_at_k": 1.0,
            "mean_reciprocal_rank": 1.0,
            "per_query": [
                {
                    "query": "q1",
                    "relevant_titles": ["France"],
                    "retrieved_titles": ["Europe", "France"],
                    "recall_at_k": 1.0,
                    "precision_at_k": 0.2,
                    "ndcg_at_k": 1.0,
                    "reciprocal_rank": 1.0,
                }
            ],
        }

        monkeypatch.setattr(
            "wikipod.evaluation.run_eval.run_eval",
            lambda retriever, dataset, k: fake_results,
        )
        monkeypatch.setattr(
            "wikipod.evaluation.run_eval.build_client",
            lambda config: MagicMock(),
        )
        monkeypatch.setattr(
            "wikipod.evaluation.run_eval.Embedder",
            MagicMock(),
        )
        monkeypatch.setattr(
            "wikipod.evaluation.run_eval.Retriever",
            MagicMock(),
        )

        result = runner.invoke(
            main,
            [
                "--dataset",
                str(dataset_path),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        assert output_dir.exists()
        assert (output_dir / "results.json").exists()
        assert (output_dir / "results.csv").exists()

def test_run_eval_marks_out_of_scope_query():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        _chunk("Germany"),
        _chunk("Nelson Mandela"),
    ]

    dataset = [
        {
            "query": "Tell me about Albert Einstein",
            "category": "out_of_scope",
            "relevant_titles": [],
        }
    ]

    result = run_eval(retriever, dataset, k=5)

    entry = result["per_query"][0]

    assert entry["is_out_of_scope"] is True


def test_run_eval_uses_normalized_query_from_analyzer():
    retriever = MagicMock()
    retriever.retrieve.return_value = [_chunk("George III")]

    analyzer = MagicMock()
    analyzer.analyze.return_value.normalized_query = "Who was George III?"

    dataset = [
        {
            "query": "Who was Geogre III?",
            "category": "typo",
            "relevant_titles": ["George III"],
        }
    ]

    run_eval(retriever, dataset, k=5, analyzer=analyzer)

    analyzer.analyze.assert_called_once_with("Who was Geogre III?")
    retriever.retrieve.assert_called_once_with(
        "Who was George III?",
        k=20,
    )


def test_run_eval_records_normalized_query():
    retriever = MagicMock()
    retriever.retrieve.return_value = [_chunk("George III")]

    analyzer = MagicMock()
    analyzer.analyze.return_value.normalized_query = "Who was George III?"

    dataset = [
        {
            "query": "Who was Geogre III?",
            "category": "typo",
            "relevant_titles": ["George III"],
        }
    ]

    result = run_eval(retriever, dataset, k=5, analyzer=analyzer)

    entry = result["per_query"][0]

    assert entry["query"] == "Who was Geogre III?"
    assert entry["normalized_query"] == "Who was George III?"

def test_main_accepts_query_analyzer_flag():
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "--use-query-analyzer" in result.output


def test_main_uses_query_analyzer_when_flag_is_set(tmp_path, monkeypatch):
    runner = CliRunner()

    dataset_path = tmp_path / "eval.yaml"
    dataset_path.write_text(
        """
- query: "Who was Geogre III?"
  category: "typo"
  relevant_titles:
    - "George III"
"""
    )

    run_eval_mock = MagicMock(
        return_value={
            "k": 5,
            "mean_recall_at_k": 1.0,
            "mean_precision_at_k": 0.2,
            "mean_ndcg_at_k": 1.0,
            "mean_reciprocal_rank": 1.0,
            "per_query": [],
        }
    )

    monkeypatch.setattr(
        "wikipod.evaluation.run_eval.run_eval",
        run_eval_mock,
    )
    monkeypatch.setattr(
        "wikipod.evaluation.run_eval.build_client",
        lambda config: MagicMock(),
    )
    monkeypatch.setattr(
        "wikipod.evaluation.run_eval.Embedder",
        MagicMock(),
    )
    monkeypatch.setattr(
        "wikipod.evaluation.run_eval.Retriever",
        MagicMock(),
    )

    result = runner.invoke(
        main,
        [
            "--dataset",
            str(dataset_path),
            "--top-k",
            "5",
            "--use-query-analyzer",
        ],
    )

    assert result.exit_code == 0

    analyzer = run_eval_mock.call_args.kwargs["analyzer"]
    assert isinstance(analyzer, QueryAnalyzer)
