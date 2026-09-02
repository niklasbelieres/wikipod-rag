"""Generate a styled HTML comparison report for an `evaluate-models` run.

`evaluate-models` writes one results.json/results.csv subfolder per model
under a dated run directory (see run_slm_eval.py). This script is run
manually afterwards: it loads every model's results.json from that run
directory and renders a single self-contained HTML page (no external
assets, so it works offline / on a Raspberry Pi) comparing model answers
and latency side by side, query by query.
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

DATE_FOLDER_FORMAT = "%d.%m.%y"


def parse_run_date(dir_name: str) -> datetime | None:
    """Parse a DD.MM.YY run folder name, or None if it doesn't match."""
    try:
        return datetime.strptime(dir_name, DATE_FOLDER_FORMAT)
    except ValueError:
        return None


def find_latest_run_dir(eval_dir: Path) -> Path:
    """Return the most recently dated subfolder under eval_dir.

    Folders named DD.MM.YY are ranked by their parsed date (a plain string
    sort of "DD.MM.YY" is wrong across month boundaries, e.g. "05.09.26"
    would sort before "31.08.26"). Any subfolder that doesn't match the
    pattern falls back to its mtime and always ranks below a properly
    dated folder.
    """
    candidates = [p for p in eval_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run subfolders found under {eval_dir}")

    def sort_key(p: Path) -> tuple[int, float]:
        parsed = parse_run_date(p.name)
        if parsed is not None:
            return (1, parsed.timestamp())
        return (0, p.stat().st_mtime)

    return max(candidates, key=sort_key)


def load_model_result(results_json_path: Path) -> dict:
    with results_json_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_run_results(run_dir: Path) -> list[dict]:
    """Load every model's results.json under run_dir, fastest model first."""
    results = []
    for model_dir in sorted(run_dir.iterdir()):
        results_path = model_dir / "results.json"
        if model_dir.is_dir() and results_path.exists():
            results.append(load_model_result(results_path))
    return sorted(results, key=lambda r: r["mean_latency_seconds"])


def build_summary_rows(results: list[dict]) -> list[dict]:
    """Extract {model, mean_latency_seconds, query_count} per model."""
    return [
        {
            "model": r["model"],
            "mean_latency_seconds": r["mean_latency_seconds"],
            "query_count": len(r["per_query"]),
        }
        for r in results
    ]


def group_by_query(results: list[dict]) -> list[dict]:
    """Pivot per-model per_query lists into one entry per query.

    Queries are matched by exact query string, since every model in one
    evaluate-models run answers the same prepared dataset.
    """
    if not results:
        return []

    order: list[str] = []
    by_query: dict[str, dict] = {}

    for result in results:
        model_name = result["model"]
        for entry in result["per_query"]:
            query = entry["query"]
            if query not in by_query:
                order.append(query)
                by_query[query] = {
                    "query": query,
                    "relevant_titles": entry["relevant_titles"],
                    "by_model": {},
                }
            by_query[query]["by_model"][model_name] = {
                "answer": entry["answer"],
                "latency_seconds": entry["latency_seconds"],
            }

    return [by_query[query] for query in order]


def render_summary_table(rows: list[dict]) -> str:
    """Render the top summary table as an HTML fragment."""
    body_rows = "\n".join(
        f"<tr><td>{html.escape(row['model'])}</td>"
        f"<td>{row['mean_latency_seconds']:.2f}</td>"
        f"<td>{row['query_count']}</td></tr>"
        for row in rows
    )
    return (
        "<table class=\"summary\">\n"
        "<thead><tr><th>Model</th><th>Mean Latency (s)</th><th>Queries</th></tr></thead>\n"
        f"<tbody>\n{body_rows}\n</tbody>\n"
        "</table>"
    )


def render_query_card(entry: dict, model_names: list[str]) -> str:
    """Render one query's comparison card: query, titles, and a per-model answer table."""
    titles = ", ".join(html.escape(t) for t in entry["relevant_titles"])

    rows = []
    for model_name in model_names:
        model_entry = entry["by_model"].get(model_name)
        if model_entry is None:
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(model_name)}</td>"
            f"<td>{model_entry['latency_seconds']:.2f}</td>"
            f"<td class=\"answer\">{html.escape(model_entry['answer'])}</td>"
            "</tr>"
        )
    body_rows = "\n".join(rows)

    return (
        "<div class=\"query-card\">\n"
        f"<h2>{html.escape(entry['query'])}</h2>\n"
        f"<p class=\"titles\">Relevant titles: {titles}</p>\n"
        "<table class=\"answers\">\n"
        "<thead><tr><th>Model</th><th>Latency (s)</th><th>Answer</th></tr></thead>\n"
        f"<tbody>\n{body_rows}\n</tbody>\n"
        "</table>\n"
        "</div>"
    )


_STYLE = """
body {
  font-family: sans-serif;
  max-width: 900px;
  margin: 2rem auto;
  padding: 0 1rem;
  color: #1a1a1a;
  background: #fff;
}
h1 { margin-bottom: 0.25rem; }
.subtitle { color: #666; margin-top: 0; }
table { border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }
th, td { border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: left; vertical-align: top; }
th { background: #f5f5f5; }
table.answers tbody tr:nth-child(even) { background: #fafafa; }
td.answer {
  font-family: monospace;
  white-space: pre-wrap;
  font-size: 0.85rem;
}
.query-card { border-bottom: 2px solid #eee; padding-bottom: 1.5rem; margin-bottom: 1.5rem; }
.titles { color: #555; font-size: 0.9rem; }
"""


def render_report(results: list[dict], run_label: str) -> str:
    """Assemble the full self-contained HTML document."""
    summary_rows = build_summary_rows(results)
    queries = group_by_query(results)
    model_names = [row["model"] for row in summary_rows]

    query_cards = "\n".join(render_query_card(entry, model_names) for entry in queries)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Model comparison - {html.escape(run_label)}</title>
<style>{_STYLE}</style>
</head>
<body>
<h1>Model comparison</h1>
<p class="subtitle">Run: {html.escape(run_label)}</p>
{render_summary_table(summary_rows)}
{query_cards}
</body>
</html>
"""


def write_report(html_text: str, run_dir: Path, output: Path | None) -> Path:
    """Write html_text to `output` if given, else to run_dir / 'report.html'."""
    target = output or (run_dir / "report.html")
    target.write_text(html_text, encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML comparison report for one evaluate-models run."
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("eval_results"),
        help="Base directory containing dated run subfolders (default: eval_results).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Explicit run folder to report on, overriding auto-discovery of the latest "
        "dated subfolder under --eval-dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="HTML output file path (default: <run-dir>/report.html).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run_dir(args.eval_dir)
    results = load_run_results(run_dir)
    report_html = render_report(results, run_label=run_dir.name)
    target = write_report(report_html, run_dir, args.output)
    print(f"Report written to {target}")


if __name__ == "__main__":
    main()
