"""Create summary statistics and plots from a monitor_run CSV file."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


def _read_metrics(path: Path) -> list[dict[str, str]]:
    """Read monitor_run CSV rows."""
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _values(rows: list[dict[str, str]], key: str) -> list[float]:
    """Convert one numeric CSV column to floats."""
    return [float(row[key]) for row in rows]


def _elapsed_minutes(rows: list[dict[str, str]]) -> list[float]:
    """Return elapsed minutes since the first sample."""
    timestamps = [datetime.fromisoformat(row["timestamp"]) for row in rows]
    start = timestamps[0]

    return [
        (timestamp - start).total_seconds() / 60
        for timestamp in timestamps
    ]


def _save_plot(
    elapsed: list[float],
    series: list[tuple[list[float], str]],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Save one time-series plot."""
    plt.figure(figsize=(9, 4.5))

    for values, label in series:
        plt.plot(elapsed, values, label=label)

    plt.xlabel("Elapsed time (minutes)")
    plt.ylabel(ylabel)
    plt.title(title)

    if len(series) > 1:
        plt.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
        )
        plt.tight_layout(rect=(0, 0, 0.78, 1))
    else:
        plt.tight_layout()


    plt.savefig(output_path, dpi=150)
    plt.close()


def analyze(csv_path: Path, output_dir: Path) -> None:
    """Print run statistics and create plots for one monitoring CSV."""
    rows = _read_metrics(csv_path)

    if not rows:
        raise ValueError(f"No samples found in {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    elapsed = _elapsed_minutes(rows)

    mem_used = _values(rows, "mem_used_mb")
    mem_available = _values(rows, "mem_available_mb")
    wikipod_rss = _values(rows, "wikipod_rss_mb")
    opensearch_rss = _values(rows, "opensearch_rss_mb")
    swap_used = _values(rows, "swap_used_mb")
    cpu = _values(rows, "cpu_pct_approx")
    temperature = _values(rows, "cpu_temp_c")

    duration_minutes = elapsed[-1]

    print(f"Run: {csv_path.name}")
    print(f"Samples: {len(rows)}")
    print(f"Duration: {duration_minutes:.1f} min")
    print(f"Peak memory used: {max(mem_used):.1f} MB")
    print(f"Minimum memory available: {min(mem_available):.1f} MB")
    print(f"Peak WikiPod RSS: {max(wikipod_rss):.1f} MB")
    print(f"Peak OpenSearch RSS: {max(opensearch_rss):.1f} MB")
    print(f"Peak swap used: {max(swap_used):.1f} MB")
    print(f"Peak CPU: {max(cpu):.1f} %")
    print(f"Peak temperature: {max(temperature):.1f} °C")

    _save_plot(
        elapsed,
        [(cpu, "CPU")],
        "CPU (%)",
        "CPU utilization",
        output_dir / "cpu.png",
    )

    _save_plot(
        elapsed,
        [
            (mem_used, "System memory used (includes processes)"),
            (mem_available, "System memory available"),
            (wikipod_rss, "WikiPod RSS"),
            (opensearch_rss, "OpenSearch RSS"),
        ],
        "Memory (MB)",
        "System and process memory",
        output_dir / "memory.png",
    )

    _save_plot(
        elapsed,
        [(swap_used, "Swap used")],
        "Swap (MB)",
        "Swap usage",
        output_dir / "swap.png",
    )

    _save_plot(
        elapsed,
        [(temperature, "CPU temperature")],
        "Temperature (°C)",
        "CPU temperature",
        output_dir / "temperature.png",
    )

    print(f"Plots written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create plots and summary statistics from monitor_run CSV output."
    )

    parser.add_argument(
        "csv",
        type=Path,
        help="Path to a monitor_run CSV file",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <csv-stem>_plots next to the CSV)",
    )

    args = parser.parse_args()

    output_dir = args.out_dir or args.csv.with_name(
        f"{args.csv.stem}_plots"
    )

    analyze(args.csv, output_dir)


if __name__ == "__main__":
    main()