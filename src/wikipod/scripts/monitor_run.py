"""Periodically samples system metrics into a CSV while a long-running job
(e.g. `wikipod index`) is active elsewhere -- Load, RAM, Swap, and on a
Raspberry Pi, CPU temperature/throttling status.

Standalone and stdlib-only on purpose: runs independently of the wikipod
package (no pip install needed, nothing to conflict with whatever's using
the venv for the actual indexing job) and reads straight from /proc and
`vcgencmd` rather than pulling in psutil for what's a one-off ops script.

Usage (in its own tmux pane, alongside the job being watched):
    python3 -m wikipod.scripts.monitor_run --out run_metrics.csv --interval 10

Ctrl+C to stop. Re-running with the same --out appends rather than overwrites,
so stopping and restarting the monitor doesn't lose earlier samples.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def _read_meminfo() -> dict[str, int]:
    """Parse /proc/meminfo into {key: value_in_kb}. Linux-only, matches the Pi target."""
    info: dict[str, int] = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            key, _, rest = line.partition(":")
            value = rest.strip().split()[0]
            info[key] = int(value)
    return info


def _read_vcgencmd(*args: str) -> str | None:
    """Run a vcgencmd subcommand, return its stdout, or None off-Pi (command missing)."""
    try:
        result = subprocess.run(
            ["vcgencmd", *args], capture_output=True, text=True, timeout=5, check=True
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def sample() -> dict[str, object]:
    """Take one snapshot of system metrics."""
    meminfo = _read_meminfo()
    load1, load5, _load15 = os.getloadavg()
    cores = os.cpu_count() or 1

    temp_raw = _read_vcgencmd("measure_temp")  # "temp=48.2'C"
    temp_c = float(temp_raw.split("=")[1].rstrip("'C")) if temp_raw else None

    throttled_raw = _read_vcgencmd("get_throttled")  # "throttled=0x0"
    throttled = throttled_raw.split("=")[1] if throttled_raw else None

    mem_total_mb = meminfo.get("MemTotal", 0) / 1024
    mem_available_mb = meminfo.get("MemAvailable", 0) / 1024
    swap_total_mb = meminfo.get("SwapTotal", 0) / 1024
    swap_free_mb = meminfo.get("SwapFree", 0) / 1024

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "load_1min": round(load1, 2),
        "load_5min": round(load5, 2),
        "cpu_pct_approx": round(load1 / cores * 100, 1),
        "mem_used_mb": round(mem_total_mb - mem_available_mb, 1),
        "mem_total_mb": round(mem_total_mb, 1),
        "swap_used_mb": round(swap_total_mb - swap_free_mb, 1),
        "swap_total_mb": round(swap_total_mb, 1),
        "cpu_temp_c": temp_c,
        "throttled": throttled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="run_metrics.csv", help="CSV output path.")
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between samples.")
    args = parser.parse_args()

    out_path = Path(args.out)
    write_header = not out_path.exists()

    print(f"Logging every {args.interval}s to {out_path} (Ctrl+C to stop)")
    with out_path.open("a", newline="") as fh:
        writer: csv.DictWriter | None = None
        try:
            while True:
                row = sample()
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                    if write_header:
                        writer.writeheader()
                writer.writerow(row)
                fh.flush()
                print(
                    f"{row['timestamp']}  load={row['load_1min']}  "
                    f"cpu~{row['cpu_pct_approx']}%  mem={row['mem_used_mb']}MB  "
                    f"swap={row['swap_used_mb']}MB  temp={row['cpu_temp_c']}C  "
                    f"throttled={row['throttled']}"
                )
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
