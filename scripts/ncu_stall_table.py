#!/usr/bin/env python3
"""
Show a ranked table of warp stall types from an NCU report.

Usage:
    python3 ncu_stall_table.py <report.ncu-rep> <kernel_name_substring>
"""

import sys
import csv
import subprocess
from io import StringIO


def extract_stalls(report: str, kernel: str) -> dict[str, float]:
    result = subprocess.run(
        ["ncu", "--import", report, "--kernel-name", kernel, "--page", "raw", "--csv"],
        capture_output=True, text=True,
        env={**__import__("os").environ, "LC_ALL": "C"},
    )
    output = result.stdout + result.stderr
    lines = [l for l in output.strip().split("\n") if l and not l.startswith("==")]
    if len(lines) < 3:
        print("ERROR: NCU produced fewer than 3 data rows. Check kernel name and report path.")
        print(output[:2000])
        sys.exit(1)

    header = next(csv.reader(StringIO(lines[0])))
    # units  = next(csv.reader(StringIO(lines[1])))  # not needed here
    data   = next(csv.reader(StringIO(lines[2])))

    stalls: dict[str, float] = {}
    for i, h in enumerate(header):
        # PC sampling stall counts (not the warpsampling: duplicate prefix)
        if ("pcsamp_warps_issue_stalled" in h
                and "warpsampling:" not in h
                and "Triage" not in h
                and i < len(data)
                and data[i] not in ("", "n/a")):
            try:
                val = float(data[i])
                if val > 0:
                    # Strip the long metric prefix, keep the stall-reason suffix
                    # e.g. smsp__pcsamp_warps_issue_stalled_long_scoreboard.sum -> long_scoreboard
                    reason = h.split("pcsamp_warps_issue_stalled_")[-1].split(".")[0]
                    stalls[reason] = stalls.get(reason, 0.0) + val
            except ValueError:
                pass

    return stalls


def print_table(stalls: dict[str, float]) -> None:
    if not stalls:
        print("No non-zero stall metrics found.")
        return

    total = sum(stalls.values())
    ranked = sorted(stalls.items(), key=lambda kv: kv[1], reverse=True)

    col_w = max(len(r) for r, _ in ranked)
    col_w = max(col_w, len("Stall Reason"))
    header = f"{'Stall Reason':<{col_w}}  {'Samples':>14}  {'% of Total':>10}"
    sep    = "-" * len(header)

    print(sep)
    print(header)
    print(sep)
    for reason, count in ranked:
        pct = 100.0 * count / total if total > 0 else 0.0
        print(f"{reason:<{col_w}}  {count:>14,.0f}  {pct:>9.2f}%")
    print(sep)
    print(f"{'TOTAL':<{col_w}}  {total:>14,.0f}  {'100.00%':>10}")
    print(sep)


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <report.ncu-rep> <kernel_name_substring>")
        sys.exit(1)

    report, kernel = sys.argv[1], sys.argv[2]
    stalls = extract_stalls(report, kernel)
    print(f"\nStall breakdown for kernel matching '{kernel}' in '{report}'\n")
    print_table(stalls)


if __name__ == "__main__":
    main()
