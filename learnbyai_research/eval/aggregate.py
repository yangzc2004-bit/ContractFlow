from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_judgments(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        source = Path(path)
        if source.suffix == ".jsonl":
            with source.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        rows.append(json.loads(line))
        else:
            with source.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, list):
                    rows.extend(data)
                else:
                    rows.append(data)
    return rows


def aggregate_pairwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        a = row.get("system_a", "A")
        b = row.get("system_b", "B")
        winner = row.get("winner", "Tie")
        key = (a, b)
        if winner == "A":
            grouped[key][a] += 1
        elif winner == "B":
            grouped[key][b] += 1
        else:
            grouped[key]["Tie"] += 1
        grouped[key]["total"] += 1
    summary: list[dict[str, Any]] = []
    for (a, b), counts in sorted(grouped.items()):
        total = max(1, counts["total"])
        summary.append(
            {
                "system_a": a,
                "system_b": b,
                "a_wins": counts[a],
                "b_wins": counts[b],
                "ties": counts["Tie"],
                "total": total,
                "a_win_rate": round(counts[a] / total, 4),
                "b_win_rate": round(counts[b] / total, 4),
                "tie_rate": round(counts["Tie"] / total, 4),
            }
        )
    return summary


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["system_a", "system_b", "a_wins", "b_wins", "ties", "total"]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
