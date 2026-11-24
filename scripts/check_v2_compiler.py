from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learnbyai_research.io_utils import read_json, write_json
from learnbyai_research.llm import OpenAICompatibleClient, completion_history
from learnbyai_research.longgenbench import (
    load_dataset,
    select_balanced_indices,
    task_from_dataset_item,
    task_id,
)
from learnbyai_research.longgenbench_v2 import compile_longgen_v2_contract


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())


def select_indices(
    dataset: list[dict[str, Any]],
    *,
    run_all: bool,
    per_type: int | None,
    indices: list[int] | None,
) -> list[int]:
    if run_all:
        return list(range(len(dataset)))
    if indices is not None:
        for index in indices:
            if index < 0 or index >= len(dataset):
                raise ValueError(f"dataset index out of range: {index}")
        return list(indices)
    if per_type is not None:
        return select_balanced_indices(dataset, per_type)
    raise ValueError("one of --all, --per-type, or --indices is required")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fast Stage-1 diagnostic for ContractFlow v2 sparse contract compiler on LongGenBench."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--per-type", type=positive_int)
    selection.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--workers", type=positive_int, default=16)
    parser.add_argument("--plan-max-output-tokens", type=positive_int, default=2048)
    parser.add_argument("--retry-cap-tokens", type=positive_int, default=8192)
    parser.add_argument("--env-prefix", default="AUTHOR")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    return parser


def evaluate_compiled_item(
    dataset_item: dict[str, Any],
    contract_dict: dict[str, Any],
) -> dict[str, Any]:
    """Compare compiled unit obligations against dataset check indices (index-level recall/FP)."""
    pred_indices = {
        int(unit["index"])
        for unit in contract_dict.get("units", [])
        if isinstance(unit, dict) and unit.get("obligations")
    }
    by_category: dict[str, dict[str, Any]] = {}
    gold_all: set[int] = set()
    for cat in ("once", "range", "periodic"):
        raw = dataset_item.get(f"checks_{cat}", {}) or {}
        cat_gold = {int(k) for k in raw.keys()}
        gold_all.update(cat_gold)
        hit = len(cat_gold & pred_indices)
        missed = sorted(cat_gold - pred_indices)
        by_category[cat] = {
            "gold": len(cat_gold),
            "hit": hit,
            "missed_count": len(missed),
            "missed_sample": missed[:10],
        }
    false_positives = sorted(pred_indices - gold_all)
    return {
        "gold_checked_units": len(gold_all),
        "pred_constrained_units": len(pred_indices),
        "hit_checked_units": len(gold_all & pred_indices),
        "missed_checked_units": len(gold_all - pred_indices),
        "false_positive_units": len(false_positives),
        "false_positive_sample": false_positives[:10],
        "by_category": by_category,
    }


def summarize_compiler_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(rows)
        ok_rows = [r for r in rows if r.get("status") == "ok"]
        failed = total - len(ok_rows)
        gold_units = sum(r["metrics"]["gold_checked_units"] for r in ok_rows)
        hit_units = sum(r["metrics"]["hit_checked_units"] for r in ok_rows)
        missed_units = sum(r["metrics"]["missed_checked_units"] for r in ok_rows)
        fp_units = sum(r["metrics"]["false_positive_units"] for r in ok_rows)
        cat_stats: dict[str, dict[str, Any]] = {}
        for cat in ("once", "range", "periodic"):
            g = sum(r["metrics"]["by_category"][cat]["gold"] for r in ok_rows)
            h = sum(r["metrics"]["by_category"][cat]["hit"] for r in ok_rows)
            cat_stats[cat] = {
                "gold": g,
                "hit": h,
                "recall": h / g if g else 0.0,
                "miss_rate": (g - h) / g if g else 0.0,
            }
        completion_tokens = [
            int(call.get("usage", {}).get("completion_tokens", 0))
            for r in ok_rows
            for call in r.get("completion_history", [])
        ]
        prompt_tokens = [
            int(call.get("usage", {}).get("prompt_tokens", 0))
            for r in ok_rows
            for call in r.get("completion_history", [])
        ]
        elapsed = [float(r.get("elapsed_seconds", 0.0)) for r in ok_rows]
        return {
            "tasks": total,
            "ok": len(ok_rows),
            "failed": failed,
            "gold_checked_units": gold_units,
            "hit_checked_units": hit_units,
            "missed_checked_units": missed_units,
            "unit_recall": hit_units / gold_units if gold_units else 0.0,
            "unit_miss_rate": missed_units / gold_units if gold_units else 0.0,
            "false_positive_units": fp_units,
            "avg_fp_per_task": fp_units / len(ok_rows) if ok_rows else 0.0,
            "by_category": cat_stats,
            "avg_completion_tokens": (
                sum(completion_tokens) / len(ok_rows) if ok_rows else 0.0
            ),
            "avg_prompt_tokens": sum(prompt_tokens) / len(ok_rows) if ok_rows else 0.0,
            "avg_elapsed_seconds": sum(elapsed) / len(ok_rows) if ok_rows else 0.0,
        }

    summary: dict[str, Any] = {"overall": _agg(records), "by_type": {}}
    types = sorted({str(r["type"]) for r in records if r.get("type")})
    for seg_type in types:
        summary["by_type"][seg_type] = _agg([r for r in records if r.get("type") == seg_type])
    return summary


def format_compiler_table(summary: dict[str, Any]) -> str:
    header = (
        f"{'Group':<14} {'OK/N':>7} {'Recall%':>8} {'Miss%':>7} "
        f"{'Once%':>7} {'Range%':>7} {'Per%':>7} {'FP_Units':>9} {'AvgOutTok':>10}"
    )
    lines = ["", header, "-" * len(header)]

    def _row(label: str, stats: dict[str, Any]) -> str:
        bc = stats["by_category"]
        return (
            f"{label:<14} {stats['ok']:>3d}/{stats['tasks']:<3d} "
            f"{stats['unit_recall'] * 100:>7.2f}% "
            f"{stats['unit_miss_rate'] * 100:>6.2f}% "
            f"{bc['once']['recall'] * 100:>6.2f}% "
            f"{bc['range']['recall'] * 100:>6.2f}% "
            f"{bc['periodic']['recall'] * 100:>6.2f}% "
            f"{stats['false_positive_units']:>9d} "
            f"{stats['avg_completion_tokens']:>10.1f}"
        )

    lines.append(_row("OVERALL", summary["overall"]))
    lines.append("-" * len(header))
    for seg_type, stats in summary["by_type"].items():
        lines.append(_row(seg_type, stats))
    return "\n".join(lines)


def compile_one(
    args: argparse.Namespace,
    dataset: list[dict[str, Any]],
    index: int,
) -> tuple[str, dict[str, Any]]:
    task = task_from_dataset_item(index, dataset[index])
    tid = task_id(task)
    cache_path = Path(args.out_dir) / "items" / f"{tid}.json"
    if cache_path.exists():
        return "cached", read_json(cache_path)

    client = OpenAICompatibleClient.from_env(args.env_prefix)
    started = time.perf_counter()
    try:
        contract = compile_longgen_v2_contract(
            task,
            client,
            max_output_tokens=args.plan_max_output_tokens,
            retry_cap_tokens=args.retry_cap_tokens,
        )
        contract_dict = contract.to_dict()
        metrics = evaluate_compiled_item(dataset[index], contract_dict)
        record: dict[str, Any] = {
            "task_id": tid,
            "dataset_index": index,
            "type": task.segment_type,
            "segment_count": task.segment_count,
            "status": "ok",
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "metrics": metrics,
            "rules": contract_dict["rules"],
            "global_brief": contract_dict["global_brief"],
            "global_invariants": contract_dict["global_invariants"],
            "completion_history": completion_history(client),
        }
        status = "ok"
    except Exception as error:
        record = {
            "task_id": tid,
            "dataset_index": index,
            "type": task.segment_type,
            "segment_count": task.segment_count,
            "status": "error",
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "completion_history": completion_history(client),
        }
        status = "error"

    write_json(cache_path, record)
    return status, record


def main() -> None:
    args = build_parser().parse_args()
    if args.env_file:
        load_env(Path(args.env_file))
    dataset = load_dataset(args.dataset)
    indices = select_indices(
        dataset,
        run_all=bool(args.all),
        per_type=args.per_type,
        indices=args.indices,
    )
    records: list[dict[str, Any]] = []
    total = len(indices)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(compile_one, args, dataset, idx): idx
            for idx in indices
        }
        for future in as_completed(futures):
            idx = futures[future]
            done += 1
            status, rec = future.result()
            records.append(rec)
            print(f"[{done}/{total}] {status}: dataset_index={idx} ({rec['type']})", flush=True)

    records.sort(key=lambda r: int(r["dataset_index"]))
    summary = summarize_compiler_records(records)
    summary_path = Path(args.out_dir) / "compiler_summary.json"
    write_json(summary_path, summary)
    print(format_compiler_table(summary), flush=True)
    print(f"\nsummary written to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
