from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learnbyai_research.io_utils import read_json, write_json
from learnbyai_research.llm import OpenAICompatibleClient
from learnbyai_research.longgenbench import (
    load_dataset,
    select_balanced_indices,
    task_from_dataset_item,
    task_id,
)
from learnbyai_research.longgenbench_eval import (
    evaluate_output,
    failed_evaluation,
    summarize_evaluations,
)


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
    parser = argparse.ArgumentParser(description="Evaluate LongGenBench generations.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--systems", nargs="+", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--per-type", type=positive_int)
    selection.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--judge-batch-size", type=positive_int, default=20)
    parser.add_argument("--judge-max-output-tokens", type=positive_int, default=4096)
    parser.add_argument(
        "--judge-env-prefix",
        default="JUDGE",
        help="Environment variable prefix for the judge model (default: JUDGE).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record judge failures and continue evaluating remaining tasks.",
    )
    parser.add_argument(
        "--env-file",
        default=str(ROOT / ".env"),
        help="Optional .env file loaded with setdefault before reading environment variables.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.env_file:
        load_env(Path(args.env_file))
    run(args)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_dataset(args.dataset)
    indices = select_indices(
        dataset,
        run_all=bool(getattr(args, "all", False)),
        per_type=getattr(args, "per_type", None),
        indices=getattr(args, "indices", None),
    )
    jobs = [(system, index) for index in indices for system in args.systems]
    total = len(jobs)
    done = 0
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(evaluate_one, args, dataset, system, index): (system, index)
            for system, index in jobs
        }
        for future in as_completed(futures):
            system, index = futures[future]
            done += 1
            status, row = future.result()
            if row is not None:
                rows.append(row)
            print(
                f"[{done}/{total}] {status}: {system} dataset_index={index}",
                flush=True,
            )

    summary = summarize_evaluations(rows)
    summary_path = Path(args.run_dir) / "evaluations" / "summary.json"
    write_json(summary_path, summary)
    print(format_summary_table(summary), flush=True)
    print(f"\nsummary written to: {summary_path}", flush=True)
    return summary


def evaluate_one(
    args: argparse.Namespace,
    dataset: list[dict[str, Any]],
    system: str,
    index: int,
) -> tuple[str, dict[str, Any] | None]:
    task = task_from_dataset_item(index, dataset[index])
    tid = task_id(task)
    run_dir = Path(args.run_dir)
    eval_path = run_dir / "evaluations" / system / f"{tid}.json"
    if eval_path.exists():
        return "cached", read_json(eval_path)

    gen_path = run_dir / "generations" / system / f"{tid}.json"
    err_path = run_dir / "errors" / system / f"{tid}.json"
    if not gen_path.exists():
        if err_path.exists():
            row = failed_evaluation(index, dataset[index], system, read_json(err_path))
            write_json(eval_path, row)
            return "failed_generation", row
        return "missing_generation", None

    output = read_json(gen_path)
    output.setdefault("system", system)
    client = OpenAICompatibleClient.from_env(args.judge_env_prefix)
    try:
        row = evaluate_output(
            dataset[index],
            output,
            client,
            batch_size=args.judge_batch_size,
            max_output_tokens=args.judge_max_output_tokens,
        )
    except Exception as error:
        if not getattr(args, "continue_on_error", False):
            raise
        write_json(
            run_dir / "evaluations" / "errors" / system / f"{tid}.json",
            {
                "task_id": tid,
                "dataset_index": index,
                "system": system,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return "judge_error", None

    write_json(eval_path, row)
    return "evaluated", row


def format_summary_table(summary: dict[str, Any]) -> str:
    systems = summary.get("systems", {})
    if not systems:
        return "No evaluation rows found."
    header = (
        f"{'System':<24} {'N':>5} {'Comp%':>8} {'Len%':>8} "
        f"{'Once%':>8} {'Range%':>8} {'Per%':>8} {'Avg%':>8}"
    )
    lines = ["", header, "-" * len(header)]
    for system, stats in systems.items():
        lines.append(
            f"{system:<24} {stats['task_count']:>5d} "
            f"{stats['completion_rate'] * 100:>7.2f}% "
            f"{stats['length_pass_rate'] * 100:>7.2f}% "
            f"{stats['accuracy_once'] * 100:>7.2f}% "
            f"{stats['accuracy_range'] * 100:>7.2f}% "
            f"{stats['accuracy_periodic'] * 100:>7.2f}% "
            f"{stats['average_accuracy'] * 100:>7.2f}%"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
