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

from learnbyai_research.llm import OpenAICompatibleClient
from learnbyai_research.longgenbench import (
    load_dataset,
    run_direct,
    run_harness,
    select_balanced_indices,
    task_from_dataset_item,
    task_id,
    task_record,
)
from learnbyai_research.longgenbench_optimized import run_harness_optimized


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
    parser = argparse.ArgumentParser(description="Run ContractFlow on LongGenBench (16K and 32K).")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=[
            "direct",
            "contractflow",
            "contractflow_optimized",
            "no_contract",
            "no_review_repair",
            "neither",
        ],
        required=True,
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--all",
        action="store_true",
        help="Run every item in the dataset (e.g., all 400 items in 16K or 32K).",
    )
    selection.add_argument("--per-type", type=positive_int)
    selection.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--plan-max-output-tokens", type=positive_int, default=4096)
    parser.add_argument(
        "--plan-chunk-size",
        type=positive_int,
        default=60,
        help="Max segments planned/audited per LLM call in contractflow (default: 60).",
    )
    parser.add_argument("--generation-max-output-tokens", type=positive_int, default=4096)
    parser.add_argument("--review-max-output-tokens", type=positive_int, default=4096)
    parser.add_argument("--repair-max-output-tokens", type=positive_int, default=4096)
    parser.add_argument("--direct-max-output-tokens", type=positive_int, default=16384)
    parser.add_argument(
        "--no-llm-plan",
        action="store_true",
        help="Skip LLM planning/audit calls in contractflow and use deterministic prompt compilation.",
    )
    parser.add_argument(
        "--max-singleton-retries",
        type=positive_int,
        default=2,
        help="Singleton recovery retries for contractflow_optimized.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record task failures under errors/<system>/ and continue remaining jobs.",
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


def run(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.dataset)
    indices = select_indices(
        dataset,
        run_all=bool(getattr(args, "all", False)),
        per_type=getattr(args, "per_type", None),
        indices=getattr(args, "indices", None),
    )
    jobs = [(system, index) for index in indices for system in args.systems]
    total = len(jobs)
    completed_count = 0
    failed_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate_one, args, dataset, system, index): (system, index)
            for system, index in jobs
        }
        for future in as_completed(futures):
            system, index = futures[future]
            completed_count += 1
            status = future.result()
            if status == "failed":
                failed_count += 1
            print(
                f"[{completed_count}/{total}] {status}: {system} dataset_index={index}",
                flush=True,
            )
    if failed_count:
        print(
            f"finished with {failed_count} failed job(s); see {Path(args.run_dir) / 'errors'}",
            flush=True,
        )


def generate_one(
    args: argparse.Namespace,
    dataset: list[dict[str, Any]],
    system: str,
    index: int,
) -> str:
    task = task_from_dataset_item(index, dataset[index])
    tid = task_id(task)
    run_dir = Path(args.run_dir)
    output_path = run_dir / "generations" / system / f"{tid}.json"
    if output_path.exists():
        return "skipped"
    error_path = run_dir / "errors" / system / f"{tid}.json"
    client = OpenAICompatibleClient.from_env("AUTHOR")
    try:
        if system == "direct":
            output = run_direct(
                task,
                client,
                max_output_tokens=args.direct_max_output_tokens,
            )
        elif system == "contractflow_optimized":
            output = run_harness_optimized(
                task,
                client,
                batch_size=args.batch_size,
                generation_max_output_tokens=args.generation_max_output_tokens,
                review_max_output_tokens=args.review_max_output_tokens,
                repair_max_output_tokens=args.repair_max_output_tokens,
                max_singleton_retries=args.max_singleton_retries,
                checkpoint_path=run_dir / "checkpoints" / system / f"{tid}.json",
            )
            output["system"] = system
        else:
            output = run_harness(
                task,
                client,
                batch_size=args.batch_size,
                plan_max_output_tokens=args.plan_max_output_tokens,
                generation_max_output_tokens=args.generation_max_output_tokens,
                review_max_output_tokens=args.review_max_output_tokens,
                repair_max_output_tokens=args.repair_max_output_tokens,
                checkpoint_path=run_dir / "checkpoints" / system / f"{tid}.json",
                use_llm_plan=not getattr(args, "no_llm_plan", False),
                use_contracts=system not in {"no_contract", "neither"},
                use_review_repair=system not in {"no_review_repair", "neither"},
                plan_chunk_size=getattr(args, "plan_chunk_size", 60),
            )
    except Exception as error:
        if not getattr(args, "continue_on_error", False):
            raise
        write_json(
            error_path,
            {
                "task": task_record(task),
                "system": system,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "completion_history": client.completion_history(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return "failed"

    write_json(output_path, output)
    if error_path.exists():
        error_path.unlink()
    return "completed"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
