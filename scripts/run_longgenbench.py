from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ContractFlow on LongGenBench.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=["direct", "contractflow"],
        required=True,
    )
    parser.add_argument("--per-type", type=positive_int, required=True)
    parser.add_argument("--workers", type=positive_int, required=True)
    parser.add_argument("--batch-size", type=positive_int, required=True)
    parser.add_argument("--plan-max-output-tokens", type=positive_int, required=True)
    parser.add_argument("--generation-max-output-tokens", type=positive_int, required=True)
    parser.add_argument("--review-max-output-tokens", type=positive_int, required=True)
    parser.add_argument("--repair-max-output-tokens", type=positive_int, required=True)
    parser.add_argument("--direct-max-output-tokens", type=positive_int, required=True)
    args = parser.parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.dataset)
    indices = select_balanced_indices(dataset, args.per_type)
    jobs = [(system, index) for index in indices for system in args.systems]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate_one, args, dataset, system, index): (system, index)
            for system, index in jobs
        }
        for future in as_completed(futures):
            system, index = futures[future]
            future.result()
            print(f"completed: {system} dataset_index={index}", flush=True)


def generate_one(
    args: argparse.Namespace,
    dataset: list[dict[str, Any]],
    system: str,
    index: int,
) -> None:
    task = task_from_dataset_item(index, dataset[index])
    output_path = Path(args.run_dir) / "generations" / system / f"{task_id(task)}.json"
    if output_path.exists():
        return
    client = OpenAICompatibleClient.from_env("AUTHOR")
    if system == "direct":
        output = run_direct(
            task,
            client,
            max_output_tokens=args.direct_max_output_tokens,
        )
    else:
        output = run_harness(
            task,
            client,
            batch_size=args.batch_size,
            plan_max_output_tokens=args.plan_max_output_tokens,
            generation_max_output_tokens=args.generation_max_output_tokens,
            review_max_output_tokens=args.review_max_output_tokens,
            repair_max_output_tokens=args.repair_max_output_tokens,
            checkpoint_path=(
                Path(args.run_dir)
                / "checkpoints"
                / system
                / f"{task_id(task)}.json"
            ),
        )
    write_json(output_path, output)


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
