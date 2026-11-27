from __future__ import annotations

import argparse
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


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
from learnbyai_research.longgenbench_agentwrite import run_agentwrite
from learnbyai_research.longgenbench_cogwriter_official import (
    resolve_official_parameter_profile,
    run_official_cogwriter,
)
from learnbyai_research.longgenbench_self_refine import run_self_refine


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
    parser = argparse.ArgumentParser(description="Run LongGenBench comparison systems.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--all",
        action="store_true",
        help="Run every item in the dataset (e.g., all 400 items in 16K or 32K).",
    )
    selection.add_argument("--per-type", type=positive_int)
    selection.add_argument("--indices", nargs="+", type=int)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=["agentwrite", "cogwriter", "self_refine"],
        required=True,
    )
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--workers", type=positive_int, default=1)
    parser.add_argument("--planning-max-output-tokens", type=positive_int, default=16_384)
    parser.add_argument("--segment-max-output-tokens", type=positive_int, default=4_096)
    parser.add_argument("--feedback-max-output-tokens", type=positive_int, default=2_048)
    parser.add_argument("--refine-max-output-tokens", type=positive_int, default=4_096)
    parser.add_argument("--max-plan-attempts", type=positive_int, default=3)
    parser.add_argument("--maximum-concurrent-requests", type=positive_int, default=16)
    parser.add_argument("--planning-validation-attempts", type=positive_int, default=3)
    parser.add_argument("--max-refinements", type=positive_int, default=5)
    parser.add_argument("--self-refine-iterations", type=positive_int, default=1)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature override (defaults to 1.0 for agentwrite, 0.2 for cogwriter).",
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

    dataset = load_dataset(args.dataset)
    indices = select_indices(
        dataset,
        run_all=bool(args.all),
        per_type=args.per_type,
        indices=args.indices,
    )
    jobs = [(system, index) for index in indices for system in args.systems]
    total = len(jobs)
    completed_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one_job, args, dataset, system, index): (system, index)
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


def run_one_job(
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
    try:
        output = generate_one(args, dataset[index], task, system)
        write_json(output_path, output)
        if error_path.exists():
            error_path.unlink()
        return "completed"
    except Exception as error:
        if not args.continue_on_error:
            raise
        write_json(
            error_path,
            {
                "dataset_index": index,
                "task_id": tid,
                "system": system,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        return "failed"


def generate_one(
    args: argparse.Namespace,
    dataset_item: dict[str, Any],
    task,
    system: str,
) -> dict[str, Any]:
    checkpoint_dir = (
        Path(args.run_dir) / "checkpoints" / system / task_id(task)
    )
    client_factory: Callable[[], OpenAICompatibleClient] = (
        lambda: OpenAICompatibleClient.from_env("AUTHOR")
    )
    if system == "agentwrite":
        temp = 1.0 if args.temperature is None else args.temperature
        return run_agentwrite(
            task,
            client_factory,
            client_factory,
            checkpoint_dir=checkpoint_dir,
            planning_max_output_tokens=args.planning_max_output_tokens,
            segment_max_output_tokens=args.segment_max_output_tokens,
            max_plan_attempts=args.max_plan_attempts,
            temperature=temp,
        )
    if system == "cogwriter":
        if not args.source_root:
            raise ValueError("--source-root is required for CogWriter")
        temp = 0.2 if args.temperature is None else args.temperature

        def official_client() -> OpenAICompatibleClient:
            client = client_factory()
            client.include_default_system = False
            return client

        output = run_official_cogwriter(
            task,
            dataset_item,
            official_client,
            source_root=args.source_root,
            model_name=official_client().model,
            maximum_concurrent_requests=args.maximum_concurrent_requests,
            planning_max_output_tokens=args.planning_max_output_tokens,
            segment_max_output_tokens=args.segment_max_output_tokens,
            temperature=temp,
            max_refinements=args.max_refinements,
            parameter_profile=resolve_official_parameter_profile(task),
            planning_validation_attempts=args.planning_validation_attempts,
        )
        output["system"] = "cogwriter"
        return output

    direct_path = (
        Path(args.run_dir)
        / "generations"
        / "direct"
        / f"{task_id(task)}.json"
    )
    if not direct_path.exists():
        raise FileNotFoundError(
            f"Self-Refine requires the matching Direct output: {direct_path}"
        )
    draft = str(read_json(direct_path).get("final_text", ""))
    return run_self_refine(
        task,
        draft,
        client_factory,
        client_factory,
        checkpoint_dir=checkpoint_dir,
        iterations=args.self_refine_iterations,
        feedback_max_output_tokens=args.feedback_max_output_tokens,
        refine_max_output_tokens=args.refine_max_output_tokens,
    )


if __name__ == "__main__":
    main()
