from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learnbyai_research.io_utils import read_json, write_json
from learnbyai_research.llm import OpenAICompatibleClient
from learnbyai_research.longgenbench import (
    load_dataset,
    task_from_dataset_item,
    task_id,
)
from learnbyai_research.longgenbench_agentwrite import run_agentwrite
from learnbyai_research.longgenbench_cogwriter_official import (
    official_parameter_profile,
    run_official_cogwriter,
)
from learnbyai_research.longgenbench_self_refine import run_self_refine


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LongGenBench comparison systems.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--indices", nargs="+", type=int, required=True)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=["agentwrite", "cogwriter", "self_refine"],
        required=True,
    )
    parser.add_argument("--source-root")
    parser.add_argument("--planning-max-output-tokens", type=positive_int, required=True)
    parser.add_argument("--segment-max-output-tokens", type=positive_int, required=True)
    parser.add_argument("--feedback-max-output-tokens", type=positive_int, required=True)
    parser.add_argument("--refine-max-output-tokens", type=positive_int, required=True)
    parser.add_argument("--max-plan-attempts", type=positive_int, required=True)
    parser.add_argument("--maximum-concurrent-requests", type=positive_int, required=True)
    parser.add_argument("--planning-validation-attempts", type=positive_int, required=True)
    parser.add_argument("--max-refinements", type=positive_int, required=True)
    parser.add_argument("--self-refine-iterations", type=positive_int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    for index in args.indices:
        if index < 0 or index >= len(dataset):
            raise ValueError(f"dataset index out of range: {index}")
        task = task_from_dataset_item(index, dataset[index])
        for system in args.systems:
            output_path = (
                Path(args.run_dir)
                / "generations"
                / system
                / f"{task_id(task)}.json"
            )
            if output_path.exists():
                continue
            output = generate_one(args, dataset[index], task, system)
            write_json(output_path, output)
            print(f"completed: {system} dataset_index={index}", flush=True)


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
        return run_agentwrite(
            task,
            client_factory,
            client_factory,
            checkpoint_dir=checkpoint_dir,
            planning_max_output_tokens=args.planning_max_output_tokens,
            segment_max_output_tokens=args.segment_max_output_tokens,
            max_plan_attempts=args.max_plan_attempts,
        )
    if system == "cogwriter":
        if not args.source_root:
            raise ValueError("--source-root is required for CogWriter")

        def official_client() -> OpenAICompatibleClient:
            client = client_factory()
            client.include_default_system = False
            return client

        return run_official_cogwriter(
            task,
            dataset_item,
            official_client,
            source_root=args.source_root,
            model_name=official_client().model,
            maximum_concurrent_requests=args.maximum_concurrent_requests,
            planning_max_output_tokens=args.planning_max_output_tokens,
            segment_max_output_tokens=args.segment_max_output_tokens,
            temperature=args.temperature,
            max_refinements=args.max_refinements,
            parameter_profile=official_parameter_profile(task),
            planning_validation_attempts=args.planning_validation_attempts,
        )

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
