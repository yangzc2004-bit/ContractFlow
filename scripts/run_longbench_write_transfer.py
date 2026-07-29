from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learnbyai_research.llm import OpenAICompatibleClient
from learnbyai_research.longbench_write_transfer import (
    MORELONGWRITE_OPTIMIZED_PROFILE,
    load_dataset,
    run_agentwrite,
    run_direct,
    run_harness,
    task_from_record,
    write_json,
)
from learnbyai_research.morelongwrite_optimized import run_morelongwrite_optimized


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configured LongBench-Write transfer study.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=["direct", "harness_full", "agentwrite"],
        required=True,
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip the shared manifest write when launched by a parallel orchestrator.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--max-task-attempts",
        type=int,
        required=True,
        help="Materialize a technical-failure generation after this many failed attempts.",
    )
    args = parser.parse_args()

    load_env(ROOT / ".env")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    dataset_path = ROOT / config["benchmark"]["dataset"]
    dataset = load_dataset(dataset_path)
    indices = args.indices or config["benchmark"]["indices"]
    run_dir = Path(args.run_dir)
    manifest = build_manifest(config, dataset_path, indices, args.systems)
    if not args.skip_manifest:
        write_json(run_dir / "experiment_manifest.json", manifest)
    if args.preflight:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    for index in indices:
        task = task_from_record(index, dataset[index])
        for system in args.systems:
            output_path = run_dir / "generations" / system / f"{task.task_id}.json"
            if output_path.exists():
                print(f"exists: {system} index={index}", flush=True)
                continue
            print(f"starting: {system} index={index}", flush=True)
            error_path = run_dir / "errors" / system / f"{task.task_id}.json"
            try:
                result = generate_one(system, task, config, run_dir)
                write_json(output_path, result)
                if error_path.exists():
                    error_path.unlink()
                print(
                    f"wrote: {system} index={index} words={result['metadata']['word_count']} "
                    f"ratio={result['metadata']['length_ratio']}",
                    flush=True,
                )
            except Exception as error:
                prior_attempts = 0
                if error_path.exists():
                    try:
                        prior_attempts = int(
                            json.loads(error_path.read_text(encoding="utf-8")).get(
                                "attempts",
                                0,
                            )
                        )
                    except (json.JSONDecodeError, OSError, ValueError):
                        prior_attempts = 0
                attempts = prior_attempts + 1
                error_record = {
                    "task": task_record(task),
                    "system": system,
                    "attempts": attempts,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                write_json(error_path, error_record)
                print(
                    f"failed: {system} index={index} attempts={attempts} "
                    f"error={type(error).__name__}: {error}",
                    flush=True,
                )
                if args.max_task_attempts and attempts >= args.max_task_attempts:
                    write_json(
                        output_path,
                        technical_failure_record(task, system, error_record),
                    )
                    print(
                        f"materialized technical failure: {system} index={index}",
                        flush=True,
                    )
                elif not args.continue_on_error:
                    raise


def generate_one(system, task, config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    generation = config["generation"]
    if system == "direct":
        client = make_client("AUTHOR", include_default_system=False)
        return run_direct(task, client, **generation["direct"])
    if system == "agentwrite":
        planning_client = make_client("AUTHOR", include_default_system=False)
        return run_agentwrite(
            task,
            planning_client,
            lambda: make_client("AUTHOR", include_default_system=False),
            source_root=ROOT / "third_party" / "LongWriter",
            checkpoint_dir=run_dir / "checkpoints" / system / task.task_id,
            **generation["agentwrite"],
        )
    adapter_name = config["task_family_adapters"][task.task_type]
    harness_generation = dict(generation["harness_full"])
    profile = str(harness_generation.get("profile", "legacy_longbench_write"))
    all_roles_use_author_model = bool(
        harness_generation.pop("all_roles_use_author_model", False)
    )
    review_prefix = "AUTHOR" if all_roles_use_author_model else "REVIEW"
    repair_prefix = "AUTHOR" if all_roles_use_author_model else "REPAIR"
    if profile == MORELONGWRITE_OPTIMIZED_PROFILE:
        harness_generation.pop("profile", None)
        return run_morelongwrite_optimized(
            task,
            adapter_name,
            make_client("AUTHOR", include_default_system=True),
            lambda: make_client("AUTHOR", include_default_system=True),
            lambda: make_client(review_prefix, include_default_system=True),
            lambda: make_client(repair_prefix, include_default_system=True),
            checkpoint_dir=run_dir / "checkpoints" / system / task.task_id,
            **harness_generation,
        )
    return run_harness(
        task,
        adapter_name,
        make_client("AUTHOR", include_default_system=True),
        lambda: make_client(review_prefix, include_default_system=True),
        lambda: make_client(repair_prefix, include_default_system=True),
        checkpoint_dir=run_dir / "checkpoints" / system / task.task_id,
        **harness_generation,
    )


def make_client(prefix: str, *, include_default_system: bool) -> OpenAICompatibleClient:
    client = OpenAICompatibleClient.from_env(prefix)
    client.include_default_system = include_default_system
    return client


def task_record(task) -> dict[str, Any]:
    return {
        "index": task.index,
        "task_id": task.task_id,
        "prompt": task.prompt,
        "type": task.task_type,
        "target_words": task.target_words,
    }


def technical_failure_record(
    task,
    system: str,
    error_record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": task_record(task),
        "system": system,
        "final_text": "",
        "plan": None,
        "units": None,
        "metadata": {
            "elapsed_seconds": 0.0,
            "word_count": 0,
            "length_count_rule": (
                "Chinese characters plus regex-matched English words"
            ),
            "target_words": task.target_words,
            "length_ratio": 0.0,
            "completion_history": [],
            "adaptations": ["technical failure retained without model substitution"],
            "technical_failure": True,
            "technical_failure_attempts": error_record["attempts"],
            "technical_failure_type": error_record["error_type"],
            "technical_failure_message": error_record["error_message"],
        },
    }


def build_manifest(
    config: dict[str, Any],
    dataset_path: Path,
    indices: list[int],
    systems: list[str],
) -> dict[str, Any]:
    source_root = ROOT / "third_party" / "LongWriter"
    return {
        "protocol": config,
        "dataset_path": str(dataset_path.resolve()),
        "indices": indices,
        "systems": systems,
        "author": client_manifest("AUTHOR"),
        "reviewer": client_manifest("REVIEW"),
        "repairer": client_manifest("REPAIR"),
        "agentwrite_source_head": git_head(source_root),
    }


def client_manifest(prefix: str) -> dict[str, Any]:
    return {
        "model": os.environ.get(f"{prefix}_MODEL"),
        "base_url": os.environ.get(f"{prefix}_BASE_URL"),
        "wire_api": os.environ.get(f"{prefix}_WIRE_API", "chat"),
        "thinking": os.environ.get(f"{prefix}_THINKING"),
        "stream": os.environ.get(f"{prefix}_STREAM"),
        "timeout": os.environ.get(f"{prefix}_TIMEOUT"),
    }


def git_head(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())


if __name__ == "__main__":
    main()
