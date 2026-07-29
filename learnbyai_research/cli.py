from __future__ import annotations

import argparse
from typing import cast

from .eval.aggregate import aggregate_pairwise, read_judgments, write_csv
from .eval.contract_metrics import aggregate_contract_metrics, read_generation_outputs, write_contract_metrics_csv
from .eval.task_spec import evaluate_task_spec_outputs, write_task_spec_metrics_csv
from .eval.judge import PairwiseJudge
from .io_utils import export_chapter_markdown_files, load_output, load_task, write_json
from .llm import MockLLM, OpenAICompatibleClient
from .pipelines import build_pipeline
from .schemas import PipelineName


PIPELINES = ["direct_prompt", "single_agent", "full", "no_contract", "no_material_routing", "no_review_repair"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="contractflow")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate one textbook output for a task.")
    gen.add_argument("--task", required=True)
    gen.add_argument("--pipeline", choices=PIPELINES, required=True)
    gen.add_argument("--output", required=True)
    gen.add_argument("--mock", action="store_true")

    judge = sub.add_parser("judge", help="Pairwise judge two generated outputs.")
    judge.add_argument("--task", required=True)
    judge.add_argument("--a", required=True)
    judge.add_argument("--b", required=True)
    judge.add_argument("--output", required=True)
    judge.add_argument("--mock", action="store_true")

    agg = sub.add_parser("aggregate", help="Aggregate pairwise judgments.")
    agg.add_argument("--judgments", nargs="+", required=True)
    agg.add_argument("--output", required=True)

    contract_metrics = sub.add_parser("contract-metrics", help="Aggregate contract verification metrics from generation outputs.")
    contract_metrics.add_argument("--generations", nargs="+", required=True)
    contract_metrics.add_argument("--output", required=True)

    task_spec_metrics = sub.add_parser("task-spec-metrics", help="Evaluate every output against frozen task metadata.")
    task_spec_metrics.add_argument("--tasks", nargs="+", required=True)
    task_spec_metrics.add_argument("--generations", nargs="+", required=True)
    task_spec_metrics.add_argument("--output", required=True)

    export_chapters = sub.add_parser("export-chapters", help="Export only chapter Markdown from a generation JSON.")
    export_chapters.add_argument("--generation", required=True)
    export_chapters.add_argument("--output-dir", required=True)

    args = parser.parse_args(argv)
    if args.command == "generate":
        llm = _llm(args.mock, "AUTHOR")
        reviewer_llm = _optional_llm(args.mock, "REVIEW") or llm
        repair_llm = _optional_llm(args.mock, "REPAIR") or llm
        task = load_task(args.task)
        pipeline = build_pipeline(
            cast(PipelineName, args.pipeline),
            llm,
            reviewer_llm=reviewer_llm,
            repair_llm=repair_llm,
        )
        output = pipeline.run(task)
        write_json(args.output, output)
        print(f"wrote {args.output}")
    elif args.command == "judge":
        llm = _llm(args.mock, "JUDGE")
        task = load_task(args.task)
        output_a = load_output(args.a)
        output_b = load_output(args.b)
        judgment = PairwiseJudge(llm).judge(task, output_a, output_b)
        write_json(args.output, judgment)
        print(f"wrote {args.output}")
    elif args.command == "aggregate":
        rows = aggregate_pairwise(read_judgments(args.judgments))
        write_csv(args.output, rows)
        print(f"wrote {args.output}")
    elif args.command == "contract-metrics":
        rows = aggregate_contract_metrics(read_generation_outputs(args.generations))
        write_contract_metrics_csv(args.output, rows)
        print(f"wrote {args.output}")
    elif args.command == "task-spec-metrics":
        tasks = [load_task(path) for path in args.tasks]
        rows = evaluate_task_spec_outputs({task.id: task for task in tasks}, read_generation_outputs(args.generations))
        write_task_spec_metrics_csv(args.output, rows)
        print(f"wrote {args.output}")
    elif args.command == "export-chapters":
        output = load_output(args.generation)
        written = export_chapter_markdown_files(output, args.output_dir)
        for path in written:
            print(f"wrote {path}")


def _llm(mock: bool, prefix: str):
    return MockLLM() if mock else OpenAICompatibleClient.from_env(prefix)


def _optional_llm(mock: bool, prefix: str):
    if mock:
        return MockLLM()
    if not _has_prefixed_config(prefix):
        return None
    return OpenAICompatibleClient.from_env(prefix)


def _has_prefixed_config(prefix: str) -> bool:
    import os

    return bool(os.environ.get(f"{prefix}_API_KEY"))


if __name__ == "__main__":
    main()
