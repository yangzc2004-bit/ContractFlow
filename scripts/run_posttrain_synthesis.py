from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learnbyai_research.contractflow_v2 import FlowConfig
from learnbyai_research.llm import OpenAICompatibleClient
from learnbyai_research.posttrain_data import (
    export_canary_review_markdown,
    export_sft_datasets,
    synthesize_posttrain_sample,
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


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Self-bootstrap long-form SFT data via ContractFlow v2, run 6-gate quality filtering, and export dual SFT JSONLs."
    )
    parser.add_argument("--prompts-jsonl", required=True, help="Path to prompts.jsonl from generate_posttrain_prompts.py.")
    parser.add_argument("--out-dir", required=True, help="Output directory for generations, checkpoints, SFT exports, and review report.")
    parser.add_argument("--env-prefix", default="AUTHOR", help="Env prefix for self-bootstrapping model (default: AUTHOR, e.g., Qwen3.5-9B).")
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--plan-max-output-tokens", type=positive_int, default=2048)
    parser.add_argument("--generation-max-output-tokens", type=positive_int, default=6144)
    parser.add_argument("--review-max-output-tokens", type=positive_int, default=4096)
    parser.add_argument("--repair-max-output-tokens", type=positive_int, default=4096)
    parser.add_argument("--max-singleton-retries", type=positive_int, default=2)
    parser.add_argument("--length-retry-ratio", type=float, default=0.7)
    parser.add_argument(
        "--draft-temperature",
        type=float,
        default=0.65,
        help="Sampling temperature for prose drafting during SFT synthesis (default: 0.65; compile/review stay deterministic).",
    )
    parser.add_argument(
        "--cjk-budget-scale",
        type=float,
        default=1.0,
        help="Optional internal drafting budget scale for CJK prompts (default: 1.0; e.g., 0.70 to compensate CJK char/token ratio).",
    )
    parser.add_argument("--limit", type=positive_int, default=None, help="Optional cap on number of prompts to process.")
    parser.add_argument("--export-all", action="store_true", help="Export even samples that failed the 6-gate quality filter (for debugging).")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    return parser


def run(args: argparse.Namespace) -> None:
    prompts_path = Path(args.prompts_jsonl)
    out_dir = Path(args.out_dir)
    gen_dir = out_dir / "generations"
    ckpt_dir = out_dir / "checkpoints"
    err_dir = out_dir / "errors"
    gen_dir.mkdir(parents=True, exist_ok=True)

    records = [
        json.loads(line)
        for line in prompts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        records = records[: args.limit]

    cfg = FlowConfig(
        batch_size=args.batch_size,
        plan_max_output_tokens=args.plan_max_output_tokens,
        plan_retry_cap_tokens=max(8_192, args.plan_max_output_tokens * 2),
        generation_max_output_tokens=args.generation_max_output_tokens,
        review_max_output_tokens=args.review_max_output_tokens,
        repair_max_output_tokens=args.repair_max_output_tokens,
        max_singleton_retries=args.max_singleton_retries,
        length_retry_ratio=args.length_retry_ratio,
        draft_temperature=args.draft_temperature,
        use_contracts=True,
        use_review_repair=True,
    )

    def _run_one(rec: dict[str, Any]) -> str:
        sid = rec["spec_id"]
        out_path = gen_dir / f"{sid}.json"
        if out_path.exists():
            return "skipped"
        client = OpenAICompatibleClient.from_env(args.env_prefix)
        try:
            sample = synthesize_posttrain_sample(
                rec,
                client,
                config=cfg,
                checkpoint_path=ckpt_dir / f"{sid}.json",
                cjk_budget_scale=args.cjk_budget_scale,
            )
            write_json_atomic(out_path, sample)
            return "pass" if sample["quality"]["passed"] else "filtered"
        except Exception as err:
            if not args.continue_on_error:
                raise
            write_json_atomic(
                err_dir / f"{sid}.json",
                {
                    "spec_id": sid,
                    "error_type": type(err).__name__,
                    "error_message": str(err),
                    "traceback": traceback.format_exc(),
                },
            )
            return "failed"

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futs = {executor.submit(_run_one, rec): rec["spec_id"] for rec in records}
        for fut in as_completed(futs):
            sid = futs[fut]
            completed += 1
            status = fut.result()
            print(f"[{completed}/{len(records)}] {status}: {sid}", flush=True)

    samples: list[dict[str, Any]] = []
    for rec in records:
        p = gen_dir / f"{rec['spec_id']}.json"
        if p.exists():
            samples.append(json.loads(p.read_text(encoding="utf-8")))

    summary = export_sft_datasets(
        samples,
        out_dir / "sft_exports",
        require_quality_pass=not args.export_all,
    )
    review_md = export_canary_review_markdown(samples, out_dir / "canary_review.md")
    summary["canary_review_md"] = str(review_md)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.env_file:
        load_env(Path(args.env_file))
    run(args)


if __name__ == "__main__":
    main()
