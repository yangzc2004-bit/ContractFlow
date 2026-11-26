from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learnbyai_research.llm import OpenAICompatibleClient
from learnbyai_research.posttrain_data import (
    DOMAIN_BY_ID,
    PromptSpec,
    check_prompt_leakage,
    deduplicate_prompts,
    export_prompts_preview_markdown,
    generate_topic_seeds,
    load_leakage_references,
    prompt_sha256,
    render_natural_prompt,
    sample_prompt_specs,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate zero-leakage multi-domain post-training prompts via Responses API."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--canary",
        action="store_true",
        help="Generate 24 canary prompts (2 per domain across all 12 domains) for manual inspection.",
    )
    mode.add_argument(
        "--count",
        type=positive_int,
        help="Total number of synthetic prompts to generate (e.g., 1200 or 4800).",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for seeds, prompts.jsonl, and preview.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--en-ratio", type=float, default=0.70)
    parser.add_argument("--workers", type=positive_int, default=6)
    parser.add_argument("--env-prefix", default="PROMPT_GEN", help="Env prefix for prompt generator LLM (default: PROMPT_GEN).")
    parser.add_argument("--model", default=None, help="Override model name (e.g., deepseek-v4.1-flash).")
    parser.add_argument("--base-url", default=None, help="Override API base URL.")
    parser.add_argument("--api-key", default=None, help="Override API key (prefer env vars over CLI flags).")
    parser.add_argument(
        "--wire-api",
        choices=["responses", "chat"],
        default="responses",
        help="API wire protocol (default: responses).",
    )
    parser.add_argument("--timeout", type=positive_int, default=120)
    parser.add_argument(
        "--leakage-ref",
        nargs="*",
        default=[],
        help="Optional test-set JSON/JSONL paths (LongGenBench / LongBench-Write) for automatic n-gram leakage check.",
    )
    parser.add_argument(
        "--specs-only",
        action="store_true",
        help="Only sample and write specs.jsonl without calling any LLM.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    return parser


def make_prompt_client(args: argparse.Namespace) -> OpenAICompatibleClient:
    prefix = args.env_prefix
    api_key = args.api_key or os.environ.get(f"{prefix}_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = args.base_url or os.environ.get(f"{prefix}_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    model = args.model or os.environ.get(f"{prefix}_MODEL") or os.environ.get("OPENAI_MODEL") or "deepseek-v4.1-flash"
    if not api_key:
        raise RuntimeError(f"API key required via {prefix}_API_KEY, OPENAI_API_KEY, or --api-key.")
    if not base_url:
        raise RuntimeError(f"Base URL required via {prefix}_BASE_URL, OPENAI_BASE_URL, or --base-url.")
    return OpenAICompatibleClient(
        model=model,
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        wire_api=args.wire_api,
        timeout=args.timeout,
    )


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def ensure_topic_seeds(
    args: argparse.Namespace,
    specs: list[PromptSpec],
    seeds_dir: Path,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    needed: dict[tuple[str, str], int] = Counter((s.domain_id, s.language) for s in specs)
    seeds_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}

    tasks: list[tuple[str, str, int, Path]] = []
    for (domain_id, lang), req_count in sorted(needed.items()):
        # Cap distinct seeds per (domain, lang) at 80 for large runs, or exact count for smaller runs
        target_seeds = min(80, max(req_count, 2))
        cache_path = seeds_dir / f"{domain_id}_{lang}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, list) and len(cached) >= target_seeds:
                seeds_by_key[(domain_id, lang)] = cached
                continue
        tasks.append((domain_id, lang, target_seeds, cache_path))

    def _gen(domain_id: str, lang: str, target: int, cache_path: Path) -> tuple[tuple[str, str], list[dict[str, Any]]]:
        client = make_prompt_client(args)
        seeds = generate_topic_seeds(
            client,
            domain=DOMAIN_BY_ID[domain_id],
            language=lang,  # type: ignore[arg-type]
            count=target,
        )
        write_json_atomic(cache_path, seeds)
        return (domain_id, lang), seeds

    if tasks:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
            futs = {pool.submit(_gen, d, l, t, p): (d, l) for d, l, t, p in tasks}
            for fut in as_completed(futs):
                d, l = futs[fut]
                key, seeds = fut.result()
                seeds_by_key[key] = seeds
                print(f"[seeds] {d} ({l}): {len(seeds)} seeds ready", flush=True)
    return seeds_by_key


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = sample_prompt_specs(
        count=args.count or 24,
        canary=bool(args.canary),
        seed=args.seed,
        en_ratio=args.en_ratio,
    )
    specs_path = out_dir / "specs.jsonl"
    with specs_path.open("w", encoding="utf-8") as fh:
        for spec in specs:
            fh.write(json.dumps(spec.to_dict(), ensure_ascii=False) + "\n")
    print(f"Sampled {len(specs)} specs -> {specs_path}", flush=True)
    if args.specs_only:
        return

    seeds_by_key = ensure_topic_seeds(args, specs, out_dir / "seeds")
    ref_shingles = load_leakage_references(args.leakage_ref)
    if args.leakage_ref:
        print(f"Loaded {len(ref_shingles)} reference test prompts for zero-leakage check.", flush=True)

    # Assign a distinct seed item round-robin per (domain_id, language)
    key_cursors: dict[tuple[str, str], int] = defaultdict(int)
    assignments: list[tuple[PromptSpec, dict[str, Any]]] = []
    for spec in specs:
        key = (spec.domain_id, spec.language)
        pool_seeds = seeds_by_key.get(key, [])
        if not pool_seeds:
            raise RuntimeError(f"No topic seeds generated for {key}")
        idx = key_cursors[key]
        key_cursors[key] += 1
        assignments.append((spec, pool_seeds[idx % len(pool_seeds)]))

    items_dir = out_dir / "items"
    errors_dir = out_dir / "errors"
    items_dir.mkdir(parents=True, exist_ok=True)

    def _render_one(spec: PromptSpec, seed_item: dict[str, Any]) -> str:
        item_path = items_dir / f"{spec.spec_id}.json"
        if item_path.exists():
            return "skipped"
        try:
            client = make_prompt_client(args)
            prompt_text, fidelity_errors = render_natural_prompt(client, spec, seed_item)
            clean_ok, leakage_reason = check_prompt_leakage(prompt_text, ref_shingles)
            record = {
                "spec_id": spec.spec_id,
                "prompt": prompt_text,
                "prompt_sha256": prompt_sha256(prompt_text),
                "spec": spec.to_dict(),
                "seed_item": seed_item,
                "fidelity_errors": fidelity_errors,
                "leakage_clean": clean_ok,
                "leakage_reason": leakage_reason,
                "generator_model": client.model,
                "generator_wire_api": client.wire_api,
            }
            write_json_atomic(item_path, record)
            return "ok" if (clean_ok and not fidelity_errors) else "warn"
        except Exception as err:
            if not args.continue_on_error:
                raise
            write_json_atomic(
                errors_dir / f"{spec.spec_id}.json",
                {
                    "spec_id": spec.spec_id,
                    "error_type": type(err).__name__,
                    "error_message": str(err),
                    "traceback": traceback.format_exc(),
                },
            )
            return "failed"

    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_render_one, spec, seed_item): spec.spec_id
            for spec, seed_item in assignments
        }
        for fut in as_completed(futures):
            sid = futures[fut]
            completed += 1
            status = fut.result()
            if status == "failed":
                failed += 1
            print(f"[{completed}/{len(assignments)}] {status}: {sid}", flush=True)

    # Collect in deterministic spec order, filter leakage, deduplicate, and export preview
    all_records: list[dict[str, Any]] = []
    leaked_records: list[dict[str, Any]] = []
    for spec in specs:
        item_path = items_dir / f"{spec.spec_id}.json"
        if not item_path.exists():
            continue
        rec = json.loads(item_path.read_text(encoding="utf-8"))
        if rec.get("leakage_clean", True):
            all_records.append(rec)
        else:
            leaked_records.append(rec)

    kept, dupes = deduplicate_prompts(all_records)
    prompts_jsonl = out_dir / "prompts.jsonl"
    with prompts_jsonl.open("w", encoding="utf-8") as fh:
        for rec in kept:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    preview_md = export_prompts_preview_markdown(kept, out_dir / "prompts_preview.md")
    summary = {
        "requested": len(specs),
        "generated": len(all_records) + len(leaked_records),
        "failed": failed,
        "dropped_leakage": len(leaked_records),
        "dropped_duplicate": len(dupes),
        "kept": len(kept),
        "fidelity_warnings": sum(1 for r in kept if r.get("fidelity_errors")),
        "by_bucket": dict(Counter(r["spec"]["bucket"] for r in kept)),
        "by_language": dict(Counter(r["spec"]["language"] for r in kept)),
        "by_length_tier": dict(Counter(r["spec"]["length_tier"] for r in kept)),
        "by_domain": dict(Counter(r["spec"]["domain_id"] for r in kept)),
        "prompts_jsonl": str(prompts_jsonl),
        "prompts_preview_md": str(preview_md),
    }
    write_json_atomic(out_dir / "prompt_generation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.env_file:
        load_env(Path(args.env_file))
    run(args)


if __name__ == "__main__":
    main()
