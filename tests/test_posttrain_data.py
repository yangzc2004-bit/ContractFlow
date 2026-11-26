from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

from learnbyai_research.contractflow_v2 import FlowConfig
from learnbyai_research.llm import extract_responses_text
from learnbyai_research.posttrain_data import (
    DOMAINS,
    DOMAIN_BY_ID,
    check_prompt_leakage,
    deduplicate_prompts,
    export_canary_review_markdown,
    export_prompts_preview_markdown,
    export_sft_datasets,
    generate_topic_seeds,
    load_leakage_references,
    render_natural_prompt,
    sample_prompt_specs,
    synthesize_posttrain_sample,
    verify_prompt_fidelity,
)


def test_domains_registry_has_12_clean_domains_including_legal_and_codebase() -> None:
    assert len(DOMAINS) == 12
    assert sum(1 for d in DOMAINS if d.bucket == "open_A") == 6
    assert sum(1 for d in DOMAINS if d.bucket == "structured_B") == 6
    assert "A6_legal_contract" in DOMAIN_BY_ID
    assert "B1_codebase_design" in DOMAIN_BY_ID
    assert DOMAIN_BY_ID["B4_expedition_log"].supports_date_offset is True


def test_extract_responses_text_skips_reasoning_items() -> None:
    payload = {
        "output": [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "internal scratchpad"}],
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Clean final prompt."}],
            },
        ]
    }
    assert extract_responses_text(payload) == "Clean final prompt."


def test_sample_prompt_specs_canary_and_1200_stratification() -> None:
    canary = sample_prompt_specs(canary=True, seed=42)
    assert len(canary) == 24
    by_domain = Counter(s.domain_id for s in canary)
    assert all(by_domain[d.domain_id] == 2 for d in DOMAINS)
    assert Counter(s.language for s in canary) == {"en": 12, "zh": 12}
    assert any(s.bucket == "open_a1" for s in canary)
    assert any(s.bucket == "open_a2" for s in canary)
    assert any(s.bucket == "structured_b" for s in canary)

    pilot = sample_prompt_specs(count=1200, canary=False, seed=42, en_ratio=0.70)
    assert len(pilot) == 1200
    assert Counter(s.length_tier for s in pilot) == {
        "4K": 180,
        "8K": 300,
        "16K": 360,
        "32K": 360,
    }
    assert Counter(s.language for s in pilot) == {"en": 840, "zh": 360}
    buckets = Counter(s.bucket for s in pilot)
    assert buckets["open_a1"] == 300
    assert buckets["open_a2"] == 300
    assert buckets["structured_b"] == 600

    # Verify every sampled rule slot is strictly within 1..unit_count and free of benchmark delimiters
    for spec in pilot:
        assert "#*#" not in spec.output_format_instruction
        for slot in spec.rule_slots:
            if slot["scope"] == "single":
                assert 1 <= slot["index"] <= spec.unit_count
            elif slot["scope"] == "range":
                assert 1 <= slot["start"] < slot["end"] <= spec.unit_count
            elif slot["scope"] == "every":
                assert 1 <= slot["start"] <= spec.unit_count
                assert slot["step"] >= 2
                if slot.get("end") is not None:
                    assert slot["start"] < slot["end"] <= spec.unit_count


def test_seed_generation_prompt_polishing_fidelity_and_leakage_gate(tmp_path) -> None:
    canary = sample_prompt_specs(canary=True, seed=7)
    b1_spec = next(s for s in canary if s.domain_id == "B1_codebase_design" and s.language == "en")

    class MockPromptGenLLM:
        model = "deepseek-v4.1-flash"
        wire_api = "responses"

        def __init__(self) -> None:
            self.render_calls = 0

        def complete(self, prompt: str, **kwargs) -> str:
            if '"seeds"' in prompt:
                return json.dumps(
                    {
                        "seeds": [
                            {
                                "title": "Distributed Consensus Storage Engine",
                                "setting": "Core storage infrastructure for a multi-region cloud database.",
                                "concepts": ["Raft log", "MVCC snapshot", "WAL compaction", "lease read"],
                                "events": ["chaos failover drill", "schema migration lock", "checksum scrub"],
                            },
                            {
                                "title": "Zero-Trust API Gateway Mesh",
                                "setting": "Enterprise service mesh handling edge TLS termination.",
                                "concepts": ["mTLS rotation", "rate limiter", "circuit breaker", "trace span"],
                                "events": ["key revocation drill", "canary rollout", "ddos throttling"],
                            },
                        ]
                    }
                )
            self.render_calls += 1
            if self.render_calls == 1:
                # First attempt intentionally drops numbers so fidelity retry is verified
                return "Write a modular software architecture document with detailed engineering notes and runbooks."
            parts = [
                f"Please author a complete {b1_spec.unit_count}-module design specification for our Distributed Consensus Storage Engine.",
                f"Target roughly {b1_spec.target_length} words overall ({b1_spec.min_unit_words}-{b1_spec.max_unit_words} words per module).",
                b1_spec.output_format_instruction,
            ]
            for slot in b1_spec.rule_slots:
                if slot["scope"] == "single":
                    parts.append(f"Module {slot['index']} must include a chaos failover drill.")
                elif slot["scope"] == "range":
                    parts.append(f"Modules {slot['start']} through {slot['end']} form the MVCC snapshot pipeline.")
                elif slot["scope"] == "every":
                    end_txt = f" through Module {slot['end']}" if slot.get("end") else ""
                    parts.append(
                        f"Starting at Module {slot['start']}{end_txt}, every {slot['step']} modules must include a checksum scrub runbook."
                    )
            return " ".join(parts)

        def completion_metadata(self) -> dict:
            return {"status": "completed"}

        def completion_history(self) -> list[dict]:
            return []

    llm = MockPromptGenLLM()
    seeds = generate_topic_seeds(llm, domain=DOMAIN_BY_ID["B1_codebase_design"], language="en", count=2)
    assert len(seeds) == 2

    prompt_text, errs = render_natural_prompt(llm, b1_spec, seeds[0])
    assert llm.render_calls == 2
    assert errs == []
    assert verify_prompt_fidelity(b1_spec, prompt_text) == []

    # Leakage reference check catches test-set overlap and forbidden markers
    ref_file = tmp_path / "ref_test.json"
    ref_file.write_text(
        json.dumps([{"prompt": "Design a 50-floor luxury skyscraper with a rooftop garden and gym."}]),
        encoding="utf-8",
    )
    refs = load_leakage_references([ref_file])
    assert check_prompt_leakage(prompt_text, refs)[0] is True
    assert check_prompt_leakage("Design a 50-floor luxury skyscraper with a rooftop garden and gym.", refs)[0] is False
    assert check_prompt_leakage("Write 10 sections separated by #*# markers.", refs)[0] is False

    # Intra-set deduplication drops near-duplicate prompts
    kept, dupes = deduplicate_prompts(
        [
            {"spec_id": b1_spec.spec_id, "prompt": prompt_text, "spec": b1_spec.to_dict(), "seed_item": seeds[0]},
            {"spec_id": "dup_2", "prompt": prompt_text + " Extra word.", "spec": b1_spec.to_dict(), "seed_item": seeds[0]},
        ]
    )
    assert len(kept) == 1 and len(dupes) == 1
    preview = export_prompts_preview_markdown(kept, tmp_path / "preview.md")
    assert preview.exists() and b1_spec.spec_id in preview.read_text(encoding="utf-8")


def test_posttrain_synthesis_quality_filter_and_dual_sft_export(tmp_path) -> None:
    class MockSelfBootstrapLLM:
        model = "qwen3.5-9b-mock"

        def complete(self, prompt: str, **kwargs) -> str:
            if "CONTRACTFLOW_SPARSE_CONTRACT_JSON" in prompt:
                return json.dumps(
                    {
                        "global_brief": "Authoritative legal compliance agreement with clear definitions.",
                        "total_units": 3,
                        "unit_label": "Article",
                        "unit_titles": {
                            "1": "Article 1: Definitions",
                            "2": "Article 2: Breach Notification",
                            "3": "Article 3: Enforcement",
                        },
                        "global_invariants": ["Use unambiguous statutory language."],
                        "rules": [
                            {"scope": "single", "index": 2, "obligation": "Specify 72-hour breach notice"}
                        ],
                    }
                )
            if "CONTRACTFLOW_DRAFT_UNITS" in prompt:
                match = re.search(r"Write ONLY the requested units: (\[[^\]]+\])", prompt)
                indices = json.loads(match.group(1))
                blocks = []
                for idx in indices:
                    words = [f"clause{idx}w{k}" for k in range(28)]
                    extra = " The processor shall provide 72-hour breach notice to the controller." if idx == 2 else ""
                    blocks.append(f"<<<UNIT {idx}>>>\n{' '.join(words)}.{extra}\n<<<END UNIT {idx}>>>")
                return "\n".join(blocks)
            if "CONTRACTFLOW_REVIEW_UNITS_JSON" in prompt:
                raw_json = prompt.split("UNITS_JSON:\n", 1)[1].strip()
                items = json.loads(raw_json)
                return json.dumps(
                    {
                        "results": [
                            {"index": int(item["index"]), "passed": True, "missing_obligations": [], "issues": []}
                            for item in items
                        ]
                    }
                )
            return "{}"

        def completion_metadata(self) -> dict:
            return {"finish_reason": "stop"}

        def completion_history(self) -> list[dict]:
            return []

    prompt_record = {
        "spec_id": "canary_011_A6_legal_contract_en_4k",
        "prompt": "Draft a concise legal data governance agreement covering definitions, 72-hour breach notice, and enforcement.",
        "spec": {
            "spec_id": "canary_011_A6_legal_contract_en_4k",
            "bucket": "open_a1",
            "domain_id": "A6_legal_contract",
            "language": "en",
            "length_tier": "4K",
            "target_length": 95,
            "unit_count": 3,
            "unit_label": "Article",
            "min_unit_words": 20,
            "max_unit_words": 40,
            "output_format_id": "markdown_open",
        },
        "seed_item": {"title": "Cross-Border Data Governance Act"},
    }

    sample = synthesize_posttrain_sample(
        prompt_record,
        MockSelfBootstrapLLM(),
        config=FlowConfig(batch_size=3, use_contracts=True, use_review_repair=True),
        checkpoint_path=tmp_path / "ckpt.json",
    )
    assert sample["quality"]["passed"] is True
    assert sample["quality"]["completion_rate"] == 100.0
    assert "## Article 1: Definitions" in sample["final_text"]

    summary = export_sft_datasets([sample], tmp_path / "sft")
    assert summary["kept_samples"] == 1
    direct_lines = (tmp_path / "sft" / "direct_sft.jsonl").read_text(encoding="utf-8").splitlines()
    prefix_lines = (tmp_path / "sft" / "contract_prefix_sft.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(direct_lines) == 1 and len(prefix_lines) == 1

    direct_obj = json.loads(direct_lines[0])
    prefix_obj = json.loads(prefix_lines[0])
    assert direct_obj["messages"][0]["role"] == "user"
    assert "<contract>" not in direct_obj["messages"][1]["content"]
    prefix_ans = prefix_obj["messages"][1]["content"]
    assert prefix_ans.startswith("<contract>\n")
    assert "</contract>\n\n## Article 1: Definitions" in prefix_ans
    # Ensure compact sparse contract does NOT dump expanded units array
    contract_json = json.loads(prefix_ans.split("<contract>\n", 1)[1].split("\n</contract>", 1)[0])
    assert "rules" in contract_json and "units" not in contract_json

    review_md = export_canary_review_markdown([sample], tmp_path / "canary_review.md")
    assert review_md.exists() and "PASS" in review_md.read_text(encoding="utf-8")


def test_posttrain_cli_scripts_parse_arguments() -> None:
    root = Path(__file__).resolve().parents[1]

    gen_spec = importlib.util.spec_from_file_location("gen_pt", root / "scripts" / "generate_posttrain_prompts.py")
    assert gen_spec is not None and gen_spec.loader is not None
    gen_mod = importlib.util.module_from_spec(gen_spec)
    gen_spec.loader.exec_module(gen_mod)
    g_args = gen_mod.build_parser().parse_args(
        ["--canary", "--out-dir", "runs/canary", "--model", "deepseek-v4.1-flash", "--wire-api", "responses"]
    )
    assert g_args.canary is True and g_args.wire_api == "responses" and g_args.model == "deepseek-v4.1-flash"

    syn_spec = importlib.util.spec_from_file_location("syn_pt", root / "scripts" / "run_posttrain_synthesis.py")
    assert syn_spec is not None and syn_spec.loader is not None
    syn_mod = importlib.util.module_from_spec(syn_spec)
    syn_spec.loader.exec_module(syn_mod)
    s_args = syn_mod.build_parser().parse_args(
        ["--prompts-jsonl", "runs/canary/prompts.jsonl", "--out-dir", "runs/canary_sft", "--workers", "4"]
    )
    assert s_args.workers == 4 and s_args.length_retry_ratio == 0.7
