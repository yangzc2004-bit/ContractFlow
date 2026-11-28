from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

from learnbyai_research.contractflow_v2 import DocumentContract, FlowConfig, UnitContract
from learnbyai_research.llm import extract_responses_text
from learnbyai_research.posttrain_data import (
    DOMAINS,
    DOMAIN_BY_ID,
    check_prompt_leakage,
    deduplicate_prompts,
    evaluate_sample_quality,
    export_canary_review_markdown,
    export_prompts_preview_markdown,
    export_sft_datasets,
    generate_topic_seeds,
    load_leakage_references,
    render_formatted_document,
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
    slot_leak_errs = verify_prompt_fidelity(b1_spec, prompt_text + " Make sure to fulfill G1 and S1.")
    assert any("leaked internal slot IDs" in e for e in slot_leak_errs)

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
    assert s_args.draft_temperature == 0.65 and s_args.cjk_budget_scale == 1.0

    s_custom = syn_mod.build_parser().parse_args(
        [
            "--prompts-jsonl",
            "runs/canary/prompts.jsonl",
            "--out-dir",
            "runs/canary_sft",
            "--draft-temperature",
            "0.5",
            "--cjk-budget-scale",
            "0.7",
        ]
    )
    assert s_custom.draft_temperature == 0.5 and s_custom.cjk_budget_scale == 0.7


def test_render_formatted_document_strips_duplicate_headings_and_separators() -> None:
    contract = DocumentContract(
        global_brief="三集连续剧大纲。",
        global_invariants=[],
        unit_label="集",
        total_units=3,
        rules=[],
        units=[
            UnitContract(index=1, title="第 1 集", min_words=50, max_words=100),
            UnitContract(index=2, title="第 2 集", min_words=50, max_words=100),
            UnitContract(index=3, title="第 3 集", min_words=50, max_words=100),
        ],
    )
    rendered_hr = render_formatted_document(
        contract,
        {
            1: "---\n**第 1 集**\n\n---\n暴雨夜里的信号塔突然失联。\n---",
            2: "**第 2 集：** 紧接上一集的线索展开调查。\n===",
            3: "第 3 集讲述了团队在黎明前完成切换的过程。",
        },
        output_format_id="hr_dash",
        language="zh",
    )
    assert rendered_hr.count("**第 1 集**") == 1
    assert rendered_hr.count("**第 2 集**") == 1
    assert rendered_hr.count("**第 3 集**") == 1
    assert "\n\n---\n\n---" not in rendered_hr
    assert "暴雨夜里的信号塔突然失联。" in rendered_hr
    # Inline bold prefix must strip ONLY the prefix, preserving prose on the same line
    assert "紧接上一集的线索展开调查。" in rendered_hr
    # Natural prose starting with "第 3 集讲述了..." (no colon/heading syntax) must be preserved
    assert "第 3 集讲述了团队在黎明前完成切换的过程。" in rendered_hr

    rendered_h2 = render_formatted_document(
        contract,
        {
            1: "## 第 1 集：序幕\n第 1 集：这是正式正文内容。",
        },
        output_format_id="markdown_h2",
        language="zh",
    )
    assert rendered_h2 == "## 第 1 集：\n\n这是正式正文内容。"

    # markdown_open: generic "章 1" -> idiomatic "## 第 1 章", and promote leading "## 第一章：..." without stacking
    open_contract = DocumentContract(
        global_brief="太空打捞科幻小说。",
        global_invariants=[],
        unit_label="章",
        total_units=2,
        rules=[],
        units=[
            UnitContract(index=1, title="章 1", min_words=50, max_words=100),
            UnitContract(index=2, title="章 2", min_words=50, max_words=100),
        ],
    )
    rendered_open = render_formatted_document(
        open_contract,
        {
            1: "## 第一章：木星边缘的静默契约\n\n晨昏线号在木星轨道附近失联。",
            2: "调查团队启程前往目标坐标。",
        },
        output_format_id="markdown_open",
        language="zh",
    )
    assert "## 章 1" not in rendered_open
    assert "## 章 2" not in rendered_open
    assert rendered_open.startswith("## 第一章：木星边缘的静默契约\n\n晨昏线号在木星轨道附近失联。")
    assert "## 第 2 章\n\n调查团队启程前往目标坐标。" in rendered_open

    # markdown_open ZH with English fallback unit_label="Chapter" + plain Chinese heading line (like canary_012)
    en_label_zh_contract = DocumentContract(
        global_brief="数据跨境合规协议草案。",
        global_invariants=[],
        unit_label="Chapter",
        total_units=2,
        rules=[],
        units=[
            UnitContract(index=1, title="Chapter 1", min_words=50, max_words=100),
            UnitContract(index=2, title="Chapter 2", min_words=50, max_words=100),
        ],
    )
    rendered_zh_localized = render_formatted_document(
        en_label_zh_contract,
        {
            1: "第一章 数据资产全景图：分级矩阵\n\n第一条【目的与定义】本章节确立治理基石。",
            2: "第二章从跨境传输场景出发展开条款推演。\n\n具体流程如下。",
        },
        output_format_id="markdown_open",
        language="zh",
    )
    assert "## Chapter 1" not in rendered_zh_localized
    assert "## Chapter 2" not in rendered_zh_localized
    assert rendered_zh_localized.startswith("## 第一章 数据资产全景图：分级矩阵\n\n第一条【目的与定义】本章节确立治理基石。")
    # Line ending with "。" is narrative prose, not a title -> gets localized "## 第 2 章"
    assert "## 第 2 章\n\n第二章从跨境传输场景出发展开条款推演。" in rendered_zh_localized


def test_open_a1_calibrates_unit_word_budgets_to_target_length(tmp_path) -> None:
    class MockOpenA1LLM:
        model = "qwen3.5-9b-mock"

        def complete(self, prompt: str, **kwargs) -> str:
            if "CONTRACTFLOW_SPARSE_CONTRACT_JSON" in prompt:
                # Stage-1 chooses 6 chapters for a 12,000-word target (instead of spec's initial 18)
                return json.dumps(
                    {
                        "global_brief": "Comprehensive six-chapter policy study.",
                        "total_units": 6,
                        "unit_label": "Chapter",
                        "unit_titles": {str(i): f"Chapter {i}: Section {i}" for i in range(1, 7)},
                        "global_invariants": ["Maintain formal analytical tone."],
                        "rules": [],
                    }
                )
            if "CONTRACTFLOW_DRAFT_UNITS" in prompt:
                match = re.search(r"Write ONLY the requested units: (\[[^\]]+\])", prompt)
                indices = json.loads(match.group(1))
                return "\n".join(
                    f"<<<UNIT {idx}>>>\n"
                    + " ".join(f"ch{idx}w{k}" for k in range(2000))
                    + f"\n<<<END UNIT {idx}>>>"
                    for idx in indices
                )
            if "CONTRACTFLOW_DRAFT_SINGLE_UNIT" in prompt:
                match = re.search(r"Write ONLY unit (\d+)", prompt)
                idx = int(match.group(1)) if match else 1
                return " ".join(f"ch{idx}w{k}" for k in range(2000))
            return "{}"

        def completion_metadata(self) -> dict:
            return {"finish_reason": "stop"}

        def completion_history(self) -> list[dict]:
            return []

    prompt_record = {
        "spec_id": "test_open_a1_12k",
        "prompt": "Write a comprehensive 12,000-word policy study on maritime port resilience.",
        "spec": {
            "spec_id": "test_open_a1_12k",
            "bucket": "open_a1",
            "domain_id": "A2_policy_report",
            "language": "en",
            "length_tier": "16K",
            "target_length": 12000,
            "unit_count": 18,
            "unit_label": "Chapter",
            "min_unit_words": 450,
            "max_unit_words": 650,
            "output_format_id": "markdown_open",
        },
        "seed_item": {"title": "Port Resilience Study"},
    }
    sample = synthesize_posttrain_sample(
        prompt_record,
        MockOpenA1LLM(),
        config=FlowConfig(batch_size=3, use_contracts=True, use_review_repair=False),
        checkpoint_path=tmp_path / "open_a1_ckpt.json",
    )
    compiled_units = sample["contract_v2"]["units"]
    assert len(compiled_units) == 6
    # 12000 / 6 = 2000 -> cal_min = 1760, cal_max = 2300
    assert compiled_units[0]["min_words"] == 1760
    assert compiled_units[0]["max_words"] == 2300
    assert sample["quality"]["passed"] is True


def test_evaluate_sample_quality_6_gates_language_and_hygiene() -> None:
    zh_record = {
        "spec": {
            "language": "zh",
            "target_length": 100,
        }
    }
    base_result = {
        "contract": {"total_units": 2, "units": [{"index": 1}, {"index": 2}]},
        "units": [{"index": 1, "text": "a"}, {"index": 2, "text": "b"}],
        "metadata": {"review_results": [], "repair_records": []},
    }

    # 1) ZH prompt with English output must FAIL Gate 5
    en_only = " ".join(f"englishword{i}" for i in range(100))
    q_zh_mismatch = evaluate_sample_quality(zh_record, {**base_result, "final_text": en_only})
    assert q_zh_mismatch["passed"] is False
    assert any(r.startswith("language_mismatch_zh") for r in q_zh_mismatch["reasons"])

    # 2) EN prompt with Chinese output must FAIL Gate 5
    en_record = {"spec": {"language": "en", "target_length": 100}}
    zh_clean = (
        "## 第一章：引言\n\n"
        "近年来分布式共识协议与容错存储架构在跨区域云基础设施中发挥着核心作用，"
        "本章节围绕状态机复制、日志压缩、租约读优化与网络分区恢复机制展开系统阐述，"
        "结合实际工程故障演练与可观测性指标给出清晰落地规范。"
    )
    q_en_mismatch = evaluate_sample_quality(en_record, {**base_result, "final_text": zh_clean})
    assert q_en_mismatch["passed"] is False
    assert any(r.startswith("language_mismatch_en") for r in q_en_mismatch["reasons"])

    # 3) Clean Chinese output passes all 6 gates
    q_zh_pass = evaluate_sample_quality(zh_record, {**base_result, "final_text": zh_clean})
    assert q_zh_pass["passed"] is True
    assert q_zh_pass["cjk_ratio"] >= 0.45
    assert q_zh_pass["format_hygiene_issues"] == []

    # 4) Duplicate consecutive headings, stacked ## headings, double --- separators, and leftover tags fail Gate 6
    dirty_text = (
        "**第 1 集**\n\n**第 1 集**\n\n"
        + "## 章 1\n\n## 第一章：引子\n\n"
        + zh_clean
        + "\n\n---\n\n---\n\n<<<UNIT 2>>>"
    )
    q_dirty = evaluate_sample_quality(zh_record, {**base_result, "final_text": dirty_text})
    assert q_dirty["passed"] is False
    assert "duplicate_consecutive_headings" in q_dirty["format_hygiene_issues"]
    assert "stacked_markdown_headings" in q_dirty["format_hygiene_issues"]
    assert "duplicate_hr_separators" in q_dirty["format_hygiene_issues"]
    assert "leftover_unit_tags" in q_dirty["format_hygiene_issues"]

    # 5) ZH prose with English-only compiled contract must FAIL Gate 5 contract language check
    en_contract = {
        "total_units": 2,
        "units": [{"index": 1}, {"index": 2}],
        "global_brief": "Episode breakdown for a serialized streaming drama.",
        "global_invariants": [
            "Each episode must be between 143 and 186 characters.",
            "The outline must support narrative rhythm planning and suspense hooks.",
        ],
        "rules": [{"scope": "single", "index": 1, "obligation": "Design a cold open using flashback."}],
    }
    q_bad_contract = evaluate_sample_quality(
        zh_record,
        {**base_result, "contract": en_contract, "final_text": zh_clean},
    )
    assert q_bad_contract["passed"] is False
    assert any(r.startswith("contract_language_mismatch_zh") for r in q_bad_contract["reasons"])

    zh_contract = {
        "total_units": 2,
        "units": [{"index": 1}, {"index": 2}],
        "global_brief": "面向分布式共识协议的中文系统设计指南。",
        "global_invariants": ["每章保持严谨工程表述，统一使用中文专业术语与操作清单。"],
        "rules": [{"scope": "single", "index": 1, "obligation": "给出状态机复制与日志压缩的核心流程。"}],
    }
    q_good_contract = evaluate_sample_quality(
        zh_record,
        {**base_result, "contract": zh_contract, "final_text": zh_clean},
    )
    assert q_good_contract["passed"] is True
    assert q_good_contract["contract_cjk_ratio"] is not None and q_good_contract["contract_cjk_ratio"] >= 0.25


def test_cjk_budget_scale_only_scales_zh_internal_drafting_budget(tmp_path) -> None:
    class MockBudgetLLM:
        model = "qwen3.5-9b-mock"

        def complete(self, prompt: str, **kwargs) -> str:
            if "CONTRACTFLOW_SPARSE_CONTRACT_JSON" in prompt:
                return json.dumps(
                    {
                        "global_brief": "测试文档大纲。",
                        "total_units": 4,
                        "unit_label": "章",
                        "unit_titles": {str(i): f"第 {i} 章" for i in range(1, 5)},
                        "global_invariants": [],
                        "rules": [],
                    }
                )
            if "CONTRACTFLOW_DRAFT_UNITS" in prompt:
                match = re.search(r"Write ONLY the requested units: (\[[^\]]+\])", prompt)
                indices = json.loads(match.group(1))
                return "\n".join(
                    f"<<<UNIT {idx}>>>\n" + ("中" * 500) + f"\n<<<END UNIT {idx}>>>"
                    for idx in indices
                )
            if "CONTRACTFLOW_DRAFT_SINGLE_UNIT" in prompt:
                return "中" * 500
            return "{}"

        def completion_metadata(self) -> dict:
            return {"finish_reason": "stop"}

        def completion_history(self) -> list[dict]:
            return []

    zh_open_record = {
        "spec_id": "test_zh_scale_open",
        "prompt": "请撰写一篇约 4000 字的中文技术分析报告。",
        "spec": {
            "spec_id": "test_zh_scale_open",
            "bucket": "open_a1",
            "domain_id": "A1_tech_handbook",
            "language": "zh",
            "length_tier": "4K",
            "target_length": 4000,
            "unit_count": 4,
            "unit_label": "章",
            "min_unit_words": 800,
            "max_unit_words": 1200,
            "output_format_id": "markdown_open",
        },
        "seed_item": {"title": "中文技术报告"},
    }
    zh_sample = synthesize_posttrain_sample(
        zh_open_record,
        MockBudgetLLM(),
        config=FlowConfig(batch_size=4, use_contracts=True, use_review_repair=False),
        checkpoint_path=tmp_path / "zh_scale_ckpt.json",
        cjk_budget_scale=0.7,
    )
    # 4000 * 0.7 / 4 = 700 -> cal_min = round(700*0.88)=616, cal_max = round(700*1.15)=805
    assert zh_sample["contract_v2"]["units"][0]["min_words"] == 616
    assert zh_sample["contract_v2"]["units"][0]["max_words"] == 805
    # Original target_length in spec & quality evaluation must remain unscaled (4000)
    assert zh_sample["spec"]["target_length"] == 4000
    assert zh_sample["quality"]["target_length"] == 4000

    en_record = {
        "spec_id": "test_en_unscaled",
        "prompt": "Write a 4000-word technical report.",
        "spec": {
            "spec_id": "test_en_unscaled",
            "bucket": "structured_b",
            "domain_id": "B1_codebase_design",
            "language": "en",
            "length_tier": "4K",
            "target_length": 4000,
            "unit_count": 4,
            "unit_label": "Section",
            "min_unit_words": 800,
            "max_unit_words": 1200,
            "output_format_id": "markdown_h2",
        },
        "seed_item": {"title": "English Tech Report"},
    }
    en_sample = synthesize_posttrain_sample(
        en_record,
        MockBudgetLLM(),
        config=FlowConfig(batch_size=4, use_contracts=True, use_review_repair=False),
        checkpoint_path=tmp_path / "en_scale_ckpt.json",
        cjk_budget_scale=0.7,
    )
    assert en_sample["contract_v2"]["units"][0]["min_words"] == 800
    assert en_sample["contract_v2"]["units"][0]["max_words"] == 1200

    try:
        synthesize_posttrain_sample(zh_open_record, MockBudgetLLM(), cjk_budget_scale=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Non-positive cjk_budget_scale should raise ValueError")

