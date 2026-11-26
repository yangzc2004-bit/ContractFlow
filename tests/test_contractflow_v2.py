from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

from learnbyai_research.contractflow_v2 import (
    ContractRule,
    DocumentContract,
    FlowConfig,
    UnitContract,
    build_singleton_draft_prompt,
    build_sparse_contract_prompt,
    build_unit_repair_prompt,
    build_window_draft_prompt,
    build_window_review_prompt,
    compile_sparse_contract,
    draft_token_budget,
    effective_batch_size,
    expand_contract_rules,
    parse_singleton_unit,
    parse_unit_batch,
    repair_token_budget,
    review_token_budget,
    run_contractflow,
    word_count,
)
from learnbyai_research.longgenbench import task_from_dataset_item
from learnbyai_research.longgenbench_official_eval import (
    completion_rate,
    official_output_blocks,
    parse_blocks,
)
from learnbyai_research.longgenbench_v2 import render_longgen_document, run_longgen_v2


def test_expand_contract_rules_supports_all_four_scopes_and_bounded_stride() -> None:
    rules = [
        ContractRule(scope="all", obligation="Do not invent citations."),
        ContractRule(scope="range", start=1, end=12, obligation="Keep tone consistent."),
        ContractRule(scope="every", start=1, step=1, obligation="End with a brief transition."),
        ContractRule(scope="single", index=1, obligation="Introduce core notation."),
        ContractRule(scope="range", start=4, end=6, obligation="Cover consensus algorithms."),
        ContractRule(scope="every", start=3, step=3, obligation="Include a hands-on lab."),
        ContractRule(scope="every", start=2, end=8, step=3, obligation="Bounded stride note."),
        ContractRule(scope="single", index=999, obligation="Out of bounds ignored."),
    ]
    invariants, obligations = expand_contract_rules(12, rules)
    assert invariants == [
        "Do not invent citations.",
        "Keep tone consistent.",
        "End with a brief transition.",
    ]
    assert obligations[1] == ["Introduce core notation."]
    assert obligations[2] == ["Bounded stride note."]
    assert obligations[3] == ["Include a hands-on lab."]
    assert obligations[4] == ["Cover consensus algorithms."]
    assert obligations[5] == ["Cover consensus algorithms.", "Bounded stride note."]
    assert obligations[6] == ["Cover consensus algorithms.", "Include a hands-on lab."]
    assert obligations[8] == ["Bounded stride note."]
    assert obligations[9] == ["Include a hands-on lab."]
    assert obligations[11] == []
    assert obligations[12] == ["Include a hands-on lab."]


def test_dynamic_token_budgets_and_adaptive_batch_size() -> None:
    # 32K short segments (112 words) fit 8 per batch with generous ~2x headroom (~2.6K)
    assert effective_batch_size(8, 112, 6144) == 8
    b32 = draft_token_budget(8, 112, 6144)
    assert 2200 <= b32 <= 3000

    # 16K medium segments (220 words) fit 8 per batch (~4.5K <= 6144)
    assert effective_batch_size(8, 220, 6144) == 8
    b16 = draft_token_budget(8, 220, 6144)
    assert 4000 <= b16 <= 5000

    # Long textbook chapters (1500 words) automatically shrink batch size to 1
    assert effective_batch_size(8, 1500, 6144) == 1
    b_book = draft_token_budget(1, 1500, 6144)
    assert 3200 <= b_book <= 4200

    # Singleton budget stays compact and bounded
    assert 512 <= draft_token_budget(1, 112, 6144) <= 900
    assert 1024 <= review_token_budget(4, 4096) <= 2048
    assert repair_token_budget(2, 220, 4096) <= 4096


def test_generic_three_level_parser_and_tag_free_singleton() -> None:
    units = [UnitContract(index=i, title=f"Floor {i}", min_words=5, max_words=50) for i in range(1, 6)]
    raw = (
        "<<<UNIT 1>>>\nFirst floor complete tag.\n<<<END UNIT 1>>>\n"
        "<<<UNIT 2>>>\nSecond floor forgot closing tag.\n"
        "### Floor 3:\nThird floor markdown heading.\n"
        "**Floor 4:** Fourth floor bold heading.\n"
        "#*# Floor 5 (Observation):\nFifth floor official marker.\n"
        "*** finished ***\n"
    )
    parsed = parse_unit_batch(units, raw, unit_label="Floor")
    assert parsed[1] == "First floor complete tag."
    assert parsed[2] == "Second floor forgot closing tag."
    assert parsed[3] == "Third floor markdown heading."
    assert parsed[4] == "Fourth floor bold heading."
    assert parsed[5] == "Fifth floor official marker."

    # A natural prose sentence starting with "Floor 1 ..." must NOT be swallowed as a heading
    prose = "Floor 1 features a grand sunlit lobby with granite columns and a cafe."
    assert parse_singleton_unit(prose, units[0], unit_label="Floor") == prose


def test_sparse_compiler_retries_on_truncation_without_echoing_raw() -> None:
    class TruncatingCompilerLLM:
        model = "trunc-compiler"

        def __init__(self) -> None:
            self.calls: list[tuple[str, int | None]] = []
            self._meta: dict = {}

        def complete(self, prompt: str, **kwargs) -> str:
            self.calls.append((prompt, kwargs.get("max_output_tokens")))
            if len(self.calls) == 1:
                self._meta = {"finish_reason": "length"}
                return '{"global_brief": "cut off'
            self._meta = {"finish_reason": "stop"}
            return json.dumps(
                {
                    "global_brief": "Write a 4-chapter systems textbook.",
                    "total_units": 4,
                    "unit_label": "Chapter",
                    "unit_titles": {"1": "Intro", "2": "Consensus"},
                    "global_invariants": ["Keep notation consistent."],
                    "rules": [
                        {"scope": "single", "index": 2, "obligation": "Explain Raft"},
                        {"scope": "every", "start": 2, "step": 2, "obligation": "Add lab"},
                    ],
                }
            )

        def completion_metadata(self) -> dict:
            return dict(self._meta)

        def completion_history(self) -> list[dict]:
            return []

    llm = TruncatingCompilerLLM()
    contract = compile_sparse_contract(
        "Write a 4-chapter textbook.",
        llm,
        max_output_tokens=2048,
        retry_cap_tokens=8192,
    )
    assert len(llm.calls) == 2
    assert llm.calls[1][1] == 4096
    assert "MALFORMED_RESPONSE" not in llm.calls[1][0]
    assert contract.total_units == 4
    assert contract.units[0].title == "Intro"
    assert contract.units[1].obligations == ["Explain Raft", "Add lab"]
    assert contract.units[3].obligations == ["Add lab"]


class MockV2FlowLLM:
    model = "v2-mock"

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.reviewed_indices: list[int] = []
        self.dropped_once = False
        self.failed_review_once = False
        self._meta: dict = {"finish_reason": "stop"}
        self._history: list[dict] = []

    def complete(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        self._meta = {
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
        }
        self._history.append(dict(self._meta))

        if "CONTRACTFLOW_SPARSE_CONTRACT_JSON" in prompt:
            return json.dumps(
                {
                    "global_brief": "Coherent structured book with concrete details.",
                    "total_units": 4,
                    "unit_label": "Chapter",
                    "unit_titles": {"1": "Chapter 1: Foundations", "2": "Chapter 2: Raft"},
                    "global_invariants": ["Avoid fabricated citations."],
                    "rules": [
                        {"scope": "single", "index": 2, "obligation": "explain leader election"},
                        {"scope": "single", "index": 4, "obligation": "include capstone project"},
                    ],
                }
            )

        if "CONTRACTFLOW_DRAFT_UNITS" in prompt:
            match = re.search(r"Write ONLY the requested units: (\[[^\]]+\])", prompt)
            indices = json.loads(match.group(1))
            # Deliberately drop the last unit on the first batch to exercise divide-and-conquer + singleton recovery
            emit = indices[:-1] if (not self.dropped_once and len(indices) > 1) else indices
            if not self.dropped_once and len(indices) > 1:
                self.dropped_once = True
            return "\n".join(
                f"<<<UNIT {idx}>>>\n"
                + f"Chapter {idx} discusses systems concepts "
                + ("alpha " * 22)
                + f"\n<<<END UNIT {idx}>>>"
                for idx in emit
            )

        if "CONTRACTFLOW_DRAFT_SINGLE_UNIT" in prompt:
            # Pure tag-free prose
            return "Recovered singleton chapter prose covering essential material " + ("beta " * 22)

        if "CONTRACTFLOW_REVIEW_UNITS_JSON" in prompt:
            raw_json = prompt.split("UNITS_JSON:\n", 1)[1].strip()
            items = json.loads(raw_json)
            results = []
            for item in items:
                idx = int(item["index"])
                self.reviewed_indices.append(idx)
                if idx == 2 and not self.failed_review_once:
                    self.failed_review_once = True
                    results.append(
                        {
                            "index": idx,
                            "passed": False,
                            "missing_obligations": ["explain leader election"],
                            "issues": ["missing leader election explanation"],
                        }
                    )
                else:
                    results.append(
                        {"index": idx, "passed": True, "missing_obligations": [], "issues": []}
                    )
            return json.dumps({"results": results})

        if "CONTRACTFLOW_REPAIR_SINGLE_UNIT" in prompt or "CONTRACTFLOW_REPAIR_UNITS" in prompt:
            return (
                "Revised prose clearly explaining leader election and Raft consensus "
                + ("gamma " * 22)
            )

        return "{}"

    def completion_metadata(self) -> dict:
        return dict(self._meta)

    def completion_history(self) -> list[dict]:
        return [dict(item) for item in self._history]


def test_run_contractflow_v2_end_to_end_markdown_recovery_and_checkpoint(tmp_path) -> None:
    raw_request = "UNIQUE_RAW_PROMPT_SENTINEL: Write a 4-chapter textbook on distributed systems."
    llm = MockV2FlowLLM()
    ckpt = tmp_path / "v2_ckpt.json"
    result = run_contractflow(
        raw_request,
        llm,
        config=FlowConfig(batch_size=2, use_contracts=True, use_review_repair=True),
        total_units=4,
        unit_label="Chapter",
        min_words=20,
        max_words=40,
        checkpoint_path=ckpt,
    )
    assert result["system"] == "v2_contractflow"
    assert len(result["units"]) == 4
    assert result["metadata"]["final_structural_issues"] == {}
    # Context isolation: raw prompt sentinel must appear only in Stage-1 compilation, never in drafting/review
    draft_prompts = [p for p in llm.prompts if "CONTRACTFLOW_DRAFT" in p]
    assert draft_prompts and all("UNIQUE_RAW_PROMPT_SENTINEL" not in p for p in draft_prompts)
    # Tier-2 review must ONLY inspect constrained units (2 and 4), never unconstrained units (1 and 3)
    assert sorted(llm.reviewed_indices) == [2, 4]
    # Monotonic localized repair fixed unit 2 cleanly
    assert "leader election" in result["units"][1]["text"]
    assert "## Chapter 1: Foundations" in result["final_text"]
    assert ckpt.exists()

    # Resuming from completed checkpoint makes zero extra LLM calls
    llm_resume = MockV2FlowLLM()
    resumed = run_contractflow(
        raw_request,
        llm_resume,
        config=FlowConfig(batch_size=2, use_contracts=True, use_review_repair=True),
        total_units=4,
        unit_label="Chapter",
        min_words=20,
        max_words=40,
        checkpoint_path=ckpt,
    )
    assert resumed["metadata"]["resumed_from_checkpoint"] is True
    assert llm_resume.prompts == []
    assert resumed["final_text"] == result["final_text"]


def test_longgen_v2_adapter_official_format_and_2x2_ablations() -> None:
    item = {
        "prompt": (
            "Design a 4-floor tower. Each floor must be at least 20 words. "
            "Designate Floor 2 for gym use. Use '#*#' to separate floors."
        ),
        "type": "Floor",
        "number": 4,
        "prefix": "#*# Floor 1:",
        "checks_once": {"2": "gym"},
        "checks_range": {},
        "checks_periodic": {},
    }
    task = task_from_dataset_item(0, item)

    # Full v2_contractflow renders official #*# blocks cleanly for official eval.py
    llm_full = MockV2FlowLLM()
    out_full = run_longgen_v2(
        task,
        llm_full,
        config=FlowConfig(batch_size=2, use_contracts=True, use_review_repair=True),
    )
    assert out_full["system"] == "v2_contractflow"
    blocks = parse_blocks(official_output_blocks(out_full["final_text"]), "Floor")
    assert completion_rate(blocks, 4) == 100.0

    # v2_neither skips both contract compilation and review/repair, and still sees original prompt
    llm_neither = MockV2FlowLLM()
    out_neither = run_longgen_v2(
        task,
        llm_neither,
        config=FlowConfig(batch_size=2, use_contracts=False, use_review_repair=False),
    )
    assert out_neither["system"] == "v2_neither"
    assert not any("CONTRACTFLOW_SPARSE_CONTRACT_JSON" in p for p in llm_neither.prompts)
    assert not any("CONTRACTFLOW_REVIEW_UNITS_JSON" in p for p in llm_neither.prompts)
    assert any("Designate Floor 2 for gym use." in p for p in llm_neither.prompts)


def test_cli_v2_systems_and_compiler_diagnostic_script() -> None:
    root = Path(__file__).resolve().parents[1]

    # Verify run_longgenbench.py accepts v2 systems
    run_spec = importlib.util.spec_from_file_location("run_lg_v2", root / "scripts" / "run_longgenbench.py")
    assert run_spec is not None and run_spec.loader is not None
    run_mod = importlib.util.module_from_spec(run_spec)
    run_spec.loader.exec_module(run_mod)
    args = run_mod.build_parser().parse_args(
        [
            "--dataset",
            "dummy.json",
            "--run-dir",
            "runs/v2",
            "--systems",
            "v2_contractflow",
            "v2_no_contract",
            "v2_no_review_repair",
            "v2_neither",
            "--per-type",
            "5",
        ]
    )
    assert args.systems == ["v2_contractflow", "v2_no_contract", "v2_no_review_repair", "v2_neither"]

    # Verify check_v2_compiler.py metrics and summary formatting
    chk_spec = importlib.util.spec_from_file_location("check_v2", root / "scripts" / "check_v2_compiler.py")
    assert chk_spec is not None and chk_spec.loader is not None
    chk_mod = importlib.util.module_from_spec(chk_spec)
    chk_spec.loader.exec_module(chk_mod)

    dataset_item = {
        "type": "Floor",
        "number": 5,
        "checks_once": {"2": "gym"},
        "checks_range": {"3": "mall", "4": "mall"},
        "checks_periodic": {"4": "garden"},
    }
    contract_dict = {
        "units": [
            {"index": 1, "obligations": ["hallucinated"]},
            {"index": 2, "obligations": ["gym"]},
            {"index": 3, "obligations": ["mall"]},
            {"index": 4, "obligations": []},
            {"index": 5, "obligations": []},
        ]
    }
    metrics = chk_mod.evaluate_compiled_item(dataset_item, contract_dict)
    assert metrics["gold_checked_units"] == 3
    assert metrics["hit_checked_units"] == 2
    assert metrics["missed_checked_units"] == 1
    assert metrics["false_positive_units"] == 1
    summary = chk_mod.summarize_compiler_records(
        [
            {
                "type": "Floor",
                "status": "ok",
                "elapsed_seconds": 0.5,
                "metrics": metrics,
                "completion_history": [{"usage": {"prompt_tokens": 100, "completion_tokens": 150}}],
            }
        ]
    )
    table = chk_mod.format_compiler_table(summary)
    assert "OVERALL" in table and "Floor" in table


def test_length_retry_ratio_and_contract_authority_prompts() -> None:
    assert FlowConfig().length_retry_ratio == 0.7
    for bad_ratio in (-0.1, 1.1):
        try:
            FlowConfig(length_retry_ratio=bad_ratio)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for length_retry_ratio={bad_ratio}")

    contract = DocumentContract(
        global_brief="Two concise handbook chapters.",
        global_invariants=[],
        unit_label="Chapter",
        total_units=2,
        rules=[],
        units=[
            UnitContract(index=1, title="Chapter 1", min_words=20, max_words=40, obligations=["cover Paxos"]),
            UnitContract(index=2, title="Chapter 2", min_words=20, max_words=40, obligations=[]),
        ],
    )

    review_prompt = build_window_review_prompt(contract, [contract.units[0]], {1: "draft text"})
    assert "authoritative" in review_prompt
    assert "Never question, recalculate, or reject an assigned obligation" in review_prompt

    single_repair = build_unit_repair_prompt(
        contract,
        [contract.units[0]],
        {1: "draft text"},
        {1: {"passed": False, "missing_obligations": ["cover Paxos"], "issues": ["missing"]}},
    )
    assert "Treat UNIT_OBLIGATIONS as authoritative" in single_repair
    assert "NEVER remove or omit any listed obligation" in single_repair

    multi_repair = build_unit_repair_prompt(
        contract,
        contract.units,
        {1: "draft 1", 2: "draft 2"},
        {1: {"passed": False, "issues": ["missing"]}, 2: {"passed": False, "issues": ["missing"]}},
    )
    assert "Treat each unit's `obligations` list as authoritative" in multi_repair
    assert "NEVER remove or omit any listed obligation" in multi_repair

    class ToleranceLLM:
        model = "tol-mock"

        def __init__(self) -> None:
            self.singleton_prompts: list[str] = []

        def complete(self, prompt: str, **kwargs) -> str:
            if "CONTRACTFLOW_DRAFT_UNITS" in prompt:
                # Unit 1 has 16 words (>= ceil(20 * 0.7) = 14) -> must NOT retry
                # Unit 2 has 10 words (< 14) -> MUST trigger singleton retry
                u1 = " ".join(f"word{i}" for i in range(16))
                u2 = " ".join(f"short{i}" for i in range(10))
                return f"<<<UNIT 1>>>\n{u1}\n<<<END UNIT 1>>>\n<<<UNIT 2>>>\n{u2}\n<<<END UNIT 2>>>"
            if "CONTRACTFLOW_DRAFT_SINGLE_UNIT" in prompt:
                self.singleton_prompts.append(prompt)
                return " ".join(f"expanded{i}" for i in range(20))
            if "CONTRACTFLOW_REVIEW_UNITS_JSON" in prompt:
                return json.dumps(
                    {
                        "results": [
                            {
                                "index": 1,
                                "passed": False,
                                "missing_obligations": ["cover Paxos"],
                                "issues": ["missing Paxos"],
                            }
                        ]
                    }
                )
            if "CONTRACTFLOW_REPAIR_SINGLE_UNIT" in prompt:
                # 15 words (< before_words=16, < min_words=20, but >= 14) -> MUST be accepted
                return "Paxos consensus protocol ensures safety " + " ".join(f"rep{i}" for i in range(10))
            return "{}"

        def completion_metadata(self) -> dict:
            return {"finish_reason": "stop"}

        def completion_history(self) -> list[dict]:
            return []

    llm = ToleranceLLM()
    result = run_contractflow(
        "Write 2 chapters.",
        llm,
        config=FlowConfig(batch_size=2, length_retry_ratio=0.7, use_contracts=True, use_review_repair=True),
        precompiled_contract=contract,
    )
    assert result["metadata"]["length_retry_ratio"] == 0.7
    assert len(llm.singleton_prompts) == 1
    assert "unit 2" in llm.singleton_prompts[0].lower()
    assert result["metadata"]["window_records"][0]["recovered_indices"] == [2]
    assert result["metadata"]["repair_records"][0]["accepted"] is True
    assert "Paxos" in result["units"][0]["text"]


def test_render_longgen_document_strips_stray_delimiters() -> None:
    item = {
        "prompt": "Write 2 menu days. Use '#*#' to separate days.",
        "type": "Menu Day",
        "number": 2,
        "prefix": "#*# Menu Day 1:",
        "checks_once": {"1": "Burger Mondays"},
        "checks_range": {},
        "checks_periodic": {},
    }
    task = task_from_dataset_item(0, item)
    rendered = render_longgen_document(
        task,
        {
            1: "Menu Day 1:\n#*# Burger Mondays features a gourmet beef burger and fries.",
            2: "#*# Soup and Salad\n#*# Rich chocolate cake for dessert.",
        },
    )
    blocks = parse_blocks(official_output_blocks(rendered), "Menu Day")
    assert completion_rate(blocks, 2) == 100.0
    assert "Burger Mondays" in blocks[1]
    assert "chocolate cake" in blocks[2]


def test_v2_no_contract_runs_prompt_based_semantic_review_and_repair() -> None:
    raw_request = "NO_CONTRACT_SENTINEL: Write 4 chapters. Chapter 2 must explain leader election."

    class PromptRefineMockLLM:
        model = "no-contract-mock"

        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.reviewed_indices: list[int] = []
            self.repaired_prompts: list[str] = []

        def complete(self, prompt: str, **kwargs) -> str:
            self.prompts.append(prompt)
            if "CONTRACTFLOW_DRAFT_UNITS" in prompt:
                match = re.search(r"Write ONLY the requested units: (\[[^\]]+\])", prompt)
                indices = json.loads(match.group(1))
                return "\n".join(
                    f"<<<UNIT {idx}>>>\n"
                    + f"Chapter {idx} discusses general distributed systems concepts "
                    + ("alpha " * 18)
                    + f"\n<<<END UNIT {idx}>>>"
                    for idx in indices
                )
            if "CONTRACTFLOW_REVIEW_UNITS_JSON" in prompt:
                assert "USER_REQUEST:" in prompt
                assert "NO_CONTRACT_SENTINEL" in prompt
                raw_json = prompt.split("UNITS_JSON:\n", 1)[1].strip()
                items = json.loads(raw_json)
                assert all("obligations" not in item for item in items)
                results = []
                for item in items:
                    idx = int(item["index"])
                    self.reviewed_indices.append(idx)
                    if idx == 3:
                        # Deliberately omit index 3 to verify no-contract mode does not destructively rewrite omitted units
                        continue
                    if idx == 2:
                        results.append(
                            {
                                "index": 2,
                                "passed": False,
                                "missing_obligations": ["explain leader election"],
                                "issues": ["Chapter 2 did not explain leader election"],
                            }
                        )
                    else:
                        results.append(
                            {"index": idx, "passed": True, "missing_obligations": [], "issues": []}
                        )
                return json.dumps({"results": results})
            if "CONTRACTFLOW_REPAIR_SINGLE_UNIT" in prompt or "CONTRACTFLOW_REPAIR_UNITS" in prompt:
                self.repaired_prompts.append(prompt)
                assert "USER_REQUEST:" in prompt
                assert "NO_CONTRACT_SENTINEL" in prompt
                assert "UNIT_OBLIGATIONS:" not in prompt
                return "Revised Chapter 2 clearly explaining leader election and Raft terms " + ("beta " * 18)
            return "{}"

        def completion_metadata(self) -> dict:
            return {"finish_reason": "stop"}

        def completion_history(self) -> list[dict]:
            return []

    llm = PromptRefineMockLLM()
    result = run_contractflow(
        raw_request,
        llm,
        config=FlowConfig(batch_size=2, use_contracts=False, use_review_repair=True),
        total_units=4,
        unit_label="Chapter",
        min_words=20,
        max_words=40,
    )
    assert result["system"] == "v2_no_contract"
    assert not any("CONTRACTFLOW_SPARSE_CONTRACT_JSON" in p for p in llm.prompts)
    # Without contracts, reviewer inspects ALL units in each window (no sparse skipping)
    assert sorted(llm.reviewed_indices) == [1, 2, 3, 4]
    # Only unit 2 failed and was repaired; omitted unit 3 was NOT destructively rewritten
    assert len(llm.repaired_prompts) == 1
    assert len(result["metadata"]["repair_records"]) == 1
    assert result["metadata"]["repair_records"][0]["index"] == 2
    assert result["metadata"]["repair_records"][0]["accepted"] is True
    assert "leader election" in result["units"][1]["text"]


def test_run_longgenbench_shared_contract_cache(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    run_spec = importlib.util.spec_from_file_location("run_lg_cache", root / "scripts" / "run_longgenbench.py")
    assert run_spec is not None and run_spec.loader is not None
    run_mod = importlib.util.module_from_spec(run_spec)
    run_spec.loader.exec_module(run_mod)

    args = run_mod.build_parser().parse_args(
        [
            "--dataset",
            "dummy.json",
            "--run-dir",
            str(tmp_path / "run"),
            "--systems",
            "v2_contractflow",
            "v2_no_review_repair",
            "--per-type",
            "1",
            "--contract-cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    assert args.contract_cache_dir == str(tmp_path / "cache")

    item = {
        "prompt": "Design a 4-floor tower. Floor 2 has a gym. Each floor at least 20 words.",
        "type": "Floor",
        "number": 4,
        "prefix": "#*# Floor 1:",
        "checks_once": {"2": "gym"},
        "checks_range": {},
        "checks_periodic": {},
    }
    task = task_from_dataset_item(0, item)
    cfg = FlowConfig(batch_size=2, use_contracts=True, use_review_repair=True)
    cache_dir = tmp_path / "cache"

    compile_calls = 0

    def make_client() -> MockV2FlowLLM:
        nonlocal compile_calls
        compile_calls += 1
        return MockV2FlowLLM()

    c1, hist1, _e1, hit1, path1 = run_mod.load_or_compile_v2_contract(cache_dir, task, cfg, make_client)
    assert hit1 is False
    assert compile_calls == 1
    assert path1.exists()
    assert len(hist1) == 1

    c2, hist2, _e2, hit2, path2 = run_mod.load_or_compile_v2_contract(cache_dir, task, cfg, make_client)
    assert hit2 is True
    assert compile_calls == 1
    assert path2 == path1
    assert c2.to_dict() == c1.to_dict()
    assert hist2 == hist1

    # Different prompt with same task_id must fail fast instead of silently reusing wrong contract
    other_task = task_from_dataset_item(0, {**item, "prompt": "Different tower prompt entirely."})
    try:
        run_mod.load_or_compile_v2_contract(cache_dir, other_task, cfg, make_client)
    except ValueError as err:
        assert "different prompt" in str(err)
    else:
        raise AssertionError("Expected ValueError for prompt_hash mismatch in contract cache")

    # Loading an existing generation JSON containing contract_v2 also works
    legacy_dir = tmp_path / "legacy_gen"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    (legacy_dir / path1.name).write_text(
        json.dumps({"contract_v2": c1.to_dict(), "metadata": {"shared_compile_history": hist1}}),
        encoding="utf-8",
    )
    c3, hist3, _e3, hit3, _p3 = run_mod.load_or_compile_v2_contract(legacy_dir, task, cfg, make_client)
    assert hit3 is True
    assert c3.to_dict() == c1.to_dict()
    assert hist3 == hist1


def test_multilingual_word_count_and_language_agnostic_prompts() -> None:
    # English, Chinese, mixed, Japanese (Kanji+Kana), Korean (Hangul), Spanish/French accented Latin
    assert word_count("Hello world, it's 2026!") == 4
    assert word_count("中文测试共八个字。") == 8
    assert word_count("使用 Saga 和幂等设计") == 8
    assert word_count("東京のシステム設計") == 9
    assert word_count("한국어 테스트") == 6
    assert word_count("España y acción: naïve café") == 5

    compile_prompt = build_sparse_contract_prompt("请撰写一份分布式存储设计文档。")
    assert "strictly in the same language as USER_REQUEST" in compile_prompt
    assert "proportional to the requested document length" in compile_prompt

    contract = DocumentContract(
        global_brief="关于高可用分布式存储系统的中文技术手册。",
        global_invariants=["保持术语一致。"],
        unit_label="章",
        total_units=2,
        rules=[],
        units=[
            UnitContract(index=1, title="第 1 章：架构概览", min_words=500, max_words=800, obligations=["介绍 Raft"]),
            UnitContract(index=2, title="第 2 章：故障恢复", min_words=500, max_words=800, obligations=[]),
        ],
    )
    window_prompt = build_window_draft_prompt(contract, contract.units, rolling_state="")
    assert "strictly in the same language as GLOBAL_BRIEF" in window_prompt
    assert "characters for non-space-delimited scripts" in window_prompt
    assert "boilerplate" in window_prompt
    assert "horizontal rules" in window_prompt

    single_prompt = build_singleton_draft_prompt(contract, contract.units[0], rolling_state="")
    assert "strictly in the same language as GLOBAL_BRIEF" in single_prompt
    assert "characters for non-space-delimited scripts" in single_prompt
    assert "horizontal rules" in single_prompt

    repair_contract = build_unit_repair_prompt(
        contract,
        [contract.units[0]],
        {1: "初稿内容"},
        {1: {"passed": False, "issues": ["缺少 Raft 说明"]}},
    )
    assert "strictly in the same language as GLOBAL_BRIEF" in repair_contract
    assert "characters for non-space-delimited scripts" in repair_contract

    repair_raw = build_unit_repair_prompt(
        contract,
        [contract.units[0]],
        {1: "初稿内容"},
        {1: {"passed": False, "issues": ["缺少 Raft 说明"]}},
        use_contracts=False,
    )
    assert "strictly in the same language as USER_REQUEST" in repair_raw
    assert "characters for non-space-delimited scripts" in repair_raw



