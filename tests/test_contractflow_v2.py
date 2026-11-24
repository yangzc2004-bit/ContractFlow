from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

from learnbyai_research.contractflow_v2 import (
    ContractRule,
    FlowConfig,
    UnitContract,
    compile_sparse_contract,
    draft_token_budget,
    effective_batch_size,
    expand_contract_rules,
    parse_singleton_unit,
    parse_unit_batch,
    repair_token_budget,
    review_token_budget,
    run_contractflow,
)
from learnbyai_research.longgenbench import task_from_dataset_item
from learnbyai_research.longgenbench_official_eval import (
    completion_rate,
    official_output_blocks,
    parse_blocks,
)
from learnbyai_research.longgenbench_v2 import run_longgen_v2


def test_expand_contract_rules_supports_all_four_scopes_and_bounded_stride() -> None:
    rules = [
        ContractRule(scope="all", obligation="Do not invent citations."),
        ContractRule(scope="single", index=1, obligation="Introduce core notation."),
        ContractRule(scope="range", start=4, end=6, obligation="Cover consensus algorithms."),
        ContractRule(scope="every", start=3, step=3, obligation="Include a hands-on lab."),
        ContractRule(scope="every", start=2, end=8, step=3, obligation="Bounded stride note."),
        ContractRule(scope="single", index=999, obligation="Out of bounds ignored."),
    ]
    invariants, obligations = expand_contract_rules(12, rules)
    assert invariants == ["Do not invent citations."]
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
