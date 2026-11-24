from __future__ import annotations

from learnbyai_research.longgenbench import (
    HARNESS_PIPELINE_VERSION,
    assemble_document,
    SegmentContract,
    _load_harness_checkpoint,
    _write_harness_checkpoint,
    apply_plan_corrections,
    compile_prompt_contracts,
    parse_contract_plan,
    parse_document_blocks,
    parse_minimum_words,
    run_harness,
    select_balanced_indices,
    structural_issues,
    task_from_dataset_item,
)
from learnbyai_research.longgenbench_eval import failed_evaluation, summarize_evaluations


def dataset_item(segment_type: str, number: int = 2) -> dict:
    return {
        "prompt": f"Write {number} entries. Each entry should be at least 20 words.",
        "type": segment_type,
        "number": number,
        "prefix": f"#*# {segment_type} 1:",
        "checks_once": {},
        "checks_range": {},
        "checks_periodic": {},
    }


def test_select_balanced_indices_uses_dataset_order() -> None:
    dataset = [
        dataset_item("Week"),
        dataset_item("Floor"),
        dataset_item("Menu Week"),
        dataset_item("Block"),
        dataset_item("Week"),
        dataset_item("Floor"),
        dataset_item("Menu Week"),
        dataset_item("Block"),
    ]
    assert select_balanced_indices(dataset, per_type=2) == list(range(8))


def test_parse_minimum_words() -> None:
    assert parse_minimum_words("Each diary entry should be at least 200 words.") == 200
    assert parse_minimum_words("Each containing at least 150 words.") == 150
    assert parse_minimum_words("Aim for between 200 and 220 words per menu.") == 200


def test_parse_contract_plan_requires_every_index() -> None:
    task = task_from_dataset_item(0, dataset_item("Week"))
    summary, contracts = parse_contract_plan(
        task,
        {
            "document_summary": "year",
            "segments": [
                {"index": 1, "purpose": "a", "required_events": ["birthday"]},
                {"index": 2, "purpose": "b", "required_events": []},
            ],
        },
    )
    assert summary == "year"
    assert [contract.index for contract in contracts] == [1, 2]


def test_apply_plan_corrections_replaces_events() -> None:
    contracts = [SegmentContract(index=1, required_events=["old"])]
    applied = apply_plan_corrections(
        contracts,
        {"corrections": [{"index": 1, "replace_required_events": ["new"]}]},
    )
    assert contracts[0].required_events == ["new"]
    assert applied[0]["before"] == ["old"]


def test_parse_document_blocks_supports_hash_star_separator() -> None:
    blocks = parse_document_blocks(
        "#*# Menu Week 1:\nAlpha\n\n#*# Menu Week 2:\nBeta\n*** finished ***",
        "Menu Week",
    )
    assert blocks == {1: "Alpha", 2: "Beta"}


def test_assemble_document_does_not_duplicate_existing_markers() -> None:
    task = task_from_dataset_item(0, dataset_item("Week"))
    text = assemble_document(
        task,
        {
            1: "#*# Week 1 (January 1st - January 7th):\nFirst body.",
            2: "Second body.",
        },
    )
    assert text.count("#*# Week 1") == 1
    assert text.count("#*# Week 2") == 1


def test_parse_generated_segments_supports_tagged_prose() -> None:
    from learnbyai_research.longgenbench import parse_generated_segments

    contracts = [SegmentContract(index=1), SegmentContract(index=2)]
    parsed = parse_generated_segments(
        contracts,
        "<<<SEGMENT 1>>>\nAlpha\n<<<END SEGMENT 1>>>\n"
        "<<<SEGMENT 2>>>\nBeta with \"quotes\"\n<<<END SEGMENT 2>>>",
    )
    assert parsed == {1: "Alpha", 2: 'Beta with "quotes"'}


def test_generated_segments_accept_only_explicit_official_markers_as_fallback() -> None:
    from learnbyai_research.longgenbench import parse_generated_segments

    contracts = [SegmentContract(index=index) for index in (1, 2, 3, 4, 5)]
    raw = (
        "Unnumbered opening.\n"
        "#*# Menu Day 1 (January 1):\nDuplicate from benchmark format.\n"
        "<<<SEGMENT 1>>>\nAlpha\n#*# Menu Day 4:\nPart of Alpha.\n<<<END SEGMENT 1>>>\n"
        "#*# Menu Day 2 (January 2):\nBeta.\n"
        "#*# Menu Day 2:\nDuplicate Beta.\n"
        "#*# Menu Day 9:\nOut of batch.\n"
        "#*# Menu Day 3 (January 3)\nGamma.\n"
        "*** finished ***\n"
    )
    tagged = {1: "Alpha\n#*# Menu Day 4:\nPart of Alpha."}
    assert parse_generated_segments(contracts, raw) == tagged
    assert parse_generated_segments(contracts, raw, segment_type="Menu Day") == {
        **tagged, 2: "Beta.", 3: "Gamma.",
    }
    assert parse_generated_segments(
        contracts, "Menu Day 4:\nA bare heading.", segment_type="Menu Day",
    ) == {}


def test_contractflow_accepts_official_format_in_generation_and_repair() -> None:
    class OfficialFormatLLM:
        model = "format-fixture"

        def complete(self, prompt: str, **kwargs) -> str:
            if prompt.startswith("LONGGEN_WRITE_SEGMENTS"):
                return "#*# Week 1:\n" + "alpha " * 20
            assert prompt.startswith("LONGGEN_REPAIR_SEGMENTS")
            return "#*# Week 2:\n" + "beta " * 20

    output = run_harness(
        task_from_dataset_item(0, dataset_item("Week")),
        OfficialFormatLLM(),
        batch_size=2,
        plan_max_output_tokens=1,
        generation_max_output_tokens=1,
        review_max_output_tokens=1,
        repair_max_output_tokens=1,
        use_llm_plan=False,
    )
    assert [section["index"] for section in output["segments"]] == [1, 2]
    assert output["metadata"]["repair_records"][0]["index"] == 2
    assert output["final_text"].count("#*# Week ") == 2


def test_structural_issues_detects_missing_and_short_segments() -> None:
    task = task_from_dataset_item(0, dataset_item("Block"))
    contracts = [SegmentContract(index=1), SegmentContract(index=2)]
    issues = structural_issues(task, contracts, {1: "short"})
    assert sorted(issues) == [1, 2]


def test_summarize_evaluations_groups_systems_and_types() -> None:
    rows = [
        {
            "system": "direct",
            "type": "Week",
            "completion_rate": 0.5,
            "length_pass_rate": 0.5,
            "accuracy_once": 0.2,
            "accuracy_range": 0.4,
            "accuracy_periodic": 0.6,
            "average_accuracy": 0.4,
        },
        {
            "system": "direct",
            "type": "Floor",
            "completion_rate": 1.0,
            "length_pass_rate": 1.0,
            "accuracy_once": 0.8,
            "accuracy_range": 0.6,
            "accuracy_periodic": 0.4,
            "average_accuracy": 0.6,
        },
    ]
    summary = summarize_evaluations(rows)
    assert summary["systems"]["direct"]["task_count"] == 2
    assert summary["systems"]["direct"]["completion_rate"] == 0.75


def test_harness_checkpoint_round_trip(tmp_path) -> None:
    task = task_from_dataset_item(0, dataset_item("Week"))
    path = tmp_path / "checkpoint.json"
    _write_harness_checkpoint(
        path,
        task,
        {"segments": []},
        "summary",
        [SegmentContract(index=1), SegmentContract(index=2)],
        [],
        [],
        {1: "text"},
        [{"indices": [1]}],
        [],
        {},
        [],
        stage="generated_through_1",
    )
    checkpoint = _load_harness_checkpoint(path, task)
    assert checkpoint is not None
    assert checkpoint["pipeline_version"] == HARNESS_PIPELINE_VERSION
    assert checkpoint["stage"] == "generated_through_1"
    assert checkpoint["segments"]["1"] == "text"


def test_harness_checkpoint_rejects_another_pipeline_version(tmp_path) -> None:
    task = task_from_dataset_item(0, dataset_item("Week"))
    path = tmp_path / "checkpoint.json"
    path.write_text(
        '{"task_id":"0000_week","pipeline_version":"optimized_v1"}',
        encoding="utf-8",
    )
    try:
        _load_harness_checkpoint(path, task)
    except ValueError as error:
        assert "incompatible pipeline version" in str(error)
    else:
        raise AssertionError("expected an incompatible checkpoint to be rejected")


class AblationLLM:
    model = "ablation-test"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        if "LONGGEN_REVIEW_SEGMENTS_JSON" in prompt:
            return '{"results":[]}'
        if "LONGGEN_WRITE_SEGMENTS" in prompt or "LONGGEN_REPAIR_SEGMENTS" in prompt:
            return (
                "<<<SEGMENT 1>>>\n"
                + "word " * 20
                + "\n<<<END SEGMENT 1>>>\n"
                + "<<<SEGMENT 2>>>\n"
                + "word " * 20
                + "\n<<<END SEGMENT 2>>>"
            )
        return "{}"

    def completion_history(self) -> list[dict]:
        return []


def test_no_contract_ablation_skips_contract_compilation_and_review() -> None:
    task = task_from_dataset_item(0, dataset_item("Week"))
    llm = AblationLLM()
    output = run_harness(
        task,
        llm,
        batch_size=2,
        plan_max_output_tokens=1,
        generation_max_output_tokens=1,
        review_max_output_tokens=1,
        repair_max_output_tokens=1,
        use_llm_plan=False,
        use_contracts=False,
        use_review_repair=True,
    )
    assert output["system"] == "no_contract"
    assert output["metadata"]["compiled_contract_overrides"] == []
    assert not any("LONGGEN_REVIEW_SEGMENTS_JSON" in prompt for prompt in llm.prompts)


def test_no_review_repair_ablation_stops_after_generation() -> None:
    task = task_from_dataset_item(0, dataset_item("Week"))
    llm = AblationLLM()
    output = run_harness(
        task,
        llm,
        batch_size=2,
        plan_max_output_tokens=1,
        generation_max_output_tokens=1,
        review_max_output_tokens=1,
        repair_max_output_tokens=1,
        use_llm_plan=False,
        use_contracts=True,
        use_review_repair=False,
    )
    assert output["system"] == "no_review_repair"
    assert output["metadata"]["review_results"] == []
    assert output["metadata"]["repair_records"] == []


def test_compile_week_contracts_maps_dates_ranges_and_periodicity() -> None:
    item = dataset_item("Week", number=52)
    item["prompt"] = (
        "Family member birthday: my father (birthday on July 27),\n"
        "2) Attending a festival in week 24-25.\n"
        "3) Attend a workshop every 5 weeks on weekends, starting from week 13.\n"
        "Each entry should be at least 200 words."
    )
    compiled = compile_prompt_contracts(task_from_dataset_item(0, item))
    assert compiled[30] == ["my father birthday"]
    assert compiled[24] == ["Attending a festival"]
    assert compiled[25] == ["Attending a festival"]
    assert compiled[13] == ["Attend a workshop"]
    assert compiled[48] == ["Attend a workshop"]


def test_compile_block_contracts_uses_public_coordinate_rules() -> None:
    item = dataset_item("Block", number=100)
    item["prompt"] = (
        "Designate Block at (4, 9) for university use.\n"
        "Allocate a shopping district along the column from (1, 9) to (4, 9).\n"
        "Include a bus station starting from Block at (0, 5) with an interval "
        "of every 2 blocks along the column.\n"
        "Each containing at least 150 words."
    )
    compiled = compile_prompt_contracts(task_from_dataset_item(0, item))
    assert compiled[95] == ["university", "shopping district"]
    assert [index for index in (51, 53, 55, 57, 59) if "bus station" in compiled[index]] == [
        51,
        53,
        55,
        57,
        59,
    ]


def test_failed_evaluation_counts_technical_failure_as_zero() -> None:
    item = dataset_item("Week", number=52)
    item["checks_once"] = {"10": "birthday"}
    row = failed_evaluation(
        0,
        item,
        "direct",
        {"error_type": "RuntimeError", "error_message": "timeout"},
    )
    assert row["status"] == "failed_technical"
    assert row["completion_rate"] == 0.0
    assert row["average_accuracy"] == 0.0
    assert row["checks"][0]["present"] is False


def test_select_balanced_indices_supports_32k_types() -> None:
    dataset = [
        dataset_item("Day", number=365),
        dataset_item("Floor", number=300),
        dataset_item("Menu Day", number=365),
        dataset_item("Block", number=361),
        dataset_item("Day", number=365),
        dataset_item("Floor", number=300),
        dataset_item("Menu Day", number=365),
        dataset_item("Block", number=361),
    ]
    assert select_balanced_indices(dataset, per_type=2) == list(range(8))


def test_run_longgenbench_selection_and_cli_options() -> None:
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_longgenbench.py"
    spec = importlib.util.spec_from_file_location("run_longgenbench", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dataset = [
        dataset_item("Day", number=365),
        dataset_item("Floor", number=300),
        dataset_item("Menu Day", number=365),
        dataset_item("Block", number=361),
    ]
    assert module.select_indices(dataset, run_all=True, per_type=None, indices=None) == [0, 1, 2, 3]
    assert module.select_indices(dataset, run_all=False, per_type=1, indices=None) == [0, 1, 2, 3]
    assert module.select_indices(dataset, run_all=False, per_type=None, indices=[3, 1]) == [3, 1]

    args = module.build_parser().parse_args(
        [
            "--dataset",
            "dummy.json",
            "--run-dir",
            "runs/test",
            "--systems",
            "direct",
            "contractflow_optimized",
            "--all",
            "--no-llm-plan",
            "--continue-on-error",
        ]
    )
    assert args.all is True
    assert args.no_llm_plan is True
    assert args.continue_on_error is True
    assert args.systems == ["direct", "contractflow_optimized"]


def test_eval_longgenbench_handles_failed_generation_and_summary(tmp_path) -> None:
    import importlib.util
    import json
    from pathlib import Path

    from learnbyai_research.longgenbench_official_eval import summarize_official

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_longgenbench.py"
    spec = importlib.util.spec_from_file_location("eval_longgenbench", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dataset = [dataset_item("Day", number=365)]
    dataset[0]["checks_once"] = {"1": "special event"}
    err_path = tmp_path / "errors" / "direct" / "0000_day.json"
    err_path.parent.mkdir(parents=True, exist_ok=True)
    err_path.write_text(
        json.dumps({"error_type": "TimeoutError", "error_message": "timed out"}),
        encoding="utf-8",
    )
    args = module.build_parser().parse_args(
        [
            "--dataset",
            "dummy.json",
            "--run-dir",
            str(tmp_path),
            "--systems",
            "direct",
            "--all",
        ]
    )
    assert args.mode == "official"
    status, row = module.evaluate_one(args, dataset, "direct", 0)
    assert status == "failed_generation"
    assert row is not None and row["status"] == "failed_technical"
    assert row["total_checks_once"] == 1

    status_cached, row_cached = module.evaluate_one(args, dataset, "direct", 0)
    assert status_cached == "cached"
    assert row_cached == row

    summary = {"mode": "official", "systems": {"direct": summarize_official([row], model_name="direct")}}
    table = module.format_official_summary_table(summary)
    assert "direct" in table
    assert "Strict Metrics" in table


def test_official_eval_matches_upstream_semantics_and_export(tmp_path) -> None:
    import importlib.util
    import json
    from pathlib import Path

    from learnbyai_research.longgenbench_official_eval import (
        evaluate_official_output,
        summarize_official,
    )

    class JudgeStub:
        model = "judge-stub"

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete(self, prompt: str, **kwargs) -> str:
            self.prompts.append(prompt)
            return "yes" if "solar observatory" in prompt else "no"

        def completion_history(self) -> list[dict]:
            return []

    item = dataset_item("Floor", number=4)
    item["checks_once"] = {"1": "solar observatory", "4": "rooftop garden"}
    item["checks_range"] = {"2": "shopping mall"}
    item["checks_periodic"] = {"2": "rest area"}

    # Only Floor 1 and Floor 2 are generated; Floor 3 and Floor 4 are missing.
    output = {
        "task": {"dataset_index": 0, "task_id": "0000_floor"},
        "system": "direct",
        "final_text": "#*# Floor 1:\nFeatures a solar observatory.\n#*# Floor 2:\nEmpty lobby.",
        "metadata": {"model": "qwen-3.5-9b"},
    }
    judge = JudgeStub()
    record = evaluate_official_output(item, output, judge)

    # Missing Floor 4 must NOT be sent to the judge (exact official eval.py behavior).
    assert len(judge.prompts) == 3
    assert record["completion_rate"] == 50.0
    assert record["results_once"] == {"1": "yes"}
    assert record["total_checks_once"] == 2
    assert record["results_range"] == {"2": "no"}
    assert record["results_periodic"] == {"2": "no"}

    summary = summarize_official([record], model_name="direct")
    # Official eval.py conditional accuracy on present blocks: once=1/1=1.0, range=0/1, periodic=0/1
    assert summary["Completion Rate"] == 50.0
    assert summary["Accuracy Once"] == 1.0
    assert summary["Accuracy Range"] == 0.0
    assert summary["Accuracy Periodic"] == 0.0
    assert abs(summary["Average Accuracy"] - (1.0 / 3.0)) < 1e-9
    # Official static.ipynb strict accuracy over all checks: once=1/2=0.5, total yes/checks = 1/4 = 0.25
    assert summary["Strict Accuracy Once"] == 0.5
    assert summary["Ratio Yes/Checks"] == 0.25

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_longgenbench.py"
    spec = importlib.util.spec_from_file_location("eval_longgenbench_export", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    gen_path = tmp_path / "generations" / "direct" / "0000_floor.json"
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    gen_path.write_text(json.dumps(output), encoding="utf-8")
    exported = module.export_official_files(tmp_path, [item], [0], ["direct"])
    exported_items = json.loads(exported["direct"].read_text(encoding="utf-8"))
    assert len(exported_items) == 1
    assert exported_items[0]["type"] == "Floor"
    assert exported_items[0]["number"] == 4
    assert len(exported_items[0]["output_blocks"]) == 3
    assert exported_items[0]["word_count"] > 0


def test_complete_json_scales_budget_without_echoing_truncated_response() -> None:
    from learnbyai_research.longgenbench import complete_json

    class TruncatingLLM:
        model = "truncating-fixture"

        def __init__(self) -> None:
            self.calls: list[tuple[str, int | None]] = []
            self._meta: dict = {}

        def complete(self, prompt: str, **kwargs) -> str:
            max_tokens = kwargs.get("max_output_tokens")
            self.calls.append((prompt, max_tokens))
            if len(self.calls) == 1:
                self._meta = {"finish_reason": "length"}
                return '{"segments": [{"index": 1'
            self._meta = {"finish_reason": "stop"}
            return '{"ok": true}'

        def completion_metadata(self) -> dict:
            return dict(self._meta)

        def completion_history(self) -> list[dict]:
            return []

    llm = TruncatingLLM()
    payload = complete_json(
        llm,
        "BASE_PROMPT",
        system="sys",
        temperature=0.1,
        max_output_tokens=2048,
    )
    assert payload == {"ok": True}
    assert len(llm.calls) == 2
    assert llm.calls[1][1] == 4096
    assert "MALFORMED_RESPONSE" not in llm.calls[1][0]
    assert "BASE_PROMPT" in llm.calls[1][0]


def test_run_harness_chunks_large_llm_plans_and_backfills_missing_indices() -> None:
    import json
    import re

    class ChunkedPlanLLM:
        model = "chunked-plan-fixture"

        def __init__(self) -> None:
            self.plan_calls = 0
            self.audit_calls = 0
            self._meta: dict = {"finish_reason": "stop"}

        def complete(self, prompt: str, **kwargs) -> str:
            if "LONGGEN_CONTRACT_PLAN_CHUNK_JSON" in prompt:
                self.plan_calls += 1
                match = re.search(r"Return ONLY the requested segment indices: (\[[^\]]+\])", prompt)
                indices = json.loads(match.group(1)) if match else []
                # Deliberately omit the last index on the very first chunk to test backfill.
                returned = indices[:-1] if self.plan_calls == 1 and len(indices) > 1 else indices
                self._meta = {"finish_reason": "stop"}
                return json.dumps(
                    {
                        "document_summary": "chunked summary",
                        "segments": [
                            {"index": idx, "purpose": f"plan {idx}", "required_events": [], "continuity": ""}
                            for idx in returned
                        ],
                    }
                )
            if "LONGGEN_CONTRACT_AUDIT_JSON" in prompt:
                self.audit_calls += 1
                self._meta = {"finish_reason": "stop"}
                return '{"corrections": []}'
            if "LONGGEN_WRITE_SEGMENTS" in prompt:
                indices = [int(x) for x in re.findall(r'"index":\s*(\d+)', prompt)]
                return "\n".join(
                    f"<<<SEGMENT {idx}>>>\n" + "word " * 20 + f"\n<<<END SEGMENT {idx}>>>"
                    for idx in indices
                )
            return '{"results": []}'

        def completion_metadata(self) -> dict:
            return dict(self._meta)

        def completion_history(self) -> list[dict]:
            return []

    task = task_from_dataset_item(0, dataset_item("Floor", number=10))
    llm = ChunkedPlanLLM()
    output = run_harness(
        task,
        llm,
        batch_size=5,
        plan_max_output_tokens=2048,
        generation_max_output_tokens=2048,
        review_max_output_tokens=2048,
        repair_max_output_tokens=2048,
        use_llm_plan=True,
        plan_chunk_size=4,
    )
    # 10 segments with chunk_size=4 -> 3 initial chunks + 1 backfill chunk = 4 plan calls
    assert llm.plan_calls == 4
    assert llm.audit_calls == 3
    assert [c["index"] for c in output["contracts"]] == list(range(1, 11))
    assert [s["index"] for s in output["segments"]] == list(range(1, 11))


def test_run_harness_splits_truncated_review_batch_instead_of_crashing() -> None:
    import json
    import re

    class SplittingReviewLLM:
        model = "split-review-fixture"

        def __init__(self) -> None:
            self.review_batch_sizes: list[int] = []
            self._meta: dict = {"finish_reason": "stop"}

        def complete(self, prompt: str, **kwargs) -> str:
            if "LONGGEN_REVIEW_SEGMENTS_JSON" in prompt:
                segments_json = prompt.split("SEGMENTS_TO_REVIEW:\n", 1)[1].strip()
                items = json.loads(segments_json)
                indices = [int(entry["index"]) for entry in items]
                self.review_batch_sizes.append(len(indices))
                if len(indices) > 1:
                    self._meta = {"finish_reason": "length"}
                    return '{"results": [{"index": 1'
                self._meta = {"finish_reason": "stop"}
                return json.dumps(
                    {
                        "results": [
                            {"index": indices[0], "passed": True, "missing_events": [], "issues": []}
                        ]
                    }
                )
            if "LONGGEN_WRITE_SEGMENTS" in prompt:
                contracts_json = prompt.split("SEGMENT_CONTRACTS:\n", 1)[1].split("\n\nUSER_PROMPT:", 1)[0].strip()
                indices = [int(entry["index"]) for entry in json.loads(contracts_json)]
                return "\n".join(
                    f"<<<SEGMENT {idx}>>>\n" + "word " * 22 + f"\n<<<END SEGMENT {idx}>>>"
                    for idx in indices
                )
            return "{}"

        def completion_metadata(self) -> dict:
            return dict(self._meta)

        def completion_history(self) -> list[dict]:
            return []

    item = dataset_item("Week", number=2)
    item["prompt"] = (
        "Diary:\n"
        "2) Attending a festival in week 1-2.\n"
        "Each entry should be at least 20 words."
    )
    task = task_from_dataset_item(0, item)
    llm = SplittingReviewLLM()
    output = run_harness(
        task,
        llm,
        batch_size=2,
        plan_max_output_tokens=2048,
        generation_max_output_tokens=2048,
        review_max_output_tokens=2048,
        repair_max_output_tokens=2048,
        use_llm_plan=False,
    )
    assert len(output["segments"]) == 2
    assert 2 in llm.review_batch_sizes
    assert llm.review_batch_sizes.count(1) == 2
    assert output["metadata"]["repair_records"] == []



