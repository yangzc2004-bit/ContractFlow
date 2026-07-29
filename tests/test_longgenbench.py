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
