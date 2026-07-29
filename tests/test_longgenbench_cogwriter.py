import json

import pytest

from learnbyai_research.longgenbench import task_from_dataset_item
from learnbyai_research.longgenbench_cogwriter import (
    SPECS,
    assemble_cogwriter_document,
    build_initial_plan_prompt,
    parse_segment_index,
    run_cogwriter,
)


def dataset_item(segment_type: str, number: int) -> dict:
    return {
        "prompt": f"Write {number} entries with special constraints.",
        "type": segment_type,
        "number": number,
        "prefix": f"#*# {segment_type} 1:",
        "checks_once": {},
        "checks_range": {},
        "checks_periodic": {},
    }


def test_initial_prompt_uses_type_specific_plan_schema() -> None:
    task = task_from_dataset_item(0, dataset_item("Floor", 100))
    prompt = build_initial_plan_prompt(task, SPECS["Floor"])
    assert '"floor_plan"' in prompt
    assert '"floor_id"' in prompt
    assert '"purpose"' in prompt


def test_assemble_cogwriter_document_uses_official_markers() -> None:
    text = assemble_cogwriter_document(
        [
            {"segment_id": "Week 1", "text": "Alpha"},
            {"segment_id": "Week 2", "text": "Beta"},
        ]
    )
    assert text.startswith("#*# Week 1:")
    assert "#*# Week 2:" in text
    assert text.endswith("*** finished ***")


def test_parse_segment_index_supports_multiword_type() -> None:
    assert parse_segment_index("Menu Week 17 (April 23rd)", "Menu Week") == 17


def test_run_cogwriter_checkpoints_successful_segments_before_raising(tmp_path) -> None:
    task = task_from_dataset_item(0, dataset_item("Week", 2))
    plan = [
        {"week_id": "Week 1", "events": "Alpha"},
        {"week_id": "Week 2", "events": "Beta"},
    ]
    (tmp_path / "plan_initial.json").write_text(
        json.dumps({"plan": plan}),
        encoding="utf-8",
    )
    (tmp_path / "plan_revised.json").write_text(
        json.dumps({"plan": plan}),
        encoding="utf-8",
    )

    class FakeClient:
        model = "fake"

        def complete(self, prompt, **kwargs):
            if "Write a 100-word week entry for Week 2." in prompt:
                raise RuntimeError("rate limited")
            return '{"diary_entry":"Alpha text"}'

        def completion_history(self):
            return []

    with pytest.raises(RuntimeError, match="1 failed segment"):
        run_cogwriter(
            task,
            FakeClient,
            checkpoint_dir=tmp_path,
            segment_workers=2,
            max_refinements=0,
        )

    saved = json.loads((tmp_path / "segments" / "0001.json").read_text(encoding="utf-8"))
    assert saved["segment_id"] == "Week 1"
    assert not (tmp_path / "segments" / "0002.json").exists()
