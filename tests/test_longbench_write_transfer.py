from __future__ import annotations

import json

import pytest

from learnbyai_research.longbench_write_transfer import (
    MORELONGWRITE_CONTRACT_PROFILE,
    MORELONGWRITE_OPTIMIZED_PROFILE,
    LongBenchWriteTask,
    contract_section_bounds,
    contract_section_word_bounds,
    normalize_section_targets,
    official_length_count,
    parse_agentwrite_plan,
    parse_contract_plan,
    run_harness,
    word_count,
)
from learnbyai_research.morelongwrite_optimized import run_morelongwrite_optimized


def test_parse_agentwrite_official_plan_format() -> None:
    plan = "\n".join(
        [
            "Paragraph 1 - Main Point: Introduce the topic - Word Count: 400 words",
            "Paragraph 2 - Main Point: Develop the topic - Word Count: 600 words.",
        ]
    )
    steps = parse_agentwrite_plan(plan)
    assert [step.index for step in steps] == [1, 2]
    assert [step.target_words for step in steps] == [400, 600]


def test_parse_contract_plan_enforces_frozen_shape() -> None:
    payload = {
        "document_contract": {"global_goal": "Explain pandas."},
        "sections": [
            {
                "index": index,
                "title": f"Section {index}",
                "purpose": "Explain.",
                "target_words": 800,
                "required_points": [],
                "depends_on": [],
                "continuity_checks": [],
            }
            for index in range(1, 6)
        ],
    }
    assert parse_contract_plan(json.dumps(payload), 4000)["sections"][0]["index"] == 1


def test_parse_contract_plan_normalizes_budget_without_changing_sections() -> None:
    payload = {
        "document_contract": {},
        "sections": [
            {"index": index, "target_words": 1000}
            for index in range(1, 6)
        ],
    }
    parsed = parse_contract_plan(json.dumps(payload), 4000)
    assert [section["target_words"] for section in parsed["sections"]] == [800] * 5
    assert parsed["budget_normalization"]["original_total"] == 5000


def test_parse_contract_plan_normalizes_positive_out_of_bound_targets() -> None:
    payload = {
        "document_contract": {},
        "sections": [
            {"index": index, "target_words": value}
            for index, value in enumerate(
                [800, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 200],
                start=1,
            )
        ],
    }
    parsed = parse_contract_plan(json.dumps(payload), 10000)
    assert [section["target_words"] for section in parsed["sections"]] == [1000] * 10


def test_budget_normalization_rejects_impossible_section_count() -> None:
    with pytest.raises(ValueError, match="cannot normalize"):
        normalize_section_targets([1000] * 5, 1000)


def test_word_count_handles_english_prose() -> None:
    assert word_count("Pandas are black-and-white bears.") == 4


def test_official_length_count_handles_chinese_and_english() -> None:
    assert official_length_count("Pandas are black-and-white bears. \u718a\u732b") == 8


def test_contract_section_bounds_preserve_10k_shape_and_extend_ultra_long() -> None:
    assert contract_section_bounds(10000) == (10, 10)
    assert contract_section_bounds(12000) == (12, 20)
    assert contract_section_bounds(15000) == (15, 20)
    assert contract_section_bounds(20000) == (20, 20)
    assert contract_section_bounds(32000) == (8, 32)
    assert contract_section_bounds(64000) == (16, 64)
    assert contract_section_word_bounds(20000) == (350, 1000)
    assert contract_section_word_bounds(32000) == (1000, 4000)


def test_parse_contract_plan_accepts_twenty_sections_for_20k() -> None:
    payload = {
        "document_contract": {},
        "sections": [
            {"index": index, "target_words": 1000}
            for index in range(1, 21)
        ],
    }
    parsed = parse_contract_plan(json.dumps(payload), 20000)
    assert len(parsed["sections"]) == 20


def test_parse_contract_plan_accepts_sixteen_sections_for_64k() -> None:
    payload = {
        "document_contract": {},
        "sections": [
            {"index": index, "target_words": 4000}
            for index in range(1, 17)
        ],
    }
    parsed = parse_contract_plan(json.dumps(payload), 64000)
    assert len(parsed["sections"]) == 16


def test_morelongwrite_profile_uses_fixed_tier_section_counts() -> None:
    assert contract_section_bounds(
        16000,
        MORELONGWRITE_CONTRACT_PROFILE,
    ) == (8, 8)
    assert contract_section_bounds(
        32000,
        MORELONGWRITE_CONTRACT_PROFILE,
    ) == (12, 12)
    assert contract_section_bounds(
        64000,
        MORELONGWRITE_CONTRACT_PROFILE,
    ) == (16, 16)
    assert contract_section_word_bounds(
        16000,
        MORELONGWRITE_CONTRACT_PROFILE,
    ) == (1200, 2800)
    assert contract_section_word_bounds(
        64000,
        MORELONGWRITE_CONTRACT_PROFILE,
    ) == (3200, 4800)


def test_morelongwrite_optimized_profile_uses_parallel_friendly_sections() -> None:
    assert contract_section_bounds(
        16000,
        MORELONGWRITE_OPTIMIZED_PROFILE,
    ) == (12, 12)
    assert contract_section_bounds(
        32000,
        MORELONGWRITE_OPTIMIZED_PROFILE,
    ) == (20, 20)
    assert contract_section_bounds(
        64000,
        MORELONGWRITE_OPTIMIZED_PROFILE,
    ) == (40, 40)
    assert contract_section_word_bounds(
        64000,
        MORELONGWRITE_OPTIMIZED_PROFILE,
    ) == (1200, 2200)


def test_morelongwrite_profile_normalizes_eight_section_16k_plan() -> None:
    payload = {
        "document_contract": {},
        "sections": [
            {
                "index": index,
                "title": f"Section {index}",
                "target_words": 1500,
            }
            for index in range(1, 9)
        ],
    }
    parsed = parse_contract_plan(
        json.dumps(payload),
        16000,
        profile=MORELONGWRITE_CONTRACT_PROFILE,
    )
    assert [section["target_words"] for section in parsed["sections"]] == [2000] * 8


class FakeModel:
    def __init__(self, responder, model: str = "deepseek-v4-flash") -> None:
        self.model = model
        self.responder = responder
        self.calls: list[dict] = []

    def complete(self, prompt: str, **kwargs) -> str:
        response = self.responder(prompt)
        self.calls.append(
            {
                "model": self.model,
                "elapsed_seconds": 0.01,
                "request_max_output_tokens": kwargs.get("max_output_tokens"),
            }
        )
        return response

    def completion_metadata(self) -> dict:
        return dict(self.calls[-1])

    def completion_history(self) -> list[dict]:
        return [dict(item) for item in self.calls]


def test_morelongwrite_harness_batches_review_and_skips_unneeded_repair(
    tmp_path,
) -> None:
    plan = {
        "document_contract": {
            "artifact_type": "test",
            "global_goal": "test",
        },
        "sections": [
            {
                "index": index,
                "title": f"Section {index}",
                "purpose": "Develop the document.",
                "target_words": 2000,
                "required_points": [],
                "depends_on": [],
                "continuity_checks": [],
            }
            for index in range(1, 9)
        ],
    }

    def author_response(prompt: str) -> str:
        if prompt.startswith("DOCUMENT_CONTRACT_PLAN_JSON"):
            return json.dumps(plan)
        if prompt.startswith("WRITE_MORELONGWRITE_SECTION"):
            return "word " * 2000
        raise AssertionError(prompt[:80])

    def reviewer_response(prompt: str) -> str:
        marker = "Return exactly one review for every index in "
        indices = json.loads(prompt.split(marker, 1)[1].split(".", 1)[0])
        return json.dumps(
            {
                "reviews": [
                    {
                        "index": index,
                        "passed": True,
                        "issues": [],
                        "summary": "Pass.",
                    }
                    for index in indices
                ]
            }
        )

    author = FakeModel(author_response)
    reviewers: list[FakeModel] = []

    def reviewer_factory() -> FakeModel:
        reviewer = FakeModel(reviewer_response)
        reviewers.append(reviewer)
        return reviewer

    def unexpected_repair_factory():
        raise AssertionError("repair should not be called")

    result = run_harness(
        LongBenchWriteTask(
            index=0,
            prompt="Write a 16,000-word test document.",
            task_type="Academic and Technical Writing",
            target_words=16000,
        ),
        "academic",
        author,
        reviewer_factory,
        unexpected_repair_factory,
        checkpoint_dir=tmp_path,
        planning_temperature=0.2,
        writing_temperature=0.35,
        review_temperature=0.0,
        repair_temperature=0.2,
        planning_max_output_tokens=16384,
        section_max_output_tokens=6144,
        review_max_output_tokens=4096,
        repair_max_output_tokens=6144,
        max_plan_attempts=3,
        max_repair_rounds_per_section=1,
        minimum_section_length_ratio=0.8,
        maximum_section_length_ratio=1.2,
        profile=MORELONGWRITE_CONTRACT_PROFILE,
        review_batch_size=4,
        repair_severities=["high"],
        continuity_context_characters=12000,
    )

    assert len(author.calls) == 9
    assert len(reviewers) == 2
    assert sum(len(reviewer.calls) for reviewer in reviewers) == 2
    assert len(result["units"]) == 8
    assert result["metadata"]["word_count"] == 16000
    assert len(result["metadata"]["completion_history"]) == 11


def test_morelongwrite_optimized_preserves_flow_and_skips_repair(tmp_path) -> None:
    plan = {
        "document_contract": {
            "artifact_type": "test",
            "global_goal": "Write the requested document.",
        },
        "sections": [
            {
                "index": index,
                "title": f"Section {index}",
                "purpose": "Develop the document.",
                "target_words": 2000,
                "required_points": [],
                "depends_on": [index - 1] if index > 1 else [],
                "continuity_checks": [],
            }
            for index in range(1, 13)
        ],
    }

    planning_client = FakeModel(
        lambda prompt: json.dumps(plan)
        if prompt.startswith("DOCUMENT_CONTRACT_PLAN_JSON")
        else (_ for _ in ()).throw(AssertionError(prompt[:80]))
    )
    author_clients: list[FakeModel] = []

    def author_factory() -> FakeModel:
        def respond(prompt: str) -> str:
            match = __import__("re").search(
                r"working budget is\s+(\d+)-(\d+)",
                prompt,
            )
            assert match
            calibrated_words = (int(match.group(1)) + int(match.group(2))) // 2
            words = round(calibrated_words / 1.7)
            return "word " * words

        client = FakeModel(respond)
        author_clients.append(client)
        return client

    reviewer_clients: list[FakeModel] = []

    def reviewer_factory() -> FakeModel:
        def respond(prompt: str) -> str:
            if prompt.startswith("REVIEW_MORELONGWRITE_CONTRACT_V2_GLOBAL_JSON"):
                return json.dumps({"reviews": [], "summary": "Pass."})
            marker = "Return one review for every index in "
            indices_text = prompt.split(marker, 1)[1]
            indices = json.loads(
                indices_text[: indices_text.index("]") + 1]
            )
            return json.dumps(
                {
                    "reviews": [
                        {
                            "index": index,
                            "passed": True,
                            "issues": [],
                            "summary": "Pass.",
                            "state_summary": {
                                "entities_and_roles": [],
                                "facts_and_claims": [],
                                "timeline_and_world_state": [],
                                "open_loops_and_dependencies": [],
                            },
                        }
                        for index in indices
                    ]
                }
            )

        client = FakeModel(respond)
        reviewer_clients.append(client)
        return client

    def unexpected_repair_factory():
        raise AssertionError("repair should not be called")

    result = run_morelongwrite_optimized(
        LongBenchWriteTask(
            index=0,
            prompt='Write a 16,000-word document titled "Test Document".',
            task_type="Academic and Technical Writing",
            target_words=16000,
        ),
        "academic",
        planning_client,
        author_factory,
        reviewer_factory,
        unexpected_repair_factory,
        checkpoint_dir=tmp_path,
        planning_temperature=0.2,
        writing_temperature=0.3,
        review_temperature=0.0,
        repair_temperature=0.15,
        planning_max_output_tokens=16384,
        section_max_output_tokens=8192,
        review_max_output_tokens=6144,
        repair_max_output_tokens=8192,
        global_review_max_output_tokens=6144,
        max_plan_attempts=3,
        section_workers=4,
        narrative_section_workers=2,
        review_workers=4,
        repair_workers=4,
        review_batch_size=4,
        max_repair_fraction=0.25,
        recent_context_characters=6000,
        state_excerpt_characters=400,
        minimum_candidate_ratio=0.65,
        maximum_candidate_ratio=1.35,
    )

    assert len(author_clients) == 12
    assert len(result["units"]) == 12
    assert result["final_text"].startswith("# Test Document")
    assert result["metadata"]["repair_stats"]["attempted"] == 0
    assert result["metadata"]["profile"] == MORELONGWRITE_OPTIMIZED_PROFILE
