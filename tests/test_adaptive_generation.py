from __future__ import annotations

import json

import pytest

from learnbyai_research.adaptive_generation import (
    AdaptiveChunkController,
    AdaptiveGenerationConfig,
    BudgetSnapshot,
    DependencyGraphCache,
    GenerationUnit,
    compile_dependency_graph,
)


def unit(
    unit_id: str,
    order: int,
    *,
    dependencies: tuple[str, ...] = (),
) -> GenerationUnit:
    return GenerationUnit(
        unit_id=unit_id,
        order=order,
        minimum_length=20,
        target_length=24,
        dependencies=dependencies,
    )


def normal_budget() -> BudgetSnapshot:
    return BudgetSnapshot(
        state="normal",
        spent_tokens=0,
        remaining_tokens=None,
        usable_tokens=None,
        predicted_remaining_tokens=100,
        elapsed_seconds=0.0,
        remaining_seconds=None,
        prediction_error_rate=None,
    )


def test_dependency_graph_cache_reuses_compiled_artifact(tmp_path) -> None:
    units = [unit("a", 1), unit("b", 2, dependencies=("a",))]
    first_cache = DependencyGraphCache(tmp_path)
    first, first_hit = first_cache.compile(units)
    second, second_hit = DependencyGraphCache(tmp_path).compile(units)

    assert first_hit is False
    assert second_hit is True
    assert second.topological_order == ("a", "b")
    assert second.critical_depth == {"b": 1, "a": 2}
    assert first.contract_hash == second.contract_hash
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_dependency_graph_rejects_cycles() -> None:
    with pytest.raises(ValueError, match="cycle"):
        compile_dependency_graph(
            [
                unit("a", 1, dependencies=("b",)),
                unit("b", 2, dependencies=("a",)),
            ]
        )


def test_chunk_controller_reduces_batch_after_constrained_response() -> None:
    controller = AdaptiveChunkController(
        AdaptiveGenerationConfig(initial_batch_size=8, maximum_batch_size=8)
    )
    controller.observe(
        requested=8,
        parsed=4,
        progressed=4,
        finish_reason="length",
        completion_tokens=4_000,
    )
    assert controller.current_batch_size == 4
    assert controller.choose_size(
        [300] * 8,
        max_output_tokens=4_096,
        budget=normal_budget(),
        all_ready=True,
    ) == 4


def test_guarded_coalescing_only_uses_all_ready_units_when_they_fit() -> None:
    controller = AdaptiveChunkController(
        AdaptiveGenerationConfig(initial_batch_size=2, maximum_batch_size=8)
    )
    pressure = BudgetSnapshot(
        state="pressure",
        spent_tokens=100,
        remaining_tokens=2_000,
        usable_tokens=1_500,
        predicted_remaining_tokens=1_000,
        elapsed_seconds=1.0,
        remaining_seconds=None,
        prediction_error_rate=0.0,
    )
    assert controller.choose_size(
        [200, 200, 200, 200],
        max_output_tokens=4_096,
        budget=pressure,
        all_ready=True,
    ) == 4
    assert controller.choose_size(
        [1_000, 1_000, 1_000, 1_000],
        max_output_tokens=4_096,
        budget=pressure,
        all_ready=True,
    ) == 1


def test_chunk_controller_returns_zero_when_next_unit_exceeds_remaining_budget() -> None:
    controller = AdaptiveChunkController(
        AdaptiveGenerationConfig(initial_batch_size=2, maximum_batch_size=8)
    )
    budget = BudgetSnapshot(
        state="survival",
        spent_tokens=900,
        remaining_tokens=100,
        usable_tokens=100,
        predicted_remaining_tokens=500,
        elapsed_seconds=1.0,
        remaining_seconds=None,
        prediction_error_rate=0.0,
    )
    assert controller.choose_size(
        [200, 200],
        max_output_tokens=4_096,
        budget=budget,
        all_ready=True,
    ) == 0


def test_generation_contract_hash_includes_payload(tmp_path) -> None:
    cache = DependencyGraphCache(tmp_path)
    first = unit("a", 1)
    second = unit("a", 1)
    second.payload = {"required": "different"}
    first_graph, _ = cache.compile([first])
    second_graph, _ = cache.compile([second])
    assert first_graph.contract_hash != second_graph.contract_hash
    assert len(list(tmp_path.glob("*.json"))) == 2
