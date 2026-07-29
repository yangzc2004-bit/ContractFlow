from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from .io_utils import append_jsonl, read_json, write_json
from .llm import ChatModel


Action = Literal["generate", "extend"]
BudgetState = Literal["normal", "pressure", "survival", "exhausted"]
ValidationStatus = Literal["missing", "short", "valid"]


@dataclass
class GenerationUnit:
    unit_id: str
    order: int
    minimum_length: int
    target_length: int
    dependencies: tuple[str, ...] = ()
    mandatory: bool = True
    priority: int = 0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnitValidation:
    status: ValidationStatus
    measured_length: int
    required_length: int
    issues: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.status == "valid"


class GenerationAdapter(Protocol):
    def action_for(
        self,
        unit: GenerationUnit,
        validation: UnitValidation,
    ) -> Action:
        ...

    def build_prompt(
        self,
        action: Action,
        units: list[GenerationUnit],
        current_texts: dict[str, str],
        completed_texts: dict[str, str],
    ) -> str:
        ...

    def parse_response(
        self,
        action: Action,
        units: list[GenerationUnit],
        raw: str,
    ) -> dict[str, str]:
        ...

    def validate(self, unit: GenerationUnit, text: str) -> UnitValidation:
        ...

    def estimate_output_tokens(
        self,
        unit: GenerationUnit,
        action: Action,
        current_text: str,
    ) -> int:
        ...

    def system_prompt(self, action: Action) -> str:
        ...


@dataclass
class AdaptiveGenerationConfig:
    initial_batch_size: int = 8
    minimum_batch_size: int = 1
    maximum_batch_size: int = 8
    generation_max_output_tokens: int = 4_096
    extension_max_output_tokens: int = 2_048
    max_generation_attempts: int = 3
    max_extension_attempts: int = 3
    total_token_budget: int | None = None
    deadline_seconds: float | None = None
    reserve_fraction: float = 0.12
    output_safety_fraction: float = 0.82
    guarded_coalescing: bool = True

    def __post_init__(self) -> None:
        if self.minimum_batch_size < 1:
            raise ValueError("minimum_batch_size must be at least 1")
        if self.maximum_batch_size < self.minimum_batch_size:
            raise ValueError("maximum_batch_size must be at least minimum_batch_size")
        if not self.minimum_batch_size <= self.initial_batch_size <= self.maximum_batch_size:
            raise ValueError("initial_batch_size must be inside the configured batch range")
        if not 0 <= self.reserve_fraction < 1:
            raise ValueError("reserve_fraction must be between 0 and 1")
        if not 0 < self.output_safety_fraction <= 1:
            raise ValueError("output_safety_fraction must be between 0 and 1")


@dataclass
class DependencyGraph:
    contract_hash: str
    dependencies: dict[str, tuple[str, ...]]
    topological_order: tuple[str, ...]
    critical_depth: dict[str, int]

    def ready(self, unresolved: set[str], completed: set[str]) -> list[str]:
        if not any(self.dependencies.values()):
            return [
                unit_id
                for unit_id in self.topological_order
                if unit_id in unresolved
            ]
        return [
            unit_id
            for unit_id in self.topological_order
            if unit_id in unresolved
            and all(dependency in completed for dependency in self.dependencies[unit_id])
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_hash": self.contract_hash,
            "dependencies": {
                unit_id: list(dependencies)
                for unit_id, dependencies in self.dependencies.items()
            },
            "topological_order": list(self.topological_order),
            "critical_depth": self.critical_depth,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DependencyGraph":
        return cls(
            contract_hash=str(payload["contract_hash"]),
            dependencies={
                str(unit_id): tuple(str(value) for value in dependencies)
                for unit_id, dependencies in payload["dependencies"].items()
            },
            topological_order=tuple(str(value) for value in payload["topological_order"]),
            critical_depth={
                str(unit_id): int(value)
                for unit_id, value in payload["critical_depth"].items()
            },
        )


class DependencyGraphCache:
    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._memory: dict[str, DependencyGraph] = {}

    def compile(self, units: list[GenerationUnit]) -> tuple[DependencyGraph, bool]:
        contract_hash = generation_contract_hash(units)
        if contract_hash in self._memory:
            return self._memory[contract_hash], True
        path = self.cache_dir / f"{contract_hash}.json" if self.cache_dir else None
        if path and path.exists():
            graph = DependencyGraph.from_dict(read_json(path))
            self._memory[contract_hash] = graph
            return graph, True
        graph = compile_dependency_graph(units, contract_hash)
        self._memory[contract_hash] = graph
        if path:
            write_json(path, graph.to_dict())
        return graph, False


def generation_contract_hash(units: list[GenerationUnit]) -> str:
    payload = [
        {
            "unit_id": unit.unit_id,
            "order": unit.order,
            "minimum_length": unit.minimum_length,
            "target_length": unit.target_length,
            "dependencies": list(unit.dependencies),
            "mandatory": unit.mandatory,
            "priority": unit.priority,
            "payload": unit.payload,
        }
        for unit in sorted(units, key=lambda item: (item.order, item.unit_id))
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compile_dependency_graph(
    units: list[GenerationUnit],
    contract_hash: str | None = None,
) -> DependencyGraph:
    by_id = {unit.unit_id: unit for unit in units}
    if len(by_id) != len(units):
        raise ValueError("generation unit IDs must be unique")
    unknown = sorted(
        {
            dependency
            for unit in units
            for dependency in unit.dependencies
            if dependency not in by_id
        }
    )
    if unknown:
        raise ValueError(f"generation units contain unknown dependencies: {unknown}")

    indegree = {unit.unit_id: len(unit.dependencies) for unit in units}
    children: dict[str, list[str]] = {unit.unit_id: [] for unit in units}
    order_key = {unit.unit_id: (unit.order, unit.unit_id) for unit in units}
    for unit in units:
        for dependency in unit.dependencies:
            children[dependency].append(unit.unit_id)
    ready = sorted(
        [unit_id for unit_id, degree in indegree.items() if degree == 0],
        key=order_key.__getitem__,
    )
    topological: list[str] = []
    while ready:
        unit_id = ready.pop(0)
        topological.append(unit_id)
        for child in sorted(children[unit_id], key=order_key.__getitem__):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort(key=order_key.__getitem__)
    if len(topological) != len(units):
        cyclic = sorted(unit_id for unit_id, degree in indegree.items() if degree > 0)
        raise ValueError(f"generation dependency graph contains a cycle: {cyclic}")

    depth: dict[str, int] = {}
    for unit_id in reversed(topological):
        depth[unit_id] = 1 + max((depth[child] for child in children[unit_id]), default=0)
    return DependencyGraph(
        contract_hash=contract_hash or generation_contract_hash(units),
        dependencies={unit.unit_id: tuple(unit.dependencies) for unit in units},
        topological_order=tuple(topological),
        critical_depth=depth,
    )


@dataclass
class BudgetSnapshot:
    state: BudgetState
    spent_tokens: int
    remaining_tokens: int | None
    usable_tokens: int | None
    predicted_remaining_tokens: int
    elapsed_seconds: float
    remaining_seconds: float | None
    prediction_error_rate: float | None


class BudgetController:
    _RANK = {"normal": 0, "pressure": 1, "survival": 2, "exhausted": 3}

    def __init__(self, config: AdaptiveGenerationConfig, *, started: float) -> None:
        self.config = config
        self.started = started
        self.spent_tokens = 0
        self.predicted_tokens = 0
        self.actual_tokens = 0
        self.progressed_units = 0
        self.request_seconds = 0.0

    def record(self, predicted: int, actual: int, elapsed: float, progressed: int) -> None:
        self.spent_tokens += max(0, actual)
        self.predicted_tokens += max(0, predicted)
        self.actual_tokens += max(0, actual)
        self.progressed_units += max(0, progressed)
        self.request_seconds += max(0.0, elapsed)

    def snapshot(self, predicted_remaining: int, remaining_units: int) -> BudgetSnapshot:
        elapsed = time.perf_counter() - self.started
        calibrated = self._calibrate(predicted_remaining)
        token_state: BudgetState = "normal"
        remaining_tokens = None
        usable_tokens = None
        if self.config.total_token_budget is not None:
            remaining_tokens = max(0, self.config.total_token_budget - self.spent_tokens)
            reserve = math.ceil(self.config.total_token_budget * self.config.reserve_fraction)
            usable_tokens = max(0, remaining_tokens - reserve)
            token_state = _ratio_state(usable_tokens, calibrated)

        time_state: BudgetState = "normal"
        remaining_seconds = None
        if self.config.deadline_seconds is not None:
            remaining_seconds = max(0.0, self.config.deadline_seconds - elapsed)
            if self.progressed_units:
                estimate = self.request_seconds / self.progressed_units * remaining_units
                time_state = _ratio_state(remaining_seconds, estimate)

        state = max((token_state, time_state), key=self._RANK.__getitem__)
        error = None
        if self.predicted_tokens:
            error = (self.actual_tokens - self.predicted_tokens) / self.predicted_tokens
        return BudgetSnapshot(
            state=state,
            spent_tokens=self.spent_tokens,
            remaining_tokens=remaining_tokens,
            usable_tokens=usable_tokens,
            predicted_remaining_tokens=calibrated,
            elapsed_seconds=round(elapsed, 4),
            remaining_seconds=remaining_seconds,
            prediction_error_rate=error,
        )

    def _calibrate(self, predicted: int) -> int:
        if not self.predicted_tokens:
            return predicted
        return math.ceil(predicted * max(1.0, self.actual_tokens / self.predicted_tokens))


def _ratio_state(available: float, predicted: float) -> BudgetState:
    if available <= 0:
        return "exhausted"
    if predicted <= 0:
        return "normal"
    ratio = available / predicted
    if ratio < 0.65:
        return "survival"
    if ratio < 1.15:
        return "pressure"
    return "normal"


class AdaptiveChunkController:
    def __init__(self, config: AdaptiveGenerationConfig) -> None:
        self.config = config
        self.current_batch_size = config.initial_batch_size
        self.clean_streak = 0
        self.completion_tokens_per_unit: list[float] = []

    def choose_size(
        self,
        estimates: list[int],
        *,
        max_output_tokens: int,
        budget: BudgetSnapshot,
        all_ready: bool,
    ) -> int:
        if not estimates:
            return 0
        safe_output = max(1, math.floor(max_output_tokens * self.config.output_safety_fraction))
        size = min(
            self.current_batch_size,
            _prefix_capacity(estimates, safe_output),
            len(estimates),
            self.config.maximum_batch_size,
        )
        if budget.usable_tokens is not None:
            if budget.usable_tokens < estimates[0]:
                return 0
            size = min(size, _prefix_capacity(estimates, budget.usable_tokens))
        size = max(self.config.minimum_batch_size, size)
        coalesce = (
            self.config.guarded_coalescing
            and budget.state in {"pressure", "survival"}
            and all_ready
            and len(estimates) <= self.config.maximum_batch_size
            and sum(estimates) <= safe_output
            and (budget.usable_tokens is None or sum(estimates) <= budget.usable_tokens)
        )
        return len(estimates) if coalesce else min(size, len(estimates))

    def observe(
        self,
        *,
        requested: int,
        parsed: int,
        progressed: int,
        finish_reason: str | None,
        completion_tokens: int,
    ) -> None:
        if requested <= 0:
            return
        if completion_tokens > 0:
            self.completion_tokens_per_unit.append(completion_tokens / requested)
            self.completion_tokens_per_unit = self.completion_tokens_per_unit[-20:]
        constrained = (
            finish_reason == "length"
            or parsed / requested < 0.8
            or progressed / requested < 0.7
        )
        if constrained:
            self.current_batch_size = max(
                self.config.minimum_batch_size,
                math.ceil(self.current_batch_size / 2),
            )
            self.clean_streak = 0
        elif parsed / requested >= 0.95 and progressed / requested >= 0.95:
            self.clean_streak += 1
            if self.clean_streak >= 2:
                self.current_batch_size = min(
                    self.config.maximum_batch_size,
                    self.current_batch_size + 1,
                )
                self.clean_streak = 0
        else:
            self.clean_streak = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "current_batch_size": self.current_batch_size,
            "clean_streak": self.clean_streak,
            "mean_completion_tokens_per_unit": (
                statistics.fmean(self.completion_tokens_per_unit)
                if self.completion_tokens_per_unit
                else None
            ),
        }


def _prefix_capacity(estimates: list[int], available: int) -> int:
    used = 0
    count = 0
    for estimate in estimates:
        if count and used + estimate > available:
            break
        used += estimate
        count += 1
        if used >= available:
            break
    return max(1, count)


@dataclass
class AdaptiveGenerationResult:
    texts: dict[str, str]
    failed_units: dict[str, list[str]]
    request_records: list[dict[str, Any]]
    metrics: dict[str, Any]
    dependency_graph: dict[str, Any]
    dependency_cache_hit: bool
    resumed_from_checkpoint: bool


class AdaptiveGenerationEngine:
    CHECKPOINT_VERSION = "adaptive_generation_core_v2"

    def __init__(
        self,
        llm: ChatModel,
        adapter: GenerationAdapter,
        *,
        config: AdaptiveGenerationConfig | None = None,
        graph_cache: DependencyGraphCache | None = None,
    ) -> None:
        self.llm = llm
        self.adapter = adapter
        self.config = config or AdaptiveGenerationConfig()
        self.graph_cache = graph_cache or DependencyGraphCache()

    def run(
        self,
        units: list[GenerationUnit],
        *,
        checkpoint_path: str | Path | None = None,
        metrics_path: str | Path | None = None,
    ) -> AdaptiveGenerationResult:
        started = time.perf_counter()
        graph, cache_hit = self.graph_cache.compile(units)
        by_id = {unit.unit_id: unit for unit in units}
        checkpoint = self._load_checkpoint(checkpoint_path, graph.contract_hash)
        texts = {
            str(key): str(value)
            for key, value in (checkpoint or {}).get("texts", {}).items()
            if str(value).strip()
        }
        generation_attempts = _int_map((checkpoint or {}).get("generation_attempts", {}))
        extension_attempts = _int_map((checkpoint or {}).get("extension_attempts", {}))
        failed = {
            str(key): [str(issue) for issue in value]
            for key, value in (checkpoint or {}).get("failed_units", {}).items()
        }
        records = [
            dict(record)
            for record in (checkpoint or {}).get("request_records", [])
            if isinstance(record, dict)
        ]

        chunker = AdaptiveChunkController(self.config)
        saved = (checkpoint or {}).get("controller", {})
        if isinstance(saved, dict):
            chunker.current_batch_size = min(
                self.config.maximum_batch_size,
                max(
                    self.config.minimum_batch_size,
                    int(saved.get("current_batch_size", chunker.current_batch_size)),
                ),
            )
        prior_elapsed = sum(float(record.get("elapsed_seconds", 0.0)) for record in records)
        budgeter = BudgetController(self.config, started=started - prior_elapsed)
        for record in records:
            budgeter.record(
                int(record.get("predicted_tokens", 0)),
                int(record.get("usage", {}).get("total_tokens", 0)),
                float(record.get("elapsed_seconds", 0.0)),
                len(record.get("progressed_units", [])),
            )

        validations = {
            unit_id: self.adapter.validate(unit, texts.get(unit_id, ""))
            for unit_id, unit in by_id.items()
        }
        while True:
            completed = {unit_id for unit_id, result in validations.items() if result.valid}
            unresolved = set(by_id) - completed - set(failed)
            if not unresolved:
                break
            ready_ids = graph.ready(unresolved, completed)
            if not ready_ids:
                for unit_id in unresolved:
                    failed[unit_id] = ["dependencies did not complete"]
                break

            predicted_remaining = self._estimate_remaining(
                by_id, validations, unresolved, texts
            )
            budget = budgeter.snapshot(predicted_remaining, len(unresolved))
            if budget.state == "exhausted":
                for unit_id in unresolved:
                    failed[unit_id] = ["generation budget exhausted"]
                break
            ready_units = self._prioritize(
                [by_id[unit_id] for unit_id in ready_ids],
                graph,
                budget.state,
            )
            action = self.adapter.action_for(
                ready_units[0],
                validations[ready_units[0].unit_id],
            )
            candidates = [
                unit
                for unit in ready_units
                if self.adapter.action_for(
                    unit,
                    validations[unit.unit_id],
                )
                == action
            ]
            estimates = [
                self.adapter.estimate_output_tokens(
                    unit, action, texts.get(unit.unit_id, "")
                )
                for unit in candidates
            ]
            max_output = (
                self.config.extension_max_output_tokens
                if action == "extend"
                else self.config.generation_max_output_tokens
            )
            batch_size = chunker.choose_size(
                estimates,
                max_output_tokens=max_output,
                budget=budget,
                all_ready=len(candidates) == len(unresolved),
            )
            if batch_size <= 0:
                for unit_id in unresolved:
                    failed[unit_id] = [
                        "remaining budget cannot fund the next generation unit"
                    ]
                break
            batch = candidates[:batch_size]
            predicted = sum(estimates[:batch_size])
            record = self._complete_batch(
                action,
                batch,
                texts,
                completed,
                max_output,
                predicted,
                budget,
            )
            records.append(record)
            if metrics_path:
                append_jsonl(metrics_path, record)
            chunker.observe(
                requested=len(batch),
                parsed=len(record["parsed_units"]),
                progressed=len(record["progressed_units"]),
                finish_reason=record.get("finish_reason"),
                completion_tokens=int(record["usage"].get("completion_tokens", 0)),
            )
            budgeter.record(
                predicted,
                int(record["usage"].get("total_tokens", 0)),
                float(record["elapsed_seconds"]),
                len(record["progressed_units"]),
            )
            for unit in batch:
                unit_id = unit.unit_id
                validation = self.adapter.validate(unit, texts.get(unit_id, ""))
                validations[unit_id] = validation
                if action == "generate":
                    generation_attempts[unit_id] = generation_attempts.get(unit_id, 0) + 1
                    if (
                        validation.status == "missing"
                        and generation_attempts[unit_id] >= self.config.max_generation_attempts
                    ):
                        failed[unit_id] = ["generation response remained missing"]
                else:
                    extension_attempts[unit_id] = extension_attempts.get(unit_id, 0) + 1
                    if (
                        validation.status == "short"
                        and extension_attempts[unit_id] >= self.config.max_extension_attempts
                    ):
                        failed[unit_id] = validation.issues or [
                            "incremental extension did not reach the minimum"
                        ]
            self._write_checkpoint(
                checkpoint_path,
                graph,
                texts,
                generation_attempts,
                extension_attempts,
                failed,
                records,
                chunker,
                budgeter.snapshot(
                    self._estimate_remaining(by_id, validations, unresolved, texts),
                    len(unresolved),
                ),
            )

        for unit_id, validation in validations.items():
            if not validation.valid:
                failed.setdefault(unit_id, validation.issues)
        final_budget = budgeter.snapshot(0, 0)
        metrics = summarize_request_records(
            records,
            final_budget=asdict(final_budget),
            controller=chunker.snapshot(),
        )
        self._write_checkpoint(
            checkpoint_path,
            graph,
            texts,
            generation_attempts,
            extension_attempts,
            failed,
            records,
            chunker,
            final_budget,
        )
        return AdaptiveGenerationResult(
            texts=texts,
            failed_units=failed,
            request_records=records,
            metrics=metrics,
            dependency_graph=graph.to_dict(),
            dependency_cache_hit=cache_hit,
            resumed_from_checkpoint=checkpoint is not None,
        )

    def _complete_batch(
        self,
        action: Action,
        batch: list[GenerationUnit],
        texts: dict[str, str],
        completed: set[str],
        max_output: int,
        predicted: int,
        budget: BudgetSnapshot,
    ) -> dict[str, Any]:
        before = {unit.unit_id: texts.get(unit.unit_id, "") for unit in batch}
        prompt = self.adapter.build_prompt(
            action,
            batch,
            before,
            {unit_id: texts[unit_id] for unit_id in completed if unit_id in texts},
        )
        raw = self.llm.complete(
            prompt,
            system=self.adapter.system_prompt(action),
            temperature=0.2 if action == "generate" else 0.1,
            max_output_tokens=max_output,
        )
        metadata = self.llm.completion_metadata()
        parsed = self.adapter.parse_response(action, batch, raw)
        progressed: list[str] = []
        accepted: list[str] = []
        results: dict[str, dict[str, Any]] = {}
        for unit in batch:
            unit_id = unit.unit_id
            candidate = parsed.get(unit_id, "").strip()
            if candidate:
                updated = (
                    _append_incremental(before[unit_id], candidate)
                    if action == "extend"
                    else candidate
                )
                if updated != before[unit_id]:
                    texts[unit_id] = updated
                    progressed.append(unit_id)
            validation = self.adapter.validate(unit, texts.get(unit_id, ""))
            results[unit_id] = asdict(validation)
            if validation.valid:
                accepted.append(unit_id)
        return {
            "event": "generation_request",
            "action": action,
            "unit_ids": [unit.unit_id for unit in batch],
            "parsed_units": sorted(parsed),
            "progressed_units": progressed,
            "accepted_units": accepted,
            "validation": results,
            "predicted_tokens": predicted,
            "usage": _usage(metadata),
            "finish_reason": metadata.get("finish_reason"),
            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0)),
            "request_max_output_tokens": max_output,
            "batch_size": len(batch),
            "budget_before": asdict(budget),
        }

    def _estimate_remaining(
        self,
        by_id: dict[str, GenerationUnit],
        validations: dict[str, UnitValidation],
        unresolved: set[str],
        texts: dict[str, str],
    ) -> int:
        return sum(
            self.adapter.estimate_output_tokens(
                by_id[unit_id],
                self.adapter.action_for(by_id[unit_id], validations[unit_id]),
                texts.get(unit_id, ""),
            )
            for unit_id in unresolved
        )

    @staticmethod
    def _prioritize(
        units: list[GenerationUnit],
        graph: DependencyGraph,
        state: BudgetState,
    ) -> list[GenerationUnit]:
        if state in {"pressure", "survival"}:
            return sorted(
                units,
                key=lambda unit: (
                    not unit.mandatory,
                    -graph.critical_depth[unit.unit_id],
                    -unit.priority,
                    unit.order,
                ),
            )
        return sorted(units, key=lambda unit: (unit.order, unit.unit_id))

    def _load_checkpoint(
        self,
        path: str | Path | None,
        contract_hash: str,
    ) -> dict[str, Any] | None:
        if path is None or not Path(path).exists():
            return None
        payload = read_json(path)
        if payload.get("version") != self.CHECKPOINT_VERSION:
            raise ValueError("adaptive generation checkpoint has an incompatible version")
        if payload.get("contract_hash") != contract_hash:
            raise ValueError("adaptive generation checkpoint belongs to another contract")
        return payload

    def _write_checkpoint(
        self,
        path: str | Path | None,
        graph: DependencyGraph,
        texts: dict[str, str],
        generation_attempts: dict[str, int],
        extension_attempts: dict[str, int],
        failed: dict[str, list[str]],
        records: list[dict[str, Any]],
        chunker: AdaptiveChunkController,
        budget: BudgetSnapshot,
    ) -> None:
        if path is None:
            return
        write_json(
            path,
            {
                "version": self.CHECKPOINT_VERSION,
                "contract_hash": graph.contract_hash,
                "texts": texts,
                "generation_attempts": generation_attempts,
                "extension_attempts": extension_attempts,
                "failed_units": failed,
                "request_records": records,
                "controller": chunker.snapshot(),
                "budget": asdict(budget),
            },
        )


def _int_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): int(item) for key, item in value.items()}


def _append_incremental(existing: str, continuation: str) -> str:
    existing = existing.strip()
    continuation = continuation.strip()
    if not existing:
        return continuation
    if continuation.startswith(existing):
        return continuation
    return f"{existing}\n\n{continuation}"


def _usage(metadata: dict[str, Any]) -> dict[str, int]:
    usage = metadata.get("usage", {})
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    total = int(usage.get("total_tokens", 0) or 0) or prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def summarize_request_records(
    records: list[dict[str, Any]],
    *,
    final_budget: dict[str, Any],
    controller: dict[str, Any],
) -> dict[str, Any]:
    requested = sum(len(record.get("unit_ids", [])) for record in records)
    parsed = sum(len(record.get("parsed_units", [])) for record in records)
    progressed = sum(len(record.get("progressed_units", [])) for record in records)
    accepted = sum(len(record.get("accepted_units", [])) for record in records)
    prompt_tokens = sum(
        int(record.get("usage", {}).get("prompt_tokens", 0)) for record in records
    )
    completion_tokens = sum(
        int(record.get("usage", {}).get("completion_tokens", 0)) for record in records
    )
    total_tokens = sum(
        int(record.get("usage", {}).get("total_tokens", 0)) for record in records
    )
    batch_sizes = [int(record.get("batch_size", 0)) for record in records]
    return {
        "request_count": len(records),
        "requested_units": requested,
        "parsed_units": parsed,
        "progressed_units": progressed,
        "accepted_units": accepted,
        "parse_success_rate": parsed / requested if requested else 1.0,
        "progress_rate": progressed / requested if requested else 1.0,
        "mean_batch_size": statistics.fmean(batch_sizes) if batch_sizes else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "average_tokens_per_requested_unit": total_tokens / requested if requested else 0.0,
        "generation_requests": sum(record.get("action") == "generate" for record in records),
        "extension_requests": sum(record.get("action") == "extend" for record in records),
        "final_budget": final_budget,
        "controller": controller,
    }
