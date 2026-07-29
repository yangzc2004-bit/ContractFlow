from __future__ import annotations

import ast
import asyncio
import copy
import hashlib
import importlib
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .llm import ChatModel, completion_history
from .longgenbench import LongGenTask, parse_document_blocks, task_record


COGWRITER_SOURCE_REPOSITORY = ""
COGWRITER_SOURCE_COMMIT = "dc3bf084e8733c951172cddd89fa4d7337121fdd"
OFFICIAL_METHOD_FILES = (
    "CogWriter_model/CogWriter.py",
    "CogWriter_model/Agents/PlanningAgent.py",
    "CogWriter_model/Agents/GenerationAgent.py",
    "utils/wordCounter.py",
)


class IncompleteOfficialPlanError(RuntimeError):
    pass


def configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")


@dataclass
class OfficialCogWriterCallAdapter:
    client_factory: Callable[[], ChatModel]
    planning_max_output_tokens: int
    segment_max_output_tokens: int
    temperature: float
    history: list[dict[str, Any]]

    async def complete(self, model: str, prompt: str) -> str:
        client = self.client_factory()
        phase = classify_official_prompt(prompt)
        max_output_tokens = (
            self.planning_max_output_tokens
            if phase == "planning"
            else self.segment_max_output_tokens
        )
        try:
            return await asyncio.to_thread(
                client.complete,
                prompt,
                temperature=self.temperature,
                max_output_tokens=max_output_tokens,
            )
        except Exception:
            # Upstream CogWriter returns an empty response after provider retries fail.
            return ""
        finally:
            for item in completion_history(client):
                enriched = dict(item)
                enriched["official_cogwriter_phase"] = phase
                enriched["official_cogwriter_model_argument"] = model
                self.history.append(enriched)


@dataclass(frozen=True)
class OfficialCogWriterParameterProfile:
    task_type: str
    upstream_type: str
    planning_function: str
    generation_function: str
    planning_string_replacements: tuple[tuple[str, str], ...]
    generation_string_replacements: tuple[tuple[str, str], ...]
    generation_length_from: int
    generation_length_to: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "upstream_type": self.upstream_type,
            "planning_function": self.planning_function,
            "generation_function": self.generation_function,
            "planning_string_replacements": [
                {"from": old, "to": new}
                for old, new in self.planning_string_replacements
            ],
            "generation_string_replacements": [
                {"from": old, "to": new}
                for old, new in self.generation_string_replacements
            ],
            "generation_length_from": self.generation_length_from,
            "generation_length_to": self.generation_length_to,
        }


def classify_official_prompt(prompt: str) -> str:
    normalized = prompt.lower()
    planning_markers = (
        "return your analysis and plan in only",
        "return your analysis and revised plan in only",
        "return your analysis and the final revised plan in only",
    )
    return "planning" if any(marker in normalized for marker in planning_markers) else "segment"


def load_official_cogwriter(
    source_root: str | Path,
    async_call_llm: Callable[[str, str], Any],
    *,
    max_refinements: int | None = None,
    parameter_profile: OfficialCogWriterParameterProfile | None = None,
) -> type:
    configure_utf8_stdio()
    source = Path(source_root).resolve()
    validate_official_source(source)
    if max_refinements is not None and max_refinements < 0:
        raise ValueError("max_refinements must be non-negative")

    llms_package = types.ModuleType("llms")
    llms_package.__path__ = []
    llms_module = types.ModuleType("llms.llms")
    llms_module.async_call_llm = async_call_llm
    llms_package.llms = llms_module
    sys.modules["llms"] = llms_package
    sys.modules["llms.llms"] = llms_module

    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    importlib.invalidate_caches()
    planning_module = importlib.import_module(
        "CogWriter_model.Agents.PlanningAgent"
    )
    generation_module = importlib.import_module(
        "CogWriter_model.Agents.GenerationAgent"
    )
    planning_source = (
        source / "CogWriter_model" / "Agents" / "PlanningAgent.py"
    )
    generation_source = (
        source / "CogWriter_model" / "Agents" / "GenerationAgent.py"
    )
    planning_code = planning_source.read_text(encoding="utf-8")
    generation_code = generation_source.read_text(encoding="utf-8")
    if parameter_profile is not None:
        planning_code = parameterize_official_agent_source(
            planning_code,
            function_name=parameter_profile.planning_function,
            string_replacements=parameter_profile.planning_string_replacements,
        )
        generation_code = parameterize_official_agent_source(
            generation_code,
            function_name=parameter_profile.generation_function,
            string_replacements=parameter_profile.generation_string_replacements,
            numeric_replacements={
                parameter_profile.generation_length_from:
                parameter_profile.generation_length_to
            },
        )
    if max_refinements is not None:
        generation_code = bound_refinement_loops(
            generation_code,
            max_refinements=max_refinements,
        )
    exec(
        compile(planning_code, str(planning_source), "exec"),
        planning_module.__dict__,
    )
    exec(
        compile(generation_code, str(generation_source), "exec"),
        generation_module.__dict__,
    )
    planning_module.async_call_llm = async_call_llm
    generation_module.async_call_llm = async_call_llm
    module = importlib.import_module("CogWriter_model.CogWriter")
    cogwriter_source = source / "CogWriter_model" / "CogWriter.py"
    exec(
        compile(
            cogwriter_source.read_text(encoding="utf-8"),
            str(cogwriter_source),
            "exec",
        ),
        module.__dict__,
    )
    return module.CogWriter


def parameterize_official_agent_source(
    source_code: str,
    *,
    function_name: str,
    string_replacements: tuple[tuple[str, str], ...] = (),
    numeric_replacements: dict[int, int] | None = None,
) -> str:
    tree = ast.parse(source_code)
    transformer = _FunctionParameterTransformer(
        function_name=function_name,
        string_replacements=string_replacements,
        numeric_replacements=numeric_replacements or {},
    )
    transformed = transformer.visit(tree)
    if not transformer.found_function:
        raise RuntimeError(f"official CogWriter function not found: {function_name}")
    missing_strings = [
        old
        for old, _ in string_replacements
        if transformer.string_replacement_counts.get(old, 0) == 0
    ]
    if missing_strings:
        raise RuntimeError(
            f"official CogWriter parameter strings not found in {function_name}: "
            f"{missing_strings}"
        )
    missing_numbers = [
        old
        for old in (numeric_replacements or {})
        if transformer.numeric_replacement_counts.get(old, 0) == 0
    ]
    if missing_numbers:
        raise RuntimeError(
            f"official CogWriter numeric parameters not found in {function_name}: "
            f"{missing_numbers}"
        )
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed)


class _FunctionParameterTransformer(ast.NodeTransformer):
    def __init__(
        self,
        *,
        function_name: str,
        string_replacements: tuple[tuple[str, str], ...],
        numeric_replacements: dict[int, int],
    ) -> None:
        self.function_name = function_name
        self.string_replacements = string_replacements
        self.numeric_replacements = numeric_replacements
        self.inside_target = False
        self.found_function = False
        self.string_replacement_counts: dict[str, int] = {}
        self.numeric_replacement_counts: dict[int, int] = {}

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        prior = self.inside_target
        if node.name == self.function_name:
            self.found_function = True
            self.inside_target = True
        node = self.generic_visit(node)
        self.inside_target = prior
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if not self.inside_target:
            return node
        value = node.value
        if isinstance(value, str):
            transformed = value
            for old, new in self.string_replacements:
                count = transformed.count(old)
                if count:
                    transformed = transformed.replace(old, new)
                    self.string_replacement_counts[old] = (
                        self.string_replacement_counts.get(old, 0) + count
                    )
            if transformed != value:
                return ast.copy_location(ast.Constant(value=transformed), node)
        elif type(value) is int and value in self.numeric_replacements:
            self.numeric_replacement_counts[value] = (
                self.numeric_replacement_counts.get(value, 0) + 1
            )
            return ast.copy_location(
                ast.Constant(value=self.numeric_replacements[value]),
                node,
            )
        return node


def bound_refinement_loops(
    source_code: str,
    *,
    max_refinements: int,
) -> str:
    if max_refinements < 0:
        raise ValueError("max_refinements must be non-negative")
    tree = ast.parse(source_code)
    transformer = _RefinementLoopTransformer(max_refinements)
    transformed = transformer.visit(tree)
    if transformer.transformed_count != 4:
        raise RuntimeError(
            "expected four CogWriter length-refinement loops, "
            f"found {transformer.transformed_count}"
        )
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed)


class _RefinementLoopTransformer(ast.NodeTransformer):
    def __init__(self, max_refinements: int) -> None:
        self.max_refinements = max_refinements
        self.transformed_count = 0

    def visit_While(self, node: ast.While) -> ast.AST:
        self.generic_visit(node)
        if not _is_length_refinement_condition(node.test):
            return node
        self.transformed_count += 1
        condition = copy.deepcopy(node.test)
        return ast.copy_location(
            ast.For(
                target=ast.Name(
                    id=f"_bounded_refinement_attempt_{self.transformed_count}",
                    ctx=ast.Store(),
                ),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[ast.Constant(value=self.max_refinements)],
                    keywords=[],
                ),
                body=[
                    ast.If(
                        test=ast.UnaryOp(op=ast.Not(), operand=condition),
                        body=[ast.Break()],
                        orelse=[],
                    ),
                    *node.body,
                ],
                orelse=node.orelse,
            ),
            node,
        )


def _is_length_refinement_condition(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "word_diff"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Gt)
    )


def official_parameter_profile(task: LongGenTask) -> OfficialCogWriterParameterProfile:
    if task.segment_type == "Day":
        return OfficialCogWriterParameterProfile(
            task_type="Day",
            upstream_type="Week",
            planning_function="async_create_week_plan",
            generation_function="async_generate_week",
            planning_string_replacements=(
                ("weekly plan containing 52 weeks", "daily plan containing 365 days"),
                ("exact week", "exact day"),
                ("which weeks they will be in", "which days they will be in"),
                ("weekly plan for each week", "daily plan for each day"),
                ("May 13th, Week 19", "Day 133 (May 13th)"),
                (
                    "Week 1 (January 1st - January 7th)",
                    "Day 1 (January 1st)",
                ),
                ("for specific weeks", "for specific days"),
                ("events of this week", "events of this day"),
                ("Current weekly plan", "Current daily plan"),
            ),
            generation_string_replacements=(
                (
                    "200-word weekly diary entry for the week of",
                    f"{task.minimum_words}-word daily diary entry for the day of",
                ),
                ("events for this week", "events for this day"),
                ("general diary entry for the week", "general diary entry for the day"),
                (
                    "Your 200-word diary entry here",
                    f"Your {task.minimum_words}-word diary entry here",
                ),
            ),
            generation_length_from=200,
            generation_length_to=task.minimum_words,
        )
    if task.segment_type == "Menu Day":
        return OfficialCogWriterParameterProfile(
            task_type="Menu Day",
            upstream_type="Menu Week",
            planning_function="async_create_menu_plan",
            generation_function="async_generate_menu",
            planning_string_replacements=(
                (
                    "weekly menu plan containing 52 weeks",
                    "daily menu plan containing 365 days",
                ),
                ("exact week", "exact day"),
                ("which weeks they will be in", "which days they will be in"),
                ("weekly menu plan for each week", "daily menu plan for each day"),
                ("05-13, Week 19", "Menu Day 133"),
                (
                    "Menu Week 1 (January 1st - January 7th)",
                    "Menu Day 1",
                ),
                ("for specific weeks", "for specific days"),
                ("Current weekly plan", "Current daily plan"),
                ("weekly plan for each week", "daily plan for each day"),
            ),
            generation_string_replacements=(
                (
                    "200-word weekly menu plan for the week of",
                    f"{task.minimum_words}-word daily menu plan for the day of",
                ),
                ("dishes for this week", "dishes for this day"),
                ("general menu plan for the week", "general menu plan for the day"),
                (
                    "Your 200-word menu plan here",
                    f"Your {task.minimum_words}-word menu plan here",
                ),
            ),
            generation_length_from=200,
            generation_length_to=task.minimum_words,
        )
    if task.segment_type == "Floor":
        return OfficialCogWriterParameterProfile(
            task_type="Floor",
            upstream_type="Floor",
            planning_function="async_create_floor_plan",
            generation_function="async_generate_floor",
            planning_string_replacements=(
                (
                    "skyscraper with 100 floors",
                    f"skyscraper with {task.segment_count} floors",
                ),
            ),
            generation_string_replacements=(
                ("150-word", f"{task.minimum_words}-word"),
            ),
            generation_length_from=150,
            generation_length_to=task.minimum_words,
        )
    if task.segment_type == "Block":
        grid_side = int(round(task.segment_count ** 0.5))
        if grid_side * grid_side != task.segment_count:
            raise ValueError(
                "CogWriter official Block task requires a square grid: "
                f"{task.segment_count}"
            )
        return OfficialCogWriterParameterProfile(
            task_type="Block",
            upstream_type="Block",
            planning_function="async_create_block_plan",
            generation_function="async_generate_block",
            planning_string_replacements=(
                (
                    "10x10 block grid, numbered from 1 to 100",
                    f"{grid_side}x{grid_side} block grid, numbered from 1 to "
                    f"{task.segment_count}",
                ),
                ("Block 10 (0, 1)", f"Block {grid_side + 1} (0, 1)"),
            ),
            generation_string_replacements=(
                ("150-word", f"{task.minimum_words}-word"),
            ),
            generation_length_from=150,
            generation_length_to=task.minimum_words,
        )
    raise ValueError(
        "unsupported task type for official CogWriter parameter adaptation: "
        f"{task.segment_type}"
    )


def official_plan_key(task_type: str) -> str:
    keys = {
        "Day": "weekly_plan",
        "Floor": "floor_plan",
        "Menu Day": "weekly_plan",
        "Block": "block_plan",
    }
    try:
        return keys[task_type]
    except KeyError as error:
        raise ValueError(f"unsupported CogWriter task type: {task_type}") from error


def official_plan_entry_count(processed: dict[str, Any], task_type: str) -> int:
    plan = processed.get(official_plan_key(task_type))
    return len(plan) if isinstance(plan, list) else 0


def run_official_cogwriter(
    task: LongGenTask,
    dataset_item: dict[str, Any],
    client_factory: Callable[[], ChatModel],
    *,
    source_root: str | Path,
    model_name: str,
    maximum_concurrent_requests: int,
    planning_max_output_tokens: int,
    segment_max_output_tokens: int,
    temperature: float,
    max_refinements: int | None = None,
    parameter_profile: OfficialCogWriterParameterProfile | None = None,
    planning_validation_attempts: int = 1,
) -> dict[str, Any]:
    if planning_validation_attempts < 1:
        raise ValueError("planning_validation_attempts must be positive")
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    planning_entry_counts: list[int] = []
    adapter = OfficialCogWriterCallAdapter(
        client_factory=client_factory,
        planning_max_output_tokens=planning_max_output_tokens,
        segment_max_output_tokens=segment_max_output_tokens,
        temperature=temperature,
        history=history,
    )
    cogwriter = load_official_cogwriter(
        source_root,
        adapter.complete,
        max_refinements=max_refinements,
        parameter_profile=parameter_profile,
    )
    example = copy.deepcopy(dataset_item)
    if parameter_profile is not None:
        if parameter_profile.task_type != task.segment_type:
            raise ValueError("CogWriter parameter profile does not match the task type")
        example["type"] = parameter_profile.upstream_type
    cogwriter_globals = cogwriter.async_generate.__globals__
    planning_agent = cogwriter_globals["PlanningAgent"]
    generation_agent = cogwriter_globals["GenerationAgent"]

    async def generate() -> dict[str, Any]:
        semaphore = asyncio.Semaphore(maximum_concurrent_requests)
        retry_count = 0
        while retry_count < 3:
            try:
                for _ in range(planning_validation_attempts):
                    planned = await planning_agent.async_create_hierarchy(
                        model_name,
                        copy.deepcopy(example),
                        semaphore,
                    )
                    entry_count = official_plan_entry_count(planned, task.segment_type)
                    planning_entry_counts.append(entry_count)
                    if entry_count == task.segment_count:
                        return await generation_agent.async_generate(
                            model_name,
                            planned,
                            semaphore,
                        )
                raise IncompleteOfficialPlanError(
                    "official CogWriter planning produced "
                    f"{planning_entry_counts[-1]}/{task.segment_count} entries "
                    f"after {planning_validation_attempts} attempt(s)"
                )
            except IncompleteOfficialPlanError:
                raise
            except Exception:
                retry_count += 1
                if retry_count == 3:
                    raise
                await asyncio.sleep(retry_count)
        raise RuntimeError("official CogWriter exhausted retries")

    processed = asyncio.run(generate())
    processed["type"] = task.segment_type
    final_text = str(processed["final_text"])
    blocks = parse_document_blocks(final_text, task.segment_type)
    source = Path(source_root).resolve()
    return {
        "task": task_record(task),
        "system": "cogwriter_official",
        "final_text": final_text,
        "segments": [
            {"index": index, "text": text}
            for index, text in sorted(blocks.items())
        ],
        "official_output": processed,
        "metadata": {
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "model_calls": len(history),
            "completion_history": history,
            "source_repository": COGWRITER_SOURCE_REPOSITORY,
            "source_commit": COGWRITER_SOURCE_COMMIT,
            "source_root": str(source),
            "official_method_sha256": official_method_hashes(source),
            "adapter_scope": [
                "inject MaaS-backed async_call_llm",
                "apply frozen backbone decoding configuration",
                "limit concurrent provider requests",
                "validate plan cardinality before segment generation",
                "record outputs, timing, and token metadata",
            ],
            "official_method_changes": (
                (
                    []
                    if parameter_profile is None
                    else [
                        "runtime AST parameter substitution for LongGenBench "
                        f"{task.segment_type} count, labels, and word target"
                    ]
                )
                + (
                    []
                    if max_refinements is None
                    else [
                        "runtime AST transform caps each segment length-refinement "
                        f"loop at {max_refinements} attempts"
                    ]
                )
            ),
            "parameter_profile": (
                None if parameter_profile is None else parameter_profile.as_dict()
            ),
            "planning_validation_attempts": planning_validation_attempts,
            "planning_entry_counts": planning_entry_counts,
            "max_refinements": max_refinements,
            "run_variant": (
                "official_unchanged"
                if max_refinements is None
                else "bounded_recovery"
            ),
        },
    }


def validate_official_source(source_root: str | Path) -> None:
    source = Path(source_root)
    missing = [path for path in OFFICIAL_METHOD_FILES if not (source / path).is_file()]
    if missing:
        raise RuntimeError(f"official CogWriter source is incomplete: {missing}")


def official_method_hashes(source_root: str | Path) -> dict[str, str]:
    source = Path(source_root)
    return {
        path: hashlib.sha256((source / path).read_bytes()).hexdigest()
        for path in OFFICIAL_METHOD_FILES
    }
