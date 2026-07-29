from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..io_utils import extract_json_object, read_json, write_json
from ..llm import ChatModel, completion_history
from ..materials import route_materials
from ..prompts import (
    build_direct_prompt,
    build_single_agent_chapter_prompt,
    build_single_agent_length_rewrite_prompt,
    build_single_agent_plan_prompt,
)
from ..schemas import ChapterOutput, CourseOutput, CourseTask, chapter_pre_review_target, to_plain, utc_now
from .common import plan_from_payload


@dataclass
class DirectPromptPipeline:
    llm: ChatModel
    name: str = "direct_prompt"

    def run(self, task: CourseTask) -> CourseOutput:
        routed = route_materials(task, enabled=False)
        markdown = self.llm.complete(
            build_direct_prompt(task, routed),
            system="You generate textbooks. Return Markdown only.",
            temperature=0.35,
        ).strip()
        plan = plan_from_payload({"title": task.topic, "description": task.goal, "purpose": "Direct textbook generation"})
        return CourseOutput(
            task=task,
            pipeline=self.name,
            chapters=[ChapterOutput(plan=plan, content_markdown=markdown)],
            routed_materials=routed,
            metadata={
                "created_at": utc_now(),
                "llm_model": self.llm.model,
                "chapter_min_characters": task.chapter_min_characters,
                "chapter_target_characters": task.chapter_target_characters,
                "pre_review_target_characters": chapter_pre_review_target(task),
                "completion": self.llm.completion_metadata(),
                "author_completion_history": completion_history(self.llm),
            },
        )


@dataclass
class SingleAgentPipeline:
    llm: ChatModel
    name: str = "single_agent"
    checkpoint_dir: str | None = None
    resume_from_checkpoints: bool = False

    def run(self, task: CourseTask) -> CourseOutput:
        routed = route_materials(task, enabled=False)
        checkpoint = self._load_resume_checkpoint(task)
        resumed = checkpoint is not None
        if checkpoint:
            plan_payload, chapters, chapter_completions, length_rewrites = checkpoint
            plan_completion: dict[str, Any] | None = None
        else:
            plan_payload = extract_json_object(
                self.llm.complete(
                    build_single_agent_plan_prompt(task, routed),
                    system="You are a single textbook author. Return valid JSON only.",
                    temperature=0.3,
                    response_format="json_object",
                )
            )
            plan_completion = self.llm.completion_metadata()
            chapters = []
            chapter_completions = []
            length_rewrites = []
        plans = [plan_from_payload(raw) for raw in plan_payload.get("chapters", []) if isinstance(raw, dict)]
        if not plans:
            raise ValueError("single_agent plan did not contain any chapter plans")
        if len(chapters) > len(plans):
            raise ValueError("single_agent checkpoint contains more chapters than its plan")
        if not resumed:
            self._write_checkpoint(
                "plan",
                {
                    "task_id": task.id,
                    "pipeline": self.name,
                    "plan": plan_payload,
                },
            )
        for index in range(len(chapters), len(plans)):
            plan = plans[index]
            markdown = self.llm.complete(
                build_single_agent_chapter_prompt(task, plans, plan, routed, index),
                system="You are a single textbook author. Return Markdown only.",
                temperature=0.3,
            ).strip()
            initial_chars = len(markdown)
            rewrite_completion: dict | None = None
            enrichment_history: list[dict[str, Any]] = []
            pre_review_target = chapter_pre_review_target(task)
            if pre_review_target and initial_chars < pre_review_target:
                candidate = self.llm.complete(
                    build_single_agent_length_rewrite_prompt(task, plan, markdown),
                    system="You are a single textbook author. Return Markdown only.",
                    temperature=0.3,
                ).strip()
                rewrite_completion = self.llm.completion_metadata()
                if len(candidate) > initial_chars:
                    markdown = candidate
                rewrite_record = {
                    "chapter_index": index + 1,
                    "before_chars": initial_chars,
                    "after_chars": len(candidate),
                    "accepted": len(candidate) > initial_chars,
                    "meets_pre_review_target": len(markdown) >= pre_review_target,
                    "completion": rewrite_completion,
                }
                length_rewrites.append(rewrite_record)
                enrichment_history.append(dict(rewrite_record))
            chapters.append(
                ChapterOutput(
                    plan=plan,
                    content_markdown=markdown,
                    length_enrichment_history=enrichment_history,
                )
            )
            chapter_completions.append(rewrite_completion or self.llm.completion_metadata())
            self._write_partial_course(
                task,
                plan_payload,
                chapters,
                chapter_completions,
                length_rewrites,
                stage=f"chapter_{index + 1}_written",
            )
        return CourseOutput(
            task=task,
            pipeline=self.name,
            chapters=chapters,
            routed_materials=routed,
            metadata={
                "created_at": utc_now(),
                "llm_model": self.llm.model,
                "generation_mode": "staged_single_author",
                "planned_chapter_count": len(plans),
                "chapter_min_characters": task.chapter_min_characters,
                "chapter_target_characters": task.chapter_target_characters,
                "pre_review_target_characters": chapter_pre_review_target(task),
                "plan_completion": plan_completion,
                "chapter_completions": chapter_completions,
                "length_rewrites": length_rewrites,
                "resumed_from_checkpoint": resumed,
                "author_completion_history": completion_history(self.llm),
            },
        )

    def _load_resume_checkpoint(
        self, task: CourseTask
    ) -> tuple[dict[str, Any], list[ChapterOutput], list[dict], list[dict]] | None:
        if not self.resume_from_checkpoints or not self.checkpoint_dir:
            return None
        checkpoint_dir = Path(self.checkpoint_dir)
        plan_path = checkpoint_dir / "plan.json"
        if not plan_path.exists():
            return None
        plan_record = read_json(plan_path)
        if plan_record.get("task_id") != task.id or plan_record.get("pipeline") != self.name:
            raise ValueError("single_agent checkpoint belongs to a different task or pipeline")
        plan_payload = plan_record.get("plan")
        if not isinstance(plan_payload, dict):
            raise ValueError("single_agent checkpoint is missing its plan")
        latest_path = checkpoint_dir / "latest.json"
        if not latest_path.exists():
            return plan_payload, [], [], []
        latest = read_json(latest_path)
        if latest.get("task", {}).get("id") != task.id or latest.get("pipeline") != self.name:
            raise ValueError("single_agent latest checkpoint belongs to a different task or pipeline")
        raw_chapters = latest.get("chapters", [])
        if not isinstance(raw_chapters, list):
            raise ValueError("single_agent latest checkpoint chapters must be a list")
        chapters = [
            ChapterOutput(
                plan=plan_from_payload(raw.get("plan", {})),
                content_markdown=str(raw.get("content_markdown", "")),
                length_enrichment_history=[
                    dict(item) for item in raw.get("length_enrichment_history", []) if isinstance(item, dict)
                ],
            )
            for raw in raw_chapters
            if isinstance(raw, dict)
        ]
        if len(chapters) != len(raw_chapters):
            raise ValueError("single_agent latest checkpoint contains an invalid chapter record")
        metadata = latest.get("metadata", {})
        return (
            plan_payload,
            chapters,
            [dict(item) for item in metadata.get("chapter_completions", []) if isinstance(item, dict)],
            [dict(item) for item in metadata.get("length_rewrites", []) if isinstance(item, dict)],
        )

    def _write_checkpoint(self, name: str, payload: dict[str, Any]) -> None:
        if self.checkpoint_dir:
            write_json(Path(self.checkpoint_dir) / f"{name}.json", payload)

    def _write_partial_course(
        self,
        task: CourseTask,
        plan_payload: dict[str, Any],
        chapters: list[ChapterOutput],
        chapter_completions: list[dict],
        length_rewrites: list[dict],
        *,
        stage: str,
    ) -> None:
        if not self.checkpoint_dir:
            return
        partial = CourseOutput(
            task=task,
            pipeline=self.name,
            chapters=chapters,
            routed_materials=route_materials(task, enabled=False),
            metadata={
                "created_at": utc_now(),
                "stage": stage,
                "llm_model": self.llm.model,
                "plan": plan_payload,
                "chapter_completions": chapter_completions,
                "length_rewrites": length_rewrites,
            },
        )
        self._write_checkpoint(stage, to_plain(partial))
        self._write_checkpoint("latest", to_plain(partial))
