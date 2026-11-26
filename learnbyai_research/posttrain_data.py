from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

from .contractflow_v2 import (
    DocumentContract,
    FlowConfig,
    render_markdown_document,
    run_contractflow,
    word_count,
)
from .io_utils import extract_json_object
from .llm import ChatModel


BucketKind = Literal["open_a1", "open_a2", "structured_b"]
LanguageCode = Literal["en", "zh"]
LengthTier = Literal["4K", "8K", "16K", "32K"]


@dataclass(frozen=True)
class DomainSpec:
    domain_id: str
    bucket: Literal["open_A", "structured_B"]
    name_en: str
    name_zh: str
    unit_label_en: str
    unit_label_zh: str
    description_en: str
    description_zh: str
    supports_date_offset: bool = False


DOMAINS: tuple[DomainSpec, ...] = (
    # Bucket A: Open-Ended Multi-Chapter Constrained Long Writing (6 domains, 50%)
    DomainSpec(
        domain_id="A1_tech_handbook",
        bucket="open_A",
        name_en="Technical Handbook & Systems Engineering Guide",
        name_zh="技术教程与系统开发手册",
        unit_label_en="Chapter",
        unit_label_zh="章",
        description_en="Practical systems engineering manuals, architecture playbooks, and implementation tutorials.",
        description_zh="面向工程师的实战开发手册、系统架构指南与深度技术教程。",
    ),
    DomainSpec(
        domain_id="A2_industry_report",
        bucket="open_A",
        name_en="Industry Research & Business Whitepaper",
        name_zh="行业研究与商业白皮书",
        unit_label_en="Section",
        unit_label_zh="部分",
        description_en="Quantitative market analyses, supply-chain deep dives, and executive business proposals.",
        description_zh="面向决策层与投资人的行业深度调研、供应链分析与商业计划白皮书。",
    ),
    DomainSpec(
        domain_id="A3_long_fiction",
        bucket="open_A",
        name_en="Long-Form Fiction & Serialized Narrative",
        name_zh="长篇叙事与连载故事",
        unit_label_en="Chapter",
        unit_label_zh="章",
        description_en="Coherent multi-chapter science fiction, mystery, or character-driven literary novellas.",
        description_zh="结构严谨的长篇科幻、悬疑推理或人物群像中长篇连载小说。",
    ),
    DomainSpec(
        domain_id="A4_science_monograph",
        bucket="open_A",
        name_en="Popular Science & Academic Monograph",
        name_zh="深度科普与学科专著",
        unit_label_en="Chapter",
        unit_label_zh="章",
        description_en="Rigorous yet accessible scientific expositions, conceptual histories, and research monographs.",
        description_zh="兼具学术严谨性与可读性的深度科普长文、科学思想史与学科专著。",
    ),
    DomainSpec(
        domain_id="A5_history_policy",
        bucket="open_A",
        name_en="Historical & Public Policy Case Study",
        name_zh="历史案例与政策剖析",
        unit_label_en="Chapter",
        unit_label_zh="章",
        description_en="Comparative historical studies, economic crisis retrospectives, and public policy evaluations.",
        description_zh="多维中立的历史比较研究、重大公共工程成败复盘与经济政策剖析。",
    ),
    DomainSpec(
        domain_id="A6_legal_contract",
        bucket="open_A",
        name_en="Legal, Contract & Policy Drafting",
        name_zh="法律、合同与政策条文起草",
        unit_label_en="Article",
        unit_label_zh="条",
        description_en="Formal enterprise agreements, regulatory compliance frameworks, and municipal or corporate policies.",
        description_zh="严谨的商业合同协议、数据合规监管条例、行业治理规范与政策条文起草。",
    ),
    # Bucket B: Multi-Domain Structured Long-Horizon Planning (6 domains, 50%; zero LongGenBench overlap)
    DomainSpec(
        domain_id="B1_codebase_design",
        bucket="structured_B",
        name_en="Multi-Module Codebase Design Specification",
        name_zh="代码库多模块设计文档",
        unit_label_en="Module",
        unit_label_zh="模块",
        description_en="Detailed multi-module software architecture specifications with APIs, invariants, and runbooks.",
        description_zh="大型软件代码库的分模块设计规格书，含接口契约、错误码、性能预算与容灾预案。",
    ),
    DomainSpec(
        domain_id="B2_course_syllabus",
        bucket="structured_B",
        name_en="University Course & Bootcamp Syllabus",
        name_zh="大学课程与训练营教学大纲",
        unit_label_en="Lecture",
        unit_label_zh="讲",
        description_en="Multi-lecture university courses or technical bootcamps with readings, labs, and assessments.",
        description_zh="系统化的高校课程或工程训练营分讲大纲，包含理论内容、实验作业与阶段考核。",
    ),
    DomainSpec(
        domain_id="B3_museum_guide",
        bucket="structured_B",
        name_en="Museum, Science Center & Botanical Garden Guide",
        name_zh="博物馆、科技馆与植物园导览规划",
        unit_label_en="Gallery",
        unit_label_zh="展厅",
        description_en="Curatorial walkthroughs across numbered exhibition galleries, interactive zones, or botanical pavilions.",
        description_zh="大型博物馆、科技馆或植物园的编号展厅策展与动线导览规划。",
    ),
    DomainSpec(
        domain_id="B4_expedition_log",
        bucket="structured_B",
        name_en="Polar, Ocean & Space Expedition Log",
        name_zh="极地、海洋与太空科考任务日志",
        unit_label_en="Log",
        unit_label_zh="日志",
        description_en="Chronological scientific field expedition or space-mission operational logs with calendar milestones.",
        description_zh="按时间顺序编排的极地、深海或深空科考任务日志，支持起始日期与里程碑换算。",
        supports_date_offset=True,
    ),
    DomainSpec(
        domain_id="B5_ops_shifts",
        bucket="structured_B",
        name_en="Medical, Energy & Transit Operations Handbook",
        name_zh="医疗、能源与交通设施运维排班手册",
        unit_label_en="Shift",
        unit_label_zh="班次",
        description_en="Sequential shift handover protocols, inspection cycles, and safety drills for critical infrastructure.",
        description_zh="关键基础设施（电网、轨道交通、医院、水电站）的连续班次巡检与应急演练手册。",
    ),
    DomainSpec(
        domain_id="B6_episode_stages",
        bucket="structured_B",
        name_en="TV Series Episode Outline & Game Stage Design",
        name_zh="影视剧集分集大纲与游戏关卡设计",
        unit_label_en="Episode",
        unit_label_zh="集",
        description_en="Long-running serialized TV episode breakdowns or multi-stage narrative game level blueprints.",
        description_zh="多集长剧集分集剧本大纲或大型叙事游戏的分关卡系统设计蓝图。",
    ),
)

DOMAIN_BY_ID: dict[str, DomainSpec] = {spec.domain_id: spec for spec in DOMAINS}


# Calibrated length tiers: 4K (15%) / 8K (25%) / 16K (30%) / 32K (30%)
LENGTH_TIER_WEIGHTS: dict[LengthTier, float] = {
    "4K": 0.15,
    "8K": 0.25,
    "16K": 0.30,
    "32K": 0.30,
}

LENGTH_TIER_SPECS: dict[LengthTier, dict[str, Any]] = {
    "4K": {
        "target_words_en": 2_800,
        "target_chars_zh": 4_000,
        "open_units": (6, 8),
        "structured_units": (18, 26),
    },
    "8K": {
        "target_words_en": 5_600,
        "target_chars_zh": 8_000,
        "open_units": (8, 12),
        "structured_units": (32, 48),
    },
    "16K": {
        "target_words_en": 11_500,
        "target_chars_zh": 16_000,
        "open_units": (12, 16),
        "structured_units": (64, 96),
    },
    "32K": {
        "target_words_en": 22_000,
        "target_chars_zh": 26_000,
        "open_units": (16, 22),
        "structured_units": (110, 160),
    },
}


def _unit_tag_template(label: str, language: LanguageCode, idx_placeholder: str = "i") -> str:
    if language == "zh":
        # Measure-word labels (讲/集/章/条) read naturally as "第 i 讲"; noun labels (模块/展厅/日志/班次) as "模块 i"
        if len(label) == 1:
            return f"第 {idx_placeholder} {label}"
        return f"{label} {idx_placeholder}"
    return f"{label} {idx_placeholder}"


# 5 diverse output formats for Bucket B (strictly NO benchmark-specific "#*#" delimiter)
STRUCTURED_FORMATS: tuple[dict[str, str], ...] = (
    {
        "format_id": "markdown_h2",
        "en": "Format each entry with a level-2 Markdown heading `## {tag}:` followed by its body paragraph(s).",
        "zh": "每个单元使用二级 Markdown 标题 `## {tag}：` 开头，随后换行书写正文。",
    },
    {
        "format_id": "markdown_h3",
        "en": "Format each entry with a level-3 Markdown heading `### {tag}` followed by its body paragraph(s).",
        "zh": "每个单元使用三级 Markdown 标题 `### {tag}` 开头，随后换行书写正文。",
    },
    {
        "format_id": "plain_colon",
        "en": "Start every entry on a new paragraph with `{tag}:` followed by the entry content.",
        "zh": "每个单元另起一段，以 `{tag}：` 开头并紧接正文内容。",
    },
    {
        "format_id": "hr_dash",
        "en": "Separate consecutive entries with a `---` horizontal rule line, heading each entry as `**{tag}**`.",
        "zh": "相邻单元之间用单独一行 `---` 分隔，每段以 `**{tag}**` 作为标题开头。",
    },
    {
        "format_id": "hr_equals",
        "en": "Separate consecutive entries with `=== {tag} ===` on its own line before the prose.",
        "zh": "每个单元前用单独一行 `=== {tag} ===` 分隔，再换行书写正文。",
    },
)

WRITING_STYLES_EN: tuple[str, ...] = (
    "formal engineering or policy specification request",
    "senior editor commissioning a comprehensive long-form piece",
    "busy practitioner delegating a detailed deliverable to a specialist",
    "professor or lead architect assigning a rigorous structured document",
    "direct, concise multi-requirement user prompt",
)

WRITING_STYLES_ZH: tuple[str, ...] = (
    "正式的甲方需求说明书口吻",
    "专业主编约稿的详细大纲委托口吻",
    "资深架构师或法务总监布置交付文档的口吻",
    "高校教授或项目负责人布置严谨长文任务的口吻",
    "直接、干练、条理清晰的用户长文本写作指令",
)

# Forbidden benchmark fingerprints (must NEVER appear in synthesized training prompts)
FORBIDDEN_PROMPT_PATTERNS: tuple[str, ...] = (
    "#*#",
    "*** finished ***",
    "menu week",
    "menu day",
    "checks_once",
    "checks_range",
    "checks_periodic",
    "longgenbench",
    "longbench-write",
)


@dataclass
class PromptSpec:
    spec_id: str
    bucket: BucketKind
    domain_id: str
    language: LanguageCode
    length_tier: LengthTier
    target_length: int
    unit_count: int
    unit_label: str
    min_unit_words: int
    max_unit_words: int
    output_format_id: str
    output_format_instruction: str
    style_hint: str
    start_date: str | None = None
    required_topic_count: int = 0
    rule_slots: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _exact_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    keys = list(weights.keys())
    raw = {k: total * weights[k] for k in keys}
    floored = {k: int(math.floor(raw[k])) for k in keys}
    rem = total - sum(floored.values())
    order = sorted(keys, key=lambda k: (raw[k] - floored[k], weights[k]), reverse=True)
    for k in order[:rem]:
        floored[k] += 1
    return floored


def _sample_rule_slots(
    rng: random.Random,
    *,
    bucket: BucketKind,
    unit_count: int,
    length_tier: LengthTier,
    supports_date_offset: bool,
) -> tuple[list[dict[str, Any]], str | None, int]:
    """Generate mathematically valid, non-overlapping constraint slots."""
    if bucket == "open_a1":
        topic_count = {"4K": 4, "8K": 5, "16K": 6, "32K": 7}[length_tier]
        slots = [
            {"scope": "all", "slot_id": "G1"},
            {"scope": "all", "slot_id": "G2"},
        ]
        return slots, None, topic_count

    slots: list[dict[str, Any]] = [{"scope": "all", "slot_id": "G1"}]
    if rng.random() < 0.7:
        slots.append({"scope": "all", "slot_id": "G2"})

    start_date_str: str | None = None
    start_dt: date | None = None
    if bucket == "structured_b" and supports_date_offset and rng.random() < 0.65:
        month = rng.randint(1, 9)
        day = rng.randint(1, 20)
        start_dt = date(2025, month, day)
        start_date_str = start_dt.isoformat()

    # Single-unit anchors: guaranteed spread across early, mid, and tail (>= 75%)
    n_single = (
        {"4K": 2, "8K": 3, "16K": 3, "32K": 4}[length_tier]
        if bucket == "open_a2"
        else {"4K": 2, "8K": 3, "16K": 4, "32K": 5}[length_tier]
    )
    used_singles: set[int] = set()
    tail_low = max(2, int(math.ceil(unit_count * 0.75)))
    tail_idx = rng.randint(tail_low, unit_count)
    used_singles.add(tail_idx)
    while len(used_singles) < n_single and len(used_singles) < unit_count:
        used_singles.add(rng.randint(1, unit_count))

    for i, idx in enumerate(sorted(used_singles), start=1):
        slot: dict[str, Any] = {"scope": "single", "slot_id": f"S{i}", "index": idx}
        if start_dt is not None and i == 1:
            target_dt = start_dt + timedelta(days=idx - 1)
            slot["date_hint"] = target_dt.isoformat()
        slots.append(slot)

    # Non-overlapping contiguous ranges
    n_range = 1 if (bucket == "open_a2" or length_tier in {"4K", "8K"}) else 2
    occupied_ranges: list[tuple[int, int]] = []
    for r_i in range(1, n_range + 1):
        span_min = 2 if bucket == "open_a2" else 3
        span_max = min(max(span_min, unit_count // 4), 5 if bucket == "open_a2" else 12)
        for _ in range(30):
            span = rng.randint(span_min, span_max)
            start = rng.randint(1, max(1, unit_count - span + 1))
            end = min(unit_count, start + span - 1)
            if end <= start:
                continue
            if all(end < a or start > b for a, b in occupied_ranges):
                occupied_ranges.append((start, end))
                slots.append(
                    {"scope": "range", "slot_id": f"R{r_i}", "start": start, "end": end}
                )
                break

    # Periodic / stride rules (with non-1 starting offset to test real arithmetic)
    n_every = 1 if (bucket == "open_a2" or length_tier in {"4K", "8K"}) else 2
    used_steps: set[int] = set()
    for p_i in range(1, n_every + 1):
        if bucket == "open_a2":
            step = rng.choice([2, 3, 4])
            start = rng.randint(1, min(step + 1, max(1, unit_count - step)))
            end = None
        else:
            candidate_steps = [s for s in (3, 4, 5, 6, 7, 8, 9, 10) if s not in used_steps and s * 3 <= unit_count]
            if not candidate_steps:
                break
            step = rng.choice(candidate_steps)
            used_steps.add(step)
            start = rng.randint(2, min(step + 3, max(2, unit_count - step * 2)))
            bounded = rng.random() < 0.35 and (start + step * 3 <= unit_count)
            end = rng.randint(start + step * 2, unit_count) if bounded else None
        slots.append(
            {
                "scope": "every",
                "slot_id": f"P{p_i}",
                "start": start,
                "step": step,
                "end": end,
            }
        )

    return slots, start_date_str, 0


def sample_prompt_specs(
    *,
    count: int = 1200,
    canary: bool = False,
    seed: int = 42,
    en_ratio: float = 0.70,
) -> list[PromptSpec]:
    """Sample stratified, zero-leakage prompt specs across all 12 domains.

    When canary=True, produces exactly 24 specs (2 per domain across all 12 domains),
    balancing EN/ZH, A-1/A-2, and 4K/8K/16K/32K for rapid human inspection.
    """
    rng = random.Random(seed)
    tiers: list[LengthTier] = ["4K", "8K", "16K", "32K"]

    if canary:
        specs: list[PromptSpec] = []
        canary_tier_pairs: list[tuple[LengthTier, LengthTier]] = [
            ("4K", "16K"),
            ("8K", "32K"),
            ("16K", "32K"),
            ("4K", "32K"),
            ("8K", "16K"),
            ("16K", "32K"),
        ]
        idx_counter = 0
        for d_idx, domain in enumerate(DOMAINS):
            pair = canary_tier_pairs[d_idx % len(canary_tier_pairs)]
            # 2 samples per domain: 1 EN + 1 ZH (exact 50/50 in canary so both languages are thoroughly inspected per domain)
            for sub_i, (lang, tier) in enumerate(zip(("en", "zh"), pair)):
                idx_counter += 1
                if domain.bucket == "open_A":
                    bucket_kind: BucketKind = "open_a1" if (d_idx + sub_i) % 2 == 0 else "open_a2"
                else:
                    bucket_kind = "structured_b"
                specs.append(
                    _build_one_spec(
                        rng,
                        spec_id=f"canary_{idx_counter:03d}_{domain.domain_id}_{lang}_{tier.lower()}",
                        domain=domain,
                        bucket=bucket_kind,
                        language=lang,  # type: ignore[arg-type]
                        length_tier=tier,
                    )
                )
        return specs

    if count <= 0:
        raise ValueError("count must be positive")

    # Allocate evenly across all 12 domains (50% Bucket A, 50% Bucket B)
    base_per_domain = count // len(DOMAINS)
    rem = count % len(DOMAINS)
    domain_counts = [base_per_domain + (1 if i < rem else 0) for i in range(len(DOMAINS))]

    # Global exact length-tier pool (15 / 25 / 30 / 30) and language pool (70 / 30)
    tier_counts = _exact_counts(count, LENGTH_TIER_WEIGHTS)
    tier_pool: list[LengthTier] = []
    for t in tiers:
        tier_pool.extend([t] * tier_counts[t])
    rng.shuffle(tier_pool)

    specs = []
    tier_cursor = 0
    global_idx = 0
    for d_idx, (domain, d_count) in enumerate(zip(DOMAINS, domain_counts)):
        n_en = int(round(d_count * en_ratio))
        langs: list[LanguageCode] = ["en"] * n_en + ["zh"] * (d_count - n_en)
        rng.shuffle(langs)
        for j in range(d_count):
            global_idx += 1
            lang = langs[j]
            tier = tier_pool[tier_cursor]
            tier_cursor += 1
            if domain.bucket == "open_A":
                bucket_kind = "open_a1" if j % 2 == 0 else "open_a2"
            else:
                bucket_kind = "structured_b"
            specs.append(
                _build_one_spec(
                    rng,
                    spec_id=f"pt_{global_idx:05d}_{domain.domain_id}_{lang}_{tier.lower()}",
                    domain=domain,
                    bucket=bucket_kind,
                    language=lang,
                    length_tier=tier,
                )
            )
    return specs


def _build_one_spec(
    rng: random.Random,
    *,
    spec_id: str,
    domain: DomainSpec,
    bucket: BucketKind,
    language: LanguageCode,
    length_tier: LengthTier,
) -> PromptSpec:
    tier_cfg = LENGTH_TIER_SPECS[length_tier]
    target_len = (
        int(tier_cfg["target_words_en"])
        if language == "en"
        else int(tier_cfg["target_chars_zh"])
    )
    if domain.bucket == "open_A":
        u_lo, u_hi = tier_cfg["open_units"]
    else:
        u_lo, u_hi = tier_cfg["structured_units"]
    unit_count = rng.randint(u_lo, u_hi)
    unit_label = domain.unit_label_en if language == "en" else domain.unit_label_zh

    avg_unit = max(80, target_len // max(1, unit_count))
    min_unit_words = max(60, int(round(avg_unit * 0.88)))
    max_unit_words = max(min_unit_words + 20, int(round(avg_unit * 1.15)))

    if domain.bucket == "open_A":
        fmt_id = "markdown_open"
        fmt_inst = (
            "Write as a cohesive Markdown document using `##` chapter headings."
            if language == "en"
            else "以流畅完整的 Markdown 长文形式输出，使用 `##` 二级标题划分各章。"
        )
    else:
        fmt = rng.choice(STRUCTURED_FORMATS)
        fmt_id = fmt["format_id"]
        fmt_inst = fmt[language].format(tag=_unit_tag_template(unit_label, language))

    style_hint = rng.choice(WRITING_STYLES_EN if language == "en" else WRITING_STYLES_ZH)
    slots, start_date_str, topic_count = _sample_rule_slots(
        rng,
        bucket=bucket,
        unit_count=unit_count,
        length_tier=length_tier,
        supports_date_offset=domain.supports_date_offset,
    )
    return PromptSpec(
        spec_id=spec_id,
        bucket=bucket,
        domain_id=domain.domain_id,
        language=language,
        length_tier=length_tier,
        target_length=target_len,
        unit_count=unit_count,
        unit_label=unit_label,
        min_unit_words=min_unit_words,
        max_unit_words=max_unit_words,
        output_format_id=fmt_id,
        output_format_instruction=fmt_inst,
        style_hint=style_hint,
        start_date=start_date_str,
        required_topic_count=topic_count,
        rule_slots=slots,
    )


# ---------------------------------------------------------------------------
# Layer 2: Two-Step LLM Topic Seed Generation & Natural Prompt Polishing
# ---------------------------------------------------------------------------

def build_topic_seed_prompt(
    domain: DomainSpec,
    language: LanguageCode,
    count: int,
) -> str:
    if language == "zh":
        return f"""请为领域「{domain.name_zh}」（{domain.description_zh}）生成 {count} 个高度具体、彼此完全不重复的细分写作主题种子，用于构建高质量长文写作提示词库。

严格要求：
1. 严禁出现以下题材：个人日记、餐厅每周/每日菜单、摩天大楼楼层规划、城市街区网格布局，禁止出现 `#*#` 分隔符。
2. `title`、`setting`、`concepts`、`events` 中一律**不要写死任何总集数、总天数、总周数、总模块数或具体序号**（例如禁止写“12集短剧”“42天航行”“第3周”“第29章”“3号展厅”），确保每条事件都是位置无关的，后续可自由排进任意编号单元。

仅返回合法 JSON 对象：
{{
  "seeds": [
    {{
      "title": "具体主题名称（10-25字，不含固定集数/天数/模块数）",
      "setting": "具体的背景设定、目标读者或使用场景（1-2句话，不含固定总单元数）",
      "concepts": ["6-8个该主题下高度具体的专业概念、子议题、条款或关键要素"],
      "events": ["6-8个位置无关、适合排进任意特定单元的具体事件、演练、案例、伏笔或检查项（不带第X周/第X集/第X天前缀）"]
    }}
  ]
}}
"""
    return f"""Generate {count} highly specific, mutually distinct scenario seeds for the domain "{domain.name_en}" ({domain.description_en}) to build a diverse long-form writing prompt corpus.

Strict rules:
1. Avoid personal diaries, restaurant menus, skyscraper floor plans, city block grids, or `#*#` delimiters.
2. Do NOT hardcode total counts or fixed unit numbers (e.g., avoid "12-week", "42-day", "three modules", "Week 3", "Day 12", "Chapter 29") anywhere in `title`, `setting`, `concepts`, or `events`. Keep every event index-free so it can be placed cleanly at any unit index later.

Return valid JSON only:
{{
  "seeds": [
    {{
      "title": "specific concrete topic title without hardcoded total counts",
      "setting": "1-2 sentences specifying audience, context, or operational setting without fixed unit counts",
      "concepts": ["6-8 concrete domain concepts, clauses, subsystems, or analytical themes"],
      "events": ["6-8 index-free localized events, milestones, drills, plot beats, or case studies"]
    }}
  ]
}}
"""


def generate_topic_seeds(
    llm: ChatModel,
    *,
    domain: DomainSpec,
    language: LanguageCode,
    count: int = 12,
    batch_size: int = 12,
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    remaining = count
    attempts = 0
    while remaining > 0 and attempts < max(4, math.ceil(count / max(1, batch_size)) + 3):
        attempts += 1
        ask = min(batch_size, remaining)
        prompt = build_topic_seed_prompt(domain, language, ask)
        if seen_titles:
            avoid = ", ".join(list(seen_titles)[-20:])
            prompt += f"\nDo NOT repeat or closely paraphrase these existing titles: {avoid}\n"
        raw = llm.complete(
            prompt,
            system="You generate diverse, realistic, non-repeating writing scenario seeds. Return valid JSON only.",
            temperature=0.8,
            response_format="json_object",
            max_output_tokens=4_096,
        )
        payload = extract_json_object(raw)
        raw_seeds = payload.get("seeds", [])
        if not isinstance(raw_seeds, list):
            continue
        for item in raw_seeds:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            setting = str(item.get("setting", "")).strip()
            concepts = [str(x).strip() for x in item.get("concepts", []) if str(x).strip()]
            events = [str(x).strip() for x in item.get("events", []) if str(x).strip()]
            if not title or title.lower() in seen_titles or len(concepts) < 2 or len(events) < 2:
                continue
            seen_titles.add(title.lower())
            collected.append(
                {
                    "title": title,
                    "setting": setting,
                    "concepts": concepts,
                    "events": events,
                }
            )
            if len(collected) >= count:
                break
        remaining = count - len(collected)
    return collected[:count]


def build_natural_prompt_synthesis_request(
    spec: PromptSpec,
    seed_item: dict[str, Any],
    *,
    retry_note: str = "",
) -> str:
    domain = DOMAIN_BY_ID[spec.domain_id]
    retry_block = f"\nCORRECTION_REQUIRED:\n{retry_note}\n" if retry_note else ""
    anonymous_rules = [
        {k: v for k, v in slot.items() if k != "slot_id"}
        for slot in spec.rule_slots
    ]
    spec_payload = {
        "bucket": spec.bucket,
        "domain": domain.name_zh if spec.language == "zh" else domain.name_en,
        "language": spec.language,
        "target_length": (
            f"约 {spec.target_length} 汉字"
            if spec.language == "zh"
            else f"approximately {spec.target_length} words"
        ),
        "unit_count": None if spec.bucket == "open_a1" else spec.unit_count,
        "unit_label": spec.unit_label,
        "per_unit_length_guidance": (
            None
            if spec.bucket == "open_a1"
            else (
                f"每{spec.unit_label}约 {spec.min_unit_words}–{spec.max_unit_words} 字"
                if spec.language == "zh"
                else f"{spec.min_unit_words} to {spec.max_unit_words} words per {spec.unit_label}"
            )
        ),
        "output_format": spec.output_format_instruction,
        "required_deep_dive_topics": spec.required_topic_count if spec.bucket == "open_a1" else None,
        "start_date": spec.start_date,
        "required_constraints": anonymous_rules,
        "writing_persona": spec.style_hint,
    }
    if spec.language == "zh":
        bucket_rule = (
            "- 本题属于【纯开放议题长文】：绝对不要硬性规定必须分成多少章/多少节，只明确目标总字数、全局风格要求，以及必须深入展开的若干具体核心议题/案例，让作者自行规划章节架构。"
            if spec.bucket == "open_a1"
            else "- 本题必须清晰写出总单元数、每单元字数要求、指定的输出格式，以及 `required_constraints` 中每一条约束（所有编号、起止区间、偏移起点、周期步长和日期必须 100% 原样保留，绝不能改动或遗漏任何数字）。"
        )
        return f"""请根据下面的结构化规格（SPEC）和具体场景种子（SEED），编写一条自然、地道、像真实人类提出的中文长文本写作 Prompt。

要求：
1. 采用指定口吻：{spec.style_hint}。绝对不要出现“约束槽位”“SPEC”“SEED”或任何内部代号（严禁出现 G1、G2、S1、S2、R1、P1 等字母数字编号标签）。
2. 必须把 `required_constraints` 中的每一条约束都用 SEED 里的**具体真实业务内容/专业概念/事件**写实写透，绝不能留下空洞占位符（例如禁止写“落实某项总体要求”“做重点设计”而不说明具体要写什么）。
3. 严禁引入与 `unit_count` 矛盾的其他总集数/总天数/超界编号。
4. {bucket_rule}
5. 严禁出现 `#*#`、`Menu Week`、`Menu Day`、`*** finished ***` 等任何测试集特有标记。
6. 只输出最终可直接发给大模型的用户 Prompt 正文，不要加引号、解释或前后缀说明。
{retry_block}
SEED:
{json.dumps(seed_item, ensure_ascii=False)}

SPEC:
{json.dumps(spec_payload, ensure_ascii=False)}
"""

    bucket_rule_en = (
        "- This is an OPEN-ENDED topic-driven long-writing prompt: do NOT force a fixed chapter count. State the target total word count, global style/audience requirements, and concrete deep-dive topics/cases to cover thoroughly while letting the writer organize the chapters naturally."
        if spec.bucket == "open_a1"
        else "- Explicitly state the total unit count, per-unit word guidance, requested output format, and EVERY constraint in `required_constraints` using the EXACT numbers, inclusive ranges, starting offsets, periodic intervals, and dates (never alter or omit any number)."
    )
    return f"""Write one realistic, natural English user prompt for a long-form writing task based on the SEED and SPEC below.

Requirements:
1. Adopt the requested user persona: {spec.style_hint}. Never mention internal terms like "SPEC", "SEED", "slot", or synthetic label codes (never output codes like G1, G2, S1, R1, P1).
2. Ground EVERY constraint in `required_constraints` with concrete, domain-authentic details drawn from SEED — never leave generic placeholders.
3. Never introduce conflicting total counts or out-of-range indices that contradict `unit_count`.
4. {bucket_rule_en}
5. Strictly avoid `#*#`, `Menu Week`, `Menu Day`, or `*** finished ***`.
6. Output ONLY the final user prompt text ready to send to an LLM (no preamble, quotes, or commentary).
{retry_block}
SEED:
{json.dumps(seed_item, ensure_ascii=False)}

SPEC:
{json.dumps(spec_payload, ensure_ascii=False)}
"""


_SLOT_ID_LEAK_RE = re.compile(r"(?<![A-Za-z0-9_])(?:G[12]|S[1-9]|R[12]|P[12])(?![A-Za-z0-9_])")


def verify_prompt_fidelity(spec: PromptSpec, prompt_text: str) -> list[str]:
    """Verify that LLM polishing preserved all critical numbers and avoided forbidden markers."""
    errors: list[str] = []
    text = prompt_text.strip()
    if len(text) < 60:
        errors.append("prompt is too short")
    lower = text.lower()
    for bad in FORBIDDEN_PROMPT_PATTERNS:
        if bad in lower:
            errors.append(f"contains forbidden marker: {bad}")
    leaked_ids = sorted(set(_SLOT_ID_LEAK_RE.findall(text)))
    if leaked_ids:
        errors.append(f"leaked internal slot IDs: {','.join(leaked_ids)}")

    def has_num(n: int) -> bool:
        s = str(n)
        comma_s = f"{n:,}"
        return s in text or comma_s in text

    if spec.bucket != "open_a1":
        if not has_num(spec.unit_count):
            errors.append(f"missing unit_count={spec.unit_count}")
        for slot in spec.rule_slots:
            scope = slot.get("scope")
            if scope == "single":
                idx = int(slot["index"])
                if not slot.get("date_hint") and not has_num(idx):
                    errors.append(f"missing single index={idx}")
            elif scope == "range":
                s, e = int(slot["start"]), int(slot["end"])
                if not has_num(s) or not has_num(e):
                    errors.append(f"missing range bounds {s}..{e}")
            elif scope == "every":
                s, step = int(slot["start"]), int(slot["step"])
                if not has_num(s) or not has_num(step):
                    errors.append(f"missing periodic start={s}/step={step}")
                if slot.get("end") is not None and not has_num(int(slot["end"])):
                    errors.append(f"missing periodic end={slot['end']}")
    return errors


def render_natural_prompt(
    llm: ChatModel,
    spec: PromptSpec,
    seed_item: dict[str, Any],
    *,
    max_retries: int = 2,
) -> tuple[str, list[str]]:
    last_text = ""
    last_errors: list[str] = []
    for attempt in range(max_retries + 1):
        retry_note = (
            f"Previous draft failed fidelity checks: {'; '.join(last_errors)}. Keep every required number intact and never output slot codes."
            if attempt > 0
            else ""
        )
        req = build_natural_prompt_synthesis_request(spec, seed_item, retry_note=retry_note)
        raw = llm.complete(
            req,
            system="You write natural, realistic, high-quality user prompts for long-form writing tasks.",
            temperature=0.7 if attempt == 0 else 0.3,
            max_output_tokens=1_800,
        )
        text = raw.strip().strip('"').strip()
        errs = verify_prompt_fidelity(spec, text)
        last_text, last_errors = text, errs
        if not errs:
            return text, []
    return last_text, last_errors


# ---------------------------------------------------------------------------
# Layer 3: Zero-Leakage & Deduplication Audit
# ---------------------------------------------------------------------------

def _shingles(text: str, *, word_n: int = 8, char_n: int = 12) -> set[str]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    words = re.findall(r"[a-z0-9']+", normalized)
    shingles: set[str] = set()
    if len(words) >= word_n:
        for i in range(len(words) - word_n + 1):
            shingles.add("w:" + " ".join(words[i : i + word_n]))
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    if len(cjk) >= char_n:
        for i in range(len(cjk) - char_n + 1):
            shingles.add("c:" + cjk[i : i + char_n])
    return shingles


def load_leakage_references(paths: list[str | Path]) -> list[set[str]]:
    """Load reference prompt shingle sets from LongGenBench JSON or LongBench-Write JSONL files."""
    ref_sets: list[set[str]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        records: list[dict[str, Any]] = []
        if content.startswith("["):
            parsed = json.loads(content)
            if isinstance(parsed, list):
                records = [r for r in parsed if isinstance(r, dict)]
        else:
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
        for rec in records:
            prompt = str(rec.get("prompt") or rec.get("input") or "").strip()
            if prompt:
                sh = _shingles(prompt)
                if sh:
                    ref_sets.append(sh)
    return ref_sets


def check_prompt_leakage(
    prompt: str,
    ref_shingles: list[set[str]],
    *,
    max_overlap_ratio: float = 0.20,
) -> tuple[bool, str]:
    lower = prompt.lower()
    for bad in FORBIDDEN_PROMPT_PATTERNS:
        if bad in lower:
            return False, f"forbidden_marker:{bad}"
    cand = _shingles(prompt)
    if not cand or not ref_shingles:
        return True, "ok"
    for idx, ref in enumerate(ref_shingles):
        inter = len(cand & ref)
        if not inter:
            continue
        ratio = inter / max(1, min(len(cand), len(ref)))
        if ratio > max_overlap_ratio:
            return False, f"test_overlap:{ratio:.3f}:ref_{idx}"
    return True, "ok"


def deduplicate_prompts(
    records: list[dict[str, Any]],
    *,
    max_jaccard: float = 0.55,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    kept_shingles: list[set[str]] = []
    for rec in records:
        sh = _shingles(str(rec.get("prompt", "")))
        is_dup = False
        if sh:
            for prev in kept_shingles:
                union = len(sh | prev)
                if union and (len(sh & prev) / union) > max_jaccard:
                    is_dup = True
                    break
        if is_dup:
            dropped.append(rec)
        else:
            kept.append(rec)
            kept_shingles.append(sh)
    return kept, dropped


# ---------------------------------------------------------------------------
# ContractFlow v2 Self-Bootstrapping Synthesis & 4-Gate Quality Filtering
# ---------------------------------------------------------------------------

def render_formatted_document(
    contract: DocumentContract,
    unit_texts: dict[int, str],
    *,
    output_format_id: str,
    language: LanguageCode,
) -> str:
    if output_format_id == "markdown_open":
        return render_markdown_document(contract, unit_texts)

    blocks: list[str] = []
    for unit in contract.units:
        text = unit_texts.get(unit.index, "").strip()
        if not text:
            continue
        tag = _unit_tag_template(contract.unit_label, language, str(unit.index))
        if output_format_id == "markdown_h2":
            colon = "：" if language == "zh" else ":"
            blocks.append(f"## {tag}{colon}\n\n{text}")
        elif output_format_id == "markdown_h3":
            blocks.append(f"### {tag}\n\n{text}")
        elif output_format_id == "plain_colon":
            colon = "：" if language == "zh" else ":"
            blocks.append(f"{tag}{colon} {text}")
        elif output_format_id == "hr_dash":
            blocks.append(f"**{tag}**\n\n{text}")
        elif output_format_id == "hr_equals":
            blocks.append(f"=== {tag} ===\n\n{text}")
        else:
            blocks.append(f"## {unit.title}\n\n{text}")

    if output_format_id == "hr_dash":
        return "\n\n---\n\n".join(blocks)
    return "\n\n".join(blocks)


def synthesize_posttrain_sample(
    prompt_record: dict[str, Any],
    llm: ChatModel,
    *,
    config: FlowConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run pure task-agnostic ContractFlow v2 on a synthetic training prompt."""
    spec = prompt_record["spec"]
    prompt = str(prompt_record["prompt"])
    bucket: BucketKind = spec["bucket"]
    language: LanguageCode = spec["language"]
    fmt_id = str(spec.get("output_format_id", "markdown_open"))

    total_units = None if bucket == "open_a1" else int(spec["unit_count"])
    unit_label = None if bucket == "open_a1" else str(spec["unit_label"])
    min_words = int(spec["min_unit_words"])
    max_words = int(spec["max_unit_words"])

    result = run_contractflow(
        prompt,
        llm,
        config=config or FlowConfig(),
        total_units=total_units,
        unit_label=unit_label,
        min_words=min_words,
        max_words=max_words,
        checkpoint_path=checkpoint_path,
        renderer=lambda contract, texts: render_formatted_document(
            contract,
            texts,
            output_format_id=fmt_id,
            language=language,
        ),
    )

    quality = evaluate_sample_quality(prompt_record, result)
    return {
        "spec_id": prompt_record["spec_id"],
        "prompt": prompt,
        "spec": spec,
        "seed_item": prompt_record.get("seed_item", {}),
        "final_text": result["final_text"],
        "contract_v2": result["contract"],
        "units": result["units"],
        "metadata": result["metadata"],
        "quality": quality,
    }


def _repetition_score(text: str) -> float:
    """Fraction of repeated 8-word English n-grams or 16-char CJK n-grams."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    scores: list[float] = []
    if len(words) >= 40:
        grams = [tuple(words[i : i + 8]) for i in range(len(words) - 7)]
        counts = Counter(grams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        scores.append(repeated / max(1, len(grams)))
    if len(cjk) >= 60:
        grams_c = [cjk[i : i + 16] for i in range(len(cjk) - 15)]
        counts_c = Counter(grams_c)
        repeated_c = sum(c - 1 for c in counts_c.values() if c > 1)
        scores.append(repeated_c / max(1, len(grams_c)))
    return round(max(scores, default=0.0), 4)


def evaluate_sample_quality(
    prompt_record: dict[str, Any],
    result: dict[str, Any],
    *,
    min_length_ratio: float = 0.75,
    max_length_ratio: float = 1.40,
    min_review_pass_rate: float = 0.85,
    max_repetition: float = 0.12,
) -> dict[str, Any]:
    """4-gate automatic quality filter for ContractFlow v2 self-bootstrapped samples."""
    spec = prompt_record["spec"]
    contract = result.get("contract", {})
    total_units = int(contract.get("total_units") or len(contract.get("units", [])) or 0)
    present_units = len(result.get("units", []))
    cr = (present_units / total_units) if total_units > 0 else 0.0

    final_text = str(result.get("final_text", ""))
    actual_len = word_count(final_text)
    target_len = max(1, int(spec.get("target_length", 1)))
    length_ratio = round(actual_len / target_len, 4)

    meta = result.get("metadata", {})
    reviews = list(meta.get("review_results", []))
    repairs = {int(r["index"]): bool(r.get("accepted")) for r in meta.get("repair_records", []) if "index" in r}
    if reviews:
        effective_passed = sum(
            1
            for rec in reviews
            if rec.get("passed") or repairs.get(int(rec.get("index", -1)), False)
        )
        review_pass_rate = round(effective_passed / len(reviews), 4)
    else:
        review_pass_rate = 1.0

    rep_score = _repetition_score(final_text)

    reasons: list[str] = []
    if cr < 1.0:
        reasons.append(f"incomplete_units:{present_units}/{total_units}")
    if length_ratio < min_length_ratio or length_ratio > max_length_ratio:
        reasons.append(f"length_ratio_out_of_bounds:{length_ratio:.2f}")
    if review_pass_rate < min_review_pass_rate:
        reasons.append(f"low_review_pass_rate:{review_pass_rate:.2f}")
    if rep_score > max_repetition:
        reasons.append(f"high_repetition:{rep_score:.3f}")

    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "completion_rate": round(cr * 100.0, 2),
        "actual_length": actual_len,
        "target_length": target_len,
        "length_ratio": length_ratio,
        "review_count": len(reviews),
        "review_pass_rate": review_pass_rate,
        "repetition_score": rep_score,
    }


def compact_sparse_contract_json(contract_dict: dict[str, Any]) -> str:
    """Serialize ONLY the compact sparse contract IR (no expanded per-unit array)."""
    compact: dict[str, Any] = {
        "total_units": contract_dict.get("total_units"),
        "unit_label": contract_dict.get("unit_label"),
        "global_invariants": contract_dict.get("global_invariants", []),
        "rules": contract_dict.get("rules", []),
    }
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def export_sft_datasets(
    samples: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    require_quality_pass: bool = True,
) -> dict[str, Any]:
    """Export direct_sft.jsonl and contract_prefix_sft.jsonl in standard OpenAI messages format."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    direct_path = out_path / "direct_sft.jsonl"
    prefix_path = out_path / "contract_prefix_sft.jsonl"

    kept = [
        s for s in samples
        if (not require_quality_pass) or bool(s.get("quality", {}).get("passed", False))
    ]

    with (
        direct_path.open("w", encoding="utf-8") as df,
        prefix_path.open("w", encoding="utf-8") as pf,
    ):
        for s in kept:
            prompt = str(s["prompt"])
            final_text = str(s["final_text"]).strip()
            contract_json = compact_sparse_contract_json(s["contract_v2"])
            prefixed = f"<contract>\n{contract_json}\n</contract>\n\n{final_text}"
            meta = {
                "spec_id": s["spec_id"],
                "bucket": s["spec"]["bucket"],
                "domain_id": s["spec"]["domain_id"],
                "language": s["spec"]["language"],
                "length_tier": s["spec"]["length_tier"],
            }
            df.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": final_text},
                        ],
                        "meta": meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            pf.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": prefixed},
                        ],
                        "meta": meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "total_samples": len(samples),
        "kept_samples": len(kept),
        "keep_rate": round(len(kept) / max(1, len(samples)), 4),
        "by_bucket": dict(Counter(s["spec"]["bucket"] for s in kept)),
        "by_language": dict(Counter(s["spec"]["language"] for s in kept)),
        "by_length_tier": dict(Counter(s["spec"]["length_tier"] for s in kept)),
        "by_domain": dict(Counter(s["spec"]["domain_id"] for s in kept)),
        "direct_sft_path": str(direct_path),
        "contract_prefix_sft_path": str(prefix_path),
    }
    (out_path / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def export_prompts_preview_markdown(
    records: list[dict[str, Any]],
    out_path: str | Path,
) -> Path:
    """Export generated prompts into a clean Markdown document for quick human review."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Synthetic Prompts Preview ({len(records)} samples)",
        "",
        "| # | Spec ID | Bucket | Domain | Lang | Tier | Units | Fidelity |",
        "|---:|---|---|---|---|---|---:|---|",
    ]
    for i, rec in enumerate(records, start=1):
        spec = rec["spec"]
        errs = rec.get("fidelity_errors", [])
        status = "OK" if not errs else f"WARN ({len(errs)})"
        lines.append(
            f"| {i} | `{rec['spec_id']}` | `{spec['bucket']}` | `{spec['domain_id']}` | "
            f"{spec['language']} | {spec['length_tier']} | {spec['unit_count']} | {status} |"
        )
    lines.append("")
    for i, rec in enumerate(records, start=1):
        spec = rec["spec"]
        lines.extend(
            [
                f"## {i}. `{rec['spec_id']}` ({spec['domain_id']} · {spec['language']} · {spec['length_tier']})",
                "",
                f"- **Bucket**: `{spec['bucket']}` | **Target Length**: {spec['target_length']} | **Format**: `{spec['output_format_id']}`",
                f"- **Seed Title**: {rec.get('seed_item', {}).get('title', 'N/A')}",
                f"- **Rule Slots**: `{json.dumps(spec.get('rule_slots', []), ensure_ascii=False)}`",
                "",
                "```text",
                str(rec["prompt"]).strip(),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def export_canary_review_markdown(
    samples: list[dict[str, Any]],
    out_path: str | Path,
    *,
    excerpt_chars: int = 500,
) -> Path:
    """Export synthesized canary samples (beginning / middle / end + sparse contract) for human reading."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# ContractFlow v2 Canary Quality Inspection ({len(samples)} samples)",
        "",
        "| # | Spec ID | Domain | Lang | Tier | CR | Words (Ratio) | Review Pass | Rep | Pass? |",
        "|---:|---|---|---|---|---:|---|---:|---:|:---:|",
    ]
    for i, s in enumerate(samples, start=1):
        spec = s["spec"]
        q = s["quality"]
        lines.append(
            f"| {i} | `{s['spec_id']}` | `{spec['domain_id']}` | {spec['language']} | {spec['length_tier']} | "
            f"{q['completion_rate']:.1f}% | {q['actual_length']} ({q['length_ratio']:.2f}x) | "
            f"{q['review_pass_rate']*100:.1f}% | {q['repetition_score']:.3f} | {'PASS' if q['passed'] else 'FAIL'} |"
        )
    lines.append("")
    for i, s in enumerate(samples, start=1):
        spec = s["spec"]
        q = s["quality"]
        text = str(s["final_text"]).strip()
        mid_start = max(0, (len(text) // 2) - (excerpt_chars // 2))
        head = text[:excerpt_chars]
        mid = text[mid_start : mid_start + excerpt_chars] if len(text) > excerpt_chars * 2 else ""
        tail = text[-excerpt_chars:] if len(text) > excerpt_chars else ""
        lines.extend(
            [
                f"## {i}. `{s['spec_id']}` — {'PASS' if q['passed'] else 'FAIL: ' + ', '.join(q['reasons'])}",
                "",
                f"- **Prompt**:",
                "```text",
                str(s["prompt"]).strip(),
                "```",
                "- **Compiled Sparse `<contract>`**:",
                "```json",
                compact_sparse_contract_json(s["contract_v2"]),
                "```",
                "- **Excerpt — Beginning**:",
                "```text",
                head,
                "```",
            ]
        )
        if mid:
            lines.extend(["- **Excerpt — Middle**:", "```text", mid, "```"])
        if tail:
            lines.extend(["- **Excerpt — End**:", "```text", tail, "```"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
