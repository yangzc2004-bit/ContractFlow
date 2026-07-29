from __future__ import annotations

import re

from .schemas import CourseTask, Material, MaterialRole, RoutedMaterials


ROLE_LIMITS: dict[MaterialRole, int] = {
    "requirements": 12_000,
    "reference": 16_000,
    "style": 6_000,
}


def route_materials(task: CourseTask, *, enabled: bool = True) -> RoutedMaterials:
    buckets: dict[MaterialRole, list[str]] = {"requirements": [], "reference": [], "style": []}
    assignments: list[dict[str, str]] = []
    for material in task.materials:
        role = infer_role(material) if enabled else "reference"
        text = material.text.strip()
        if not text:
            continue
        buckets[role].append(format_material_block(material, role))
        assignments.append({"name": material.name, "role": role, "source": material.source or ""})

    return RoutedMaterials(
        requirements=limit_text("\n\n".join(buckets["requirements"]), ROLE_LIMITS["requirements"]),
        reference=limit_text("\n\n".join(buckets["reference"]), ROLE_LIMITS["reference"]),
        style=limit_text("\n\n".join(buckets["style"]), ROLE_LIMITS["style"]),
        assignments=assignments,
    )


def infer_role(material: Material) -> MaterialRole:
    if material.role in {"requirements", "reference", "style"}:
        return material.role
    sample = f"{material.name}\n{material.text[:1500]}".lower()
    if re.search(r"requirements?|rubric|outline|syllabus|目录|大纲|要求|评分|目标", sample):
        return "requirements"
    if re.search(r"style|tone|sample|范文|文风|风格|样例", sample):
        return "style"
    return "reference"


def format_material_block(material: Material, role: MaterialRole) -> str:
    label = {"requirements": "REQUIREMENTS", "reference": "REFERENCE", "style": "STYLE_SAMPLE"}[role]
    return f"[{label}: {material.name}]\n{material.text.strip()}"


def limit_text(text: str, limit: int) -> str:
    value = text.strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "\n...[truncated for prompt budget]"
