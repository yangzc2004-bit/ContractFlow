from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_PATH_PARTS = {
    ".env",
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "checkpoints",
    "output",
    "outputs",
    "runs",
    "tmp",
}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "generic API key": re.compile(
        r"(?i)(api[_-]?key|secret|password|bearer|authorization)\s*[:=]\s*"
        r"['\"][^'\"]{8,}['\"]"
    ),
    "provider-style token": re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    "external URL": re.compile(r"https?://", re.IGNORECASE),
}


def tracked_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line for line in completed.stdout.splitlines() if line.strip()]


def main() -> None:
    failures: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_PATH_PARTS for part in relative.parts):
            failures.append(f"forbidden path: {relative}")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{name}: {relative}")
    if failures:
        raise SystemExit("\n".join(failures))
    print("release validation passed")


if __name__ == "__main__":
    main()
