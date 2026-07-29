from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learnbyai_research.io_utils import export_chapter_markdown_files, read_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export only chapter Markdown from a generation JSON.")
    parser.add_argument("generation", help="Path to direct_prompt/single_agent/full generation JSON.")
    parser.add_argument("output_dir", help="Directory where chapter .md files will be written.")
    args = parser.parse_args()

    written = export_chapter_markdown_files(read_json(args.generation), args.output_dir)
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
