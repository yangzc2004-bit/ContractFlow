from learnbyai_research.io_utils import export_chapter_markdown_files, extract_json_object


def test_extract_json_object_repairs_trailing_commas() -> None:
    payload = extract_json_object(
        """
        ```json
        {
          "course_bible": {
            "target_learner": "beginner",
          },
          "chapters": [
            {"title": "Intro"},
          ],
        }
        ```
        """
    )

    assert payload["course_bible"]["target_learner"] == "beginner"
    assert payload["chapters"][0]["title"] == "Intro"


def test_export_chapter_markdown_files_writes_content_only(tmp_path) -> None:
    output = {
        "pipeline": "single_agent",
        "metadata": {"ignored": True},
        "chapters": [
            {
                "plan": {"title": "第一章：探索/利用"},
                "content_markdown": "# 第一章\n\n正文",
                "review": {"ignored": True},
            },
            {
                "plan": {"title": "Second Chapter"},
                "content_markdown": "  # Second\n\nBody  ",
                "contract": {"ignored": True},
            },
        ],
    }

    written = export_chapter_markdown_files(output, tmp_path)

    assert [path.name for path in written] == ["01_第一章_探索_利用.md", "02_Second_Chapter.md"]
    assert written[0].read_text(encoding="utf-8") == "# 第一章\n\n正文\n"
    assert written[1].read_text(encoding="utf-8") == "# Second\n\nBody\n"
