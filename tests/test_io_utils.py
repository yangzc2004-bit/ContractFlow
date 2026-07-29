from dataclasses import dataclass

from learnbyai_research.io_utils import extract_json_object, read_json, write_json


def test_extract_json_object_repairs_trailing_commas() -> None:
    payload = extract_json_object(
        """
        ```json
        {
          "plan": {
            "summary": "example",
          },
          "segments": [
            {"index": 1},
          ],
        }
        ```
        """
    )

    assert payload["plan"]["summary"] == "example"
    assert payload["segments"][0]["index"] == 1


@dataclass
class ExampleRecord:
    index: int
    labels: tuple[str, ...]


def test_write_json_serializes_dataclasses(tmp_path) -> None:
    path = tmp_path / "record.json"
    write_json(path, ExampleRecord(index=1, labels=("a", "b")))
    assert read_json(path) == {"index": 1, "labels": ["a", "b"]}
