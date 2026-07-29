from learnbyai_research.cli import main


def test_cli_help() -> None:
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
