from pathlib import Path

from safetranslate.data.download import fetch


def test_fetch_downloads_then_skips_existing(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("hello")
    target = tmp_path / "raw" / "copy.txt"

    fetch(source.as_uri(), target)
    assert target.read_text() == "hello"
    assert not (tmp_path / "raw" / "copy.txt.part").exists()

    # An existing file is kept, not re-downloaded
    source.write_text("changed")
    fetch(source.as_uri(), target)
    assert target.read_text() == "hello"
