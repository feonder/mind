import os

from mindllm.project import scan_project, project_context


def test_scan_finds_code_skips_junk(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "README.md").write_text("# proje\n")
    (tmp_path / "data.bin").write_bytes(b"\x00\x01")  # kod değil
    junk = tmp_path / "__pycache__"
    junk.mkdir()
    (junk / "b.py").write_text("ignored\n")
    rels = [r for r, _ in scan_project(str(tmp_path))]
    assert "a.py" in rels
    assert "README.md" in rels
    assert "data.bin" not in rels
    assert not any("__pycache__" in r for r in rels)


def test_scan_respects_max_files(tmp_path):
    for i in range(8):
        (tmp_path / f"f{i}.py").write_text("a=1\n")
    assert len(scan_project(str(tmp_path), max_files=3)) == 3


def test_project_context_text(tmp_path):
    (tmp_path / "main.py").write_text('"""ana dosya."""\nprint("hi")\n')
    ctx = project_context(str(tmp_path))
    assert "Proje:" in ctx
    assert "main.py" in ctx
