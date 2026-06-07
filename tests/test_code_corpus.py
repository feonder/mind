from mindllm.code_corpus import gather_py_sources


def test_gathers_py_and_skips_non_py(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "b.py").write_text("x = 2\n")
    (tmp_path / "c.txt").write_text("not python")
    srcs = gather_py_sources(roots=[str(tmp_path)], max_chars=10000, max_files=100)
    joined = "\n".join(srcs)
    assert "def foo" in joined and "x = 2" in joined
    assert "not python" not in joined


def test_respects_max_files(tmp_path):
    for i in range(10):
        (tmp_path / f"f{i}.py").write_text(f"a = {i}\n")
    srcs = gather_py_sources(roots=[str(tmp_path)], max_chars=10_000_000, max_files=3)
    assert len(srcs) == 3


def test_default_roots_finds_stdlib():
    srcs = gather_py_sources(max_chars=50000, max_files=20)
    assert len(srcs) > 0
    assert any("def " in s or "import " in s for s in srcs)
