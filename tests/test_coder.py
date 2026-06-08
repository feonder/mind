import os

from mindllm.coder import scaffold, code_assist
from mindllm.memory import Memory


def test_scaffold_python(tmp_path):
    created = scaffold("demo", kind="python", base=str(tmp_path))
    root = tmp_path / "demo"
    assert (root / "main.py").exists()
    assert (root / "README.md").exists()
    assert (root / ".gitignore").exists()
    assert "demo" in (root / "main.py").read_text()
    assert any("main.py" in c for c in created)


def test_scaffold_web(tmp_path):
    scaffold("site", kind="web", base=str(tmp_path))
    root = tmp_path / "site"
    assert (root / "index.html").exists()
    assert (root / "app.js").exists()
    assert "site" in (root / "index.html").read_text()


def test_code_assist_uses_memory_and_records(tmp_path):
    mem = Memory(str(tmp_path / "m.json"))
    mem.add("Bu projede MLX kullanıyoruz, PyTorch değil", tags=["kural"])
    captured = {}

    def fake_gen(prompt, **kw):
        captured["prompt"] = prompt
        return "MLX ile şöyle yaparsın: ..."

    res = code_assist("Bu projede ne kullanmalıyım?", memory=mem,
                      generate_fn=fake_gen)
    # hafıza prompt'a dahil edilmeli
    assert "MLX" in captured["prompt"]
    # cevap dönmeli
    assert "MLX" in res["answer"]
    # etkileşim hafızaya yazılmalı (1 başlangıç + 1 yeni = 2)
    assert len(mem.all()) == 2
    assert res["recalled"]  # ilgili anı hatırlandı


def test_code_assist_no_memory_no_root(tmp_path):
    mem = Memory(str(tmp_path / "m.json"))
    res = code_assist("merhaba", memory=mem, generate_fn=lambda p, **k: "selam")
    assert res["answer"] == "selam"
    assert res["recalled"] == []
