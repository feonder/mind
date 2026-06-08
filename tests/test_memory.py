from mindllm.memory import Memory


def test_add_and_all(tmp_path):
    mem = Memory(str(tmp_path / "m.json"))
    mem.add("Emir mavi temayı seviyor", tags=["tercih"])
    assert len(mem.all()) == 1
    assert mem.all()[0]["text"] == "Emir mavi temayı seviyor"


def test_recall_finds_relevant(tmp_path):
    mem = Memory(str(tmp_path / "m.json"))
    mem.add("Proje adı mind, sıfırdan LLM")
    mem.add("Kahvaltıda yumurta yendi")
    hits = mem.recall("mind LLM projesi", k=1)
    assert len(hits) == 1
    assert "mind" in hits[0]["text"]


def test_recall_empty_query(tmp_path):
    mem = Memory(str(tmp_path / "m.json"))
    mem.add("bir şey")
    assert mem.recall("") == []


def test_persistence_across_instances(tmp_path):
    path = str(tmp_path / "m.json")
    m1 = Memory(path)
    m1.add("kalıcı anı")
    m2 = Memory(path)  # yeniden yükle
    assert len(m2.all()) == 1
    assert m2.all()[0]["text"] == "kalıcı anı"


def test_clear(tmp_path):
    mem = Memory(str(tmp_path / "m.json"))
    mem.add("a")
    mem.clear()
    assert mem.all() == []
