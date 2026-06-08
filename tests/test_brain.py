from mindllm.brain import _parse_choice, choose_tool, run, TOOLS


def test_tools_have_desc_and_registry():
    assert "weather" in TOOLS and "wikipedia" in TOOLS
    for name, t in TOOLS.items():
        assert "desc" in t and "fn" in t


def test_parse_choice_extracts_json():
    raw = 'Tabii: {"tool":"weather","args":{"location":"Istanbul"}} işte bu.'
    c = _parse_choice(raw)
    assert c["tool"] == "weather"
    assert c["args"]["location"] == "Istanbul"


def test_parse_choice_unknown_tool_falls_back():
    c = _parse_choice('{"tool":"banana","args":{}}')
    assert c["tool"] == "none"


def test_parse_choice_garbage():
    assert _parse_choice("hiç json yok")["tool"] == "none"


def test_choose_tool_uses_brain():
    def fake(prompt, **kw):
        assert "weather" in prompt  # araç menüsü prompt'a girmiş
        return '{"tool":"weather","args":{"location":"Ankara"}}'
    c = choose_tool("Ankara'da hava nasıl?", fake)
    assert c["tool"] == "weather" and c["args"]["location"] == "Ankara"


def test_run_none_tool_no_network():
    calls = {"n": 0}

    def fake(prompt, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return '{"tool":"none","args":{}}'  # araç seçimi
        return "42"  # nihai cevap

    res = run("2+2 kaç... değil, sadece selam", generate_fn=fake)
    assert res["tool"] == "none"
    assert res["answer"] == "42"
    assert res["result"] == ""


def test_calculator_real_math():
    from mindllm.brain import tool_calculator
    assert tool_calculator("17 + 28") == "17 + 28 = 45"
    assert tool_calculator("6 * 7") == "6 * 7 = 42"
    assert "calculator" in TOOLS


def test_calculator_rejects_code():
    from mindllm.brain import tool_calculator
    import pytest
    with pytest.raises(Exception):
        tool_calculator("__import__('os').system('ls')")
