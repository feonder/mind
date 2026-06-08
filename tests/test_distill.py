from mindllm.distill import build_prompt, TOPICS, NAMES


def test_build_prompt_contains_name_and_topic():
    p = build_prompt(0)
    assert NAMES[0] in p
    assert TOPICS[0] in p
    assert "story" in p.lower()


def test_build_prompt_varies_by_index():
    assert build_prompt(0) != build_prompt(1)


def test_build_prompt_wraps_with_modulo():
    n = len(TOPICS)
    assert TOPICS[0] in build_prompt(n)  # index n -> topic 0 (modulo)


def test_build_prompt_turkish():
    from mindllm.distill import build_prompt, TR_TOPICS, TR_NAMES
    p = build_prompt(0, lang="tr")
    assert TR_NAMES[0] in p
    assert TR_TOPICS[0] in p
    assert "Türkçe" in p


def test_build_prompt_default_is_english():
    from mindllm.distill import build_prompt
    assert "children's story" in build_prompt(0)


def test_build_prompt_code_mode():
    from mindllm.distill import build_prompt, CODE_TASKS
    p = build_prompt(0, mode="code")
    assert CODE_TASKS[0] in p
    assert "Python" in p
