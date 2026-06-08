from mindllm.agent import build_rag_prompt


def test_prompt_includes_question_and_context():
    p = build_rag_prompt("İstanbul nedir?", "İstanbul bir şehirdir.", lang="tr")
    assert "İstanbul nedir?" in p
    assert "İstanbul bir şehirdir." in p
    assert "WIKIPEDIA" in p


def test_prompt_without_context_tr():
    p = build_rag_prompt("Merhaba", "", lang="tr")
    assert "Merhaba" in p
    assert "WIKIPEDIA" not in p


def test_prompt_english_labels():
    p = build_rag_prompt("What is X?", "X is a thing.", lang="en")
    assert "QUESTION" in p and "ANSWER" in p
