from mindllm.bpe import BPETokenizer


def test_train_reaches_vocab_size():
    text = ("the quick brown fox jumps over the lazy dog near the river. "
            "a wizard's job is to vex chumps quickly in fog. "
            "pack my box with five dozen liquor jugs every morning. ") * 400
    tok = BPETokenizer().train(text, vocab_size=320)
    assert tok.vocab_size == 320


def test_roundtrip_ascii_and_utf8():
    text = "Once upon a time, there was a tiny model. " * 50 + "Merhaba dünya — çörek!"
    tok = BPETokenizer().train(text, vocab_size=400)
    assert tok.decode(tok.encode("Once upon a time")) == "Once upon a time"
    assert tok.decode(tok.encode("Merhaba dünya — çörek!")) == "Merhaba dünya — çörek!"


def test_encode_compresses_repetitive_text():
    text = "the cat sat on the mat. " * 300
    tok = BPETokenizer().train(text, vocab_size=400)
    s = "the cat sat on the mat. " * 10
    assert len(tok.encode(s)) < len(s.encode("utf-8"))


def test_save_load_roundtrip(tmp_path):
    text = "hello world hello mind " * 100
    tok = BPETokenizer().train(text, vocab_size=320)
    p = str(tmp_path / "tok.json")
    tok.save(p)
    tok2 = BPETokenizer.load(p)
    s = "hello mind world"
    assert tok2.encode(s) == tok.encode(s)
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.decode(tok2.encode(s)) == s


def test_train_on_code_text():
    from mindllm.bpe import BPETokenizer
    code = ("def foo():\n    return 1\n" * 100
            + "import os\nfor i in range(10):\n    print(i)\n" * 100)
    tok = BPETokenizer().train(code, vocab_size=320)
    assert tok.vocab_size == 320
    assert tok.decode(tok.encode("def foo():")) == "def foo():"


def test_fast_encode_equals_naive():
    """Hızlı encode, yavaş referansla BİREBİR aynı sonucu vermeli."""
    import random
    corpus = ("the quick brown fox jumps over the lazy dog. " * 100
              + "Merhaba dünya çörek ışık ağaç. " * 50
              + "aaa bbb ccc aabbcc the the the cat sat. " * 80)
    tok = BPETokenizer().train(corpus, vocab_size=500)
    # kenar durumlar
    for s in ["", "a", "the", "aaaa", "the the the", "çörek", "Merhaba dünya"]:
        assert tok.encode(s) == tok._encode_naive(s), f"uyuşmazlık: {s!r}"
    # rastgele dizeler
    rng = random.Random(42)
    alphabet = "the quick brown fox abcçğ 123 .,! "
    for _ in range(300):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        assert tok.encode(s) == tok._encode_naive(s), f"uyuşmazlık: {s!r}"
