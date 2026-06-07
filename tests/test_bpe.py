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
