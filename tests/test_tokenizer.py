from mindllm.tokenizer import ByteTokenizer


def test_roundtrip_ascii():
    t = ByteTokenizer()
    s = "Once upon a time."
    assert t.decode(t.encode(s)) == s


def test_roundtrip_utf8():
    t = ByteTokenizer()
    s = "Merhaba dünya — çörek!"
    assert t.decode(t.encode(s)) == s


def test_encode_returns_byte_range():
    t = ByteTokenizer()
    ids = t.encode("AB")
    assert ids == [65, 66]
    assert t.vocab_size == 256
