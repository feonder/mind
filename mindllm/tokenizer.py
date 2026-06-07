class ByteTokenizer:
    """UTF-8 byte seviyesinde tokenizer. Vocab = 256, sıfır bağımlılık."""

    vocab_size = 256

    def encode(self, text: str) -> list:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")
