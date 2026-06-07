import json


def get_stats(ids):
    """Bitişik token çiftlerinin frekansını sayar."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """ids içindeki tüm `pair` geçişlerini `idx` ile değiştirir."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Sıfırdan byte-level BPE (GPT-2 tarzı). Tüm UTF-8 metni işler."""

    def __init__(self):
        self.merges = {}  # (int, int) -> int
        self.vocab = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        ids = list(text.encode("utf-8"))
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and i % 200 == 0:
                print(f"merge {i + 1}/{num_merges}: idx={idx}")
        return self

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        tokens = b"".join(self.vocab[int(i)] for i in ids)
        return tokens.decode("utf-8", errors="replace")

    def save(self, path):
        data = {
            "vocab_size": self.vocab_size,
            "merges": [[p0, p1, idx] for (p0, p1), idx in self.merges.items()],
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        tok = cls()
        tok.merges = {(p0, p1): idx for p0, p1, idx in data["merges"]}
        tok.vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in sorted(tok.merges.items(), key=lambda kv: kv[1]):
            tok.vocab[idx] = tok.vocab[p0] + tok.vocab[p1]
        return tok


if __name__ == "__main__":
    import argparse
    import os
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="out/tokenizer.json")
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument("--max_chars", type=int, default=2_000_000)
    p.add_argument("--max_docs", type=int, default=30000)
    a = p.parse_args()
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories")
    parts, total = [], 0
    for i, ex in enumerate(ds["train"]):
        if i >= a.max_docs or total >= a.max_chars:
            break
        parts.append(ex["text"])
        total += len(ex["text"])
    text = "\n".join(parts)
    print(f"BPE eğitimi: {len(text)} karakter, hedef vocab {a.vocab_size}")
    tok = BPETokenizer().train(text, vocab_size=a.vocab_size, verbose=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tok.save(a.out)
    print(f"tokenizer kaydedildi: {a.out} (vocab {tok.vocab_size})")
