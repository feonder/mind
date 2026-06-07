import os
import numpy as np
import mlx.core as mx

from mindllm.tokenizer import ByteTokenizer


def get_batch(data, block_size: int, batch_size: int, rng=None):
    """data (np dizisi) içinden rastgele (x, y) batch'i üretir. y, x'in 1 kaydırılmışı."""
    if rng is None:
        rng = np.random.default_rng()
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([np.asarray(data[i:i + block_size]) for i in ix]).astype(np.int32)
    y = np.stack([np.asarray(data[i + 1:i + 1 + block_size]) for i in ix]).astype(np.int32)
    return mx.array(x), mx.array(y)


def load_bin(path: str):
    """uint16 bin dosyasını salt-okunur memmap olarak yükler."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def encode_texts(texts, tokenizer):
    """Metin listesini verilen tokenizer ile token'lar; her metin sonuna newline ekler.
    uint16 numpy dizisi döner (vocab <= 65535)."""
    nl = tokenizer.encode("\n")
    ids = []
    for t in texts:
        ids.extend(tokenizer.encode(t))
        ids.extend(nl)
    return np.array(ids, dtype=np.uint16)


def _load_tokenizer(tokenizer_path):
    if tokenizer_path is None:
        return ByteTokenizer()
    from mindllm.bpe import BPETokenizer
    return BPETokenizer.load(tokenizer_path)


def prepare_from_textfile(text_file, out_dir, tokenizer, val_frac=0.05):
    """Bir metin dosyasını okur, parçalara böler, tokenizer ile token'lar,
    son %val_frac'ı val olarak train.bin/val.bin yazar."""
    os.makedirs(out_dir, exist_ok=True)
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [c for c in text.split("\n\n") if c.strip()]
    arr = encode_texts(chunks, tokenizer)
    n_val = max(1, int(len(arr) * val_frac))
    arr[:-n_val].tofile(os.path.join(out_dir, "train.bin"))
    arr[-n_val:].tofile(os.path.join(out_dir, "val.bin"))
    print(f"train.bin: {len(arr) - n_val} token, val.bin: {n_val} token (vocab {tokenizer.vocab_size})")


def prepare_tinystories(out_dir, tokenizer=None, max_train=20000, max_val=2000):
    """TinyStories'i indirir, verilen tokenizer ile token'lar, train.bin/val.bin yazar.
    tokenizer None ise byte-level (ByteTokenizer)."""
    from datasets import load_dataset

    if tokenizer is None:
        tokenizer = ByteTokenizer()
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset("roneneldan/TinyStories")
    for split, n, fname in [("train", max_train, "train.bin"),
                            ("validation", max_val, "val.bin")]:
        texts = []
        for i, ex in enumerate(ds[split]):
            if i >= n:
                break
            texts.append(ex["text"])
        arr = encode_texts(texts, tokenizer)
        arr.tofile(os.path.join(out_dir, fname))
        print(f"{fname}: {len(arr)} token (vocab {tokenizer.vocab_size})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="mindllm/data")
    p.add_argument("--tokenizer", default=None, help="BPE tokenizer.json yolu; yoksa byte-level")
    p.add_argument("--text_file", default=None, help="verilirse TinyStories yerine bu metin dosyasından tokenize")
    p.add_argument("--max_train", type=int, default=20000)
    p.add_argument("--max_val", type=int, default=2000)
    p.add_argument("--val_frac", type=float, default=0.05)
    args = p.parse_args()
    tok = _load_tokenizer(args.tokenizer)
    if args.text_file:
        prepare_from_textfile(args.text_file, args.out_dir, tok, val_frac=args.val_frac)
    else:
        prepare_tinystories(args.out_dir, tokenizer=tok,
                            max_train=args.max_train, max_val=args.max_val)
