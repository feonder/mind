import os
import numpy as np
import mlx.core as mx


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


def prepare_tinystories(out_dir: str, max_train: int = 20000, max_val: int = 2000):
    """TinyStories'i indirir, byte'lara çevirir, train.bin/val.bin yazar."""
    from datasets import load_dataset

    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset("roneneldan/TinyStories")
    for split, n, fname in [("train", max_train, "train.bin"),
                            ("validation", max_val, "val.bin")]:
        chunks = []
        for i, ex in enumerate(ds[split]):
            if i >= n:
                break
            chunks.append(ex["text"])
        raw = ("\n".join(chunks)).encode("utf-8")
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
        arr.tofile(os.path.join(out_dir, fname))
        print(f"{fname}: {len(arr)} token")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="mindllm/data")
    p.add_argument("--max_train", type=int, default=20000)
    p.add_argument("--max_val", type=int, default=2000)
    args = p.parse_args()
    prepare_tinystories(args.out_dir, args.max_train, args.max_val)
