# Mind LLM — F2 Uygulama Planı (Sıfırdan BPE Tokenizer)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Byte-level tokenizer yerine sıfırdan yazılmış bir BPE (Byte-Pair Encoding) tokenizer ekle; pipeline'ı tokenizer-agnostik yap; yeni tokenizer'la modeli yeniden eğitip F1 ile karşılaştır.

**Architecture:** Yeni `mindllm/bpe.py` → `BPETokenizer` (train/encode/decode/save/load), GPT-2 tarzı byte-level BPE, minbpe ruhu, sıfır bağımlılık. `data.py`, `train.py`, `sample.py` herhangi bir tokenizer nesnesiyle (`.encode`/`.decode`/`.vocab_size`) çalışacak şekilde genelleştirilir. `ByteTokenizer` (F1 baseline) korunur.

**Tech Stack:** Mevcut F1 yığını (Python 3.9 venv, MLX 0.29.3, numpy, datasets, pytest). venv: `source /Users/emir/Desktop/Projeler/mind/.venv/bin/activate`. Testler repo kökünden `python -m pytest`.

> **Not (pure-Python BPE hızı):** Sıfırdan BPE eğitimi ve kodlaması pure-Python'da yavaştır (her merge tüm diziyi tarar). Bu yüzden Task 4'teki gerçek koşular **modest alt küme** (~2MB tokenizer eğitimi, ~20000 hikâye) ve **arka plan** ile yapılır. Birim testleri minik veriyle çalışır, hızlıdır.

---

## Dosya Yapısı

```
mindllm/
├── bpe.py          # YENİ: BPETokenizer + get_stats/merge yardımcıları + eğitim CLI
├── tokenizer.py    # değişmez (ByteTokenizer korunur)
├── data.py         # DEĞİŞ: encode_texts() ekle, prepare_tinystories tokenizer alır, --tokenizer CLI
├── train.py        # DEĞİŞ: --tokenizer CLI (vocab_size'ı tokenizer'dan al)
├── sample.py       # DEĞİŞ: --tokenizer CLI + sample_main(tokenizer_path=...)
└── model.py        # değişmez
tests/
├── test_bpe.py     # YENİ
├── test_data.py    # DEĞİŞ: encode_texts testleri eklenir
├── test_train.py   # DEĞİŞ: custom vocab eğitim testi eklenir
└── test_sample.py  # DEĞİŞ: BPE ile sample testi eklenir
```

---

## Task 1: Sıfırdan BPE Tokenizer

**Files:**
- Create: `mindllm/bpe.py`
- Test: `tests/test_bpe.py`

- [ ] **Step 1: Başarısız testi yaz**

`tests/test_bpe.py`:

```python
from mindllm.bpe import BPETokenizer


def test_train_reaches_vocab_size():
    text = "the quick brown fox jumps over the lazy dog. " * 300
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
```

- [ ] **Step 2: Testi çalıştır, başarısız olduğunu gör**

Run: `python -m pytest tests/test_bpe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mindllm.bpe'`

- [ ] **Step 3: Minimal implementasyonu yaz**

`mindllm/bpe.py`:

```python
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
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx}")
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
```

- [ ] **Step 4: Testi çalıştır, geçtiğini gör**

Run: `python -m pytest tests/test_bpe.py -v`
Expected: PASS (4 test)

- [ ] **Step 5: Commit**

```bash
git add mindllm/bpe.py tests/test_bpe.py
git commit -m "feat: sıfırdan byte-level BPE tokenizer"
```

---

## Task 2: data.py — tokenizer-agnostik veri hazırlama

**Files:**
- Modify: `mindllm/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Başarısız testleri ekle**

`tests/test_data.py` dosyasının SONUNA ekle:

```python
def test_encode_texts_byte():
    from mindllm.data import encode_texts
    from mindllm.tokenizer import ByteTokenizer
    arr = encode_texts(["AB", "C"], ByteTokenizer())
    # "AB"+"\n"+"C"+"\n" => [65, 66, 10, 67, 10]
    assert list(arr) == [65, 66, 10, 67, 10]
    assert arr.dtype == np.uint16


def test_encode_texts_bpe_ids_in_range():
    from mindllm.data import encode_texts
    from mindllm.bpe import BPETokenizer
    tok = BPETokenizer().train("hello world " * 100, vocab_size=300)
    arr = encode_texts(["hello world"], tok)
    assert arr.dtype == np.uint16
    assert arr.max() < tok.vocab_size
```

- [ ] **Step 2: Testi çalıştır, başarısız olduğunu gör**

Run: `python -m pytest tests/test_data.py::test_encode_texts_byte -v`
Expected: FAIL — `ImportError: cannot import name 'encode_texts'`

- [ ] **Step 3: data.py'ı güncelle**

Mevcut `mindllm/data.py` içinde `prepare_tinystories` ve `__main__` bloğunu AŞAĞIDAKİYLE değiştir (üstteki `get_batch` ve `load_bin` aynen kalır). Ayrıca dosyanın başına `from mindllm.tokenizer import ByteTokenizer` importunu ekle:

```python
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
    p.add_argument("--max_train", type=int, default=20000)
    p.add_argument("--max_val", type=int, default=2000)
    args = p.parse_args()
    tok = _load_tokenizer(args.tokenizer)
    prepare_tinystories(args.out_dir, tokenizer=tok,
                        max_train=args.max_train, max_val=args.max_val)
```

- [ ] **Step 4: Testleri çalıştır, geçtiğini gör**

Run: `python -m pytest tests/test_data.py -v`
Expected: PASS (4 test: 2 eski + 2 yeni)

- [ ] **Step 5: Commit**

```bash
git add mindllm/data.py tests/test_data.py
git commit -m "feat: data.py tokenizer-agnostik (encode_texts + --tokenizer)"
```

---

## Task 3: train.py & sample.py — tokenizer entegrasyonu

**Files:**
- Modify: `mindllm/train.py` (sadece `__main__` bloğu)
- Modify: `mindllm/sample.py` (`sample_main` + `__main__`)
- Test: `tests/test_train.py`, `tests/test_sample.py`

- [ ] **Step 1: Başarısız testleri ekle**

`tests/test_train.py` SONUNA ekle:

```python
def test_train_with_custom_vocab(tmp_path):
    import numpy as np
    from mindllm.model import GPTConfig
    from mindllm.train import train_main
    d = tmp_path / "data"
    d.mkdir()
    arr = np.tile(np.arange(60, dtype=np.uint16), 3000)  # id'ler < 300
    arr.tofile(str(d / "train.bin"))
    arr.tofile(str(d / "val.bin"))
    cfg = GPTConfig(vocab_size=300, block_size=32, n_layer=2, n_head=2, n_embd=64)
    hist = train_main(str(d), str(tmp_path / "out"), cfg=cfg,
                      max_iters=200, batch_size=16, eval_interval=50)
    assert hist[-1][1] < hist[0][1]
```

`tests/test_sample.py` SONUNA ekle:

```python
def test_sample_main_with_bpe(tmp_path):
    import os
    import json
    import mlx.core as mx
    from mindllm.model import GPT, GPTConfig
    from mindllm.bpe import BPETokenizer
    from mindllm.sample import sample_main
    tok = BPETokenizer().train("once upon a time there was a cat " * 100, vocab_size=300)
    tp = str(tmp_path / "tok.json")
    tok.save(tp)
    out = str(tmp_path / "out")
    os.makedirs(out)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=64)
    m = GPT(cfg)
    mx.eval(m.parameters())
    m.save_weights(f"{out}/ckpt.safetensors")
    json.dump({"iter": 1, "config": vars(cfg)}, open(f"{out}/meta.json", "w"))
    text = sample_main(f"{out}/ckpt.safetensors", out, prompt="once",
                       max_new_tokens=5, temperature=0.8, tokenizer_path=tp)
    assert isinstance(text, str) and len(text) >= 1
```

- [ ] **Step 2: Testleri çalıştır, başarısız olduğunu gör**

Run: `python -m pytest tests/test_sample.py::test_sample_main_with_bpe -v`
Expected: FAIL — `TypeError: sample_main() got an unexpected keyword argument 'tokenizer_path'`

- [ ] **Step 3: sample.py'ı güncelle**

`mindllm/sample.py` içindeki `sample_main` fonksiyonunu ve `__main__` bloğunu AŞAĞIDAKİYLE değiştir (üstteki `generate` ve importlar aynen kalır):

```python
def sample_main(ckpt, out_dir, prompt="Once upon a time", max_new_tokens=200,
                temperature=0.8, tokenizer_path=None):
    import json
    meta = json.load(open(f"{out_dir}/meta.json"))
    cfg = GPTConfig(**meta["config"])
    model = GPT(cfg)
    model.load_weights(ckpt)
    mx.eval(model.parameters())
    if tokenizer_path:
        from mindllm.bpe import BPETokenizer
        tok = BPETokenizer.load(tokenizer_path)
    else:
        tok = ByteTokenizer()
    idx = mx.array([tok.encode(prompt)], dtype=mx.int32)
    out = generate(model, idx, max_new_tokens, cfg.block_size, temperature)
    text = tok.decode([int(t) for t in out[0].tolist()])
    print(text)
    return text


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="out")
    p.add_argument("--ckpt", default="out/ckpt.safetensors")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--tokenizer", default=None, help="BPE tokenizer.json yolu; yoksa byte-level")
    a = p.parse_args()
    sample_main(a.ckpt, a.out_dir, a.prompt, a.max_new_tokens, a.temperature,
                tokenizer_path=a.tokenizer)
```

- [ ] **Step 4: train.py `__main__` bloğunu güncelle**

`mindllm/train.py` içindeki `if __name__ == "__main__":` bloğunu AŞAĞIDAKİYLE değiştir (üstteki `loss_fn`, `make_step`, `train_main` aynen kalır):

```python
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="mindllm/data")
    p.add_argument("--out_dir", default="out")
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--tokenizer", default=None, help="BPE tokenizer.json yolu; vocab_size buradan alınır")
    a = p.parse_args()
    cfg = GPTConfig()
    if a.tokenizer:
        from mindllm.bpe import BPETokenizer
        tok = BPETokenizer.load(a.tokenizer)
        cfg = GPTConfig(vocab_size=tok.vocab_size)
        print(f"tokenizer yüklendi: vocab {cfg.vocab_size}")
    train_main(a.data_dir, a.out_dir, cfg=cfg, max_iters=a.max_iters,
               batch_size=a.batch_size, eval_interval=a.eval_interval,
               lr=a.lr, resume=a.resume)
```

- [ ] **Step 5: Testleri çalıştır, geçtiğini gör**

Run: `python -m pytest -v`
Expected: PASS (tüm testler — F1'in 14'ü + bpe 4 + data 2 + train 1 + sample 1 = 22 test)

- [ ] **Step 6: Commit**

```bash
git add mindllm/train.py mindllm/sample.py tests/test_train.py tests/test_sample.py
git commit -m "feat: train/sample BPE tokenizer entegrasyonu (--tokenizer)"
```

---

## Task 4: Gerçek koşu — BPE eğit, yeniden eğit, F1 ile karşılaştır (manuel)

Otomatik test değil; gerçek tokenizer + eğitim + karşılaştırma. **Pure-Python BPE yavaş → her ağır adım arka planda.**

- [ ] **Step 1: BPE tokenizer'ı eğit (arka plan, ~2MB, vocab 4096)**

Run:
```bash
source .venv/bin/activate && python -u -m mindllm.bpe --out out/tokenizer.json --vocab_size 4096 --max_chars 2000000 2>&1 | tee out/bpe_train.log
```
Expected: `tokenizer kaydedildi: out/tokenizer.json (vocab 4096)`. (Süre: pure-Python'da ~15-40 dk olabilir; yavaşsa `--max_chars 1000000` veya `--vocab_size 2048` ile tekrar dene.)

- [ ] **Step 2: Korpusu BPE ile token'la (arka plan, ~20000 hikâye)**

Run:
```bash
python -u -m mindllm.data --out_dir mindllm/data_bpe --tokenizer out/tokenizer.json --max_train 20000 --max_val 1000 2>&1 | tee out/bpe_tokenize.log
```
Expected: `train.bin: ... token (vocab 4096)`. Byte-level'a göre **belirgin daha az token** (~3-4x sıkışma).

- [ ] **Step 3: Modeli BPE ile yeniden eğit (arka plan)**

Run:
```bash
python -u -m mindllm.train --data_dir mindllm/data_bpe --out_dir out_bpe --tokenizer out/tokenizer.json --max_iters 5000 --batch_size 32 --eval_interval 100 2>&1 | tee out_bpe/train.log
```
Expected: `tokenizer yüklendi: vocab 4096`, loss düşüyor. Aynı 5000 iter'de model artık **çok daha fazla metin** görüyor (token başına ~3-4 byte).

- [ ] **Step 4: Örnek üret ve F1 ile karşılaştır**

Run:
```bash
python -m mindllm.sample --out_dir out_bpe --ckpt out_bpe/ckpt.safetensors --tokenizer out/tokenizer.json --prompt "Once upon a time" --max_new_tokens 200
```
Expected: tutarlı İngilizce; byte-level F1'e göre eşit/daha iyi tutarlılık (daha kısa diziler sayesinde).

- [ ] **Step 5: Sonucu vault'a yaz ve commit**

`Notlar/F2-sonuc.md` oluştur: BPE vocab, sıkışma oranı (byte token sayısı vs BPE token sayısı), eğitim süreleri, son loss, örnek çıktı, F1 ile karşılaştırma. Sonra:
```bash
git add "Notlar/F2-sonuc.md"
git commit -m "docs: F2 sonuç (sıfırdan BPE tokenizer + karşılaştırma)"
```

---

## Başarı Kriteri

- [ ] `BPETokenizer` sıfırdan çalışıyor: train/encode/decode/save/load (Task 1, 4 test)
- [ ] Pipeline tokenizer-agnostik: byte VEYA BPE ile aynı kod çalışıyor (Task 2-3)
- [ ] Tüm birim testleri geçiyor (22 test, Task 3 Step 5)
- [ ] BPE byte-level'a göre diziyi belirgin sıkıştırıyor (Task 4 Step 2)
- [ ] Model BPE ile yeniden eğitilip tutarlı metin üretiyor (Task 4 Step 3-4)
- [ ] F2 sonucu F1 ile karşılaştırmalı belgelendi (Task 4 Step 5)
```
