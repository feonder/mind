# Mind LLM — Kod Modeli Deneyi (Planı)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. TDD, checkbox steps.

**Goal:** Mind modelini **Python kodu** üstünde eğit. Kod korpusu = makinedeki gerçek `.py` dosyaları (Python stdlib + venv site-packages) — internet/gating yok, offline güvenilir. Kendi kod-BPE'si + eğitim + örnekleme.

**Architecture:** Yeni `mindllm/code_corpus.py` (lokal `.py` topla). `bpe.py` ve `data.py`'a `--text_file` genelleştirmesi (TinyStories yerine herhangi bir metin korpusundan eğit/tokenize). Eğitim/örnekleme mevcut tokenizer-agnostik pipeline ile.

**Beklenti (dürüst):** 10M model kodun *görünüşünü* öğrenir (girinti, `def`, `import`, parantez), kod-benzeri üretir ama çalışan program yazamaz. Kanıt/öğrenme amaçlı.

**Tech Stack:** Mevcut yığın (Python 3.9 venv, MLX, pytest). venv: `source /Users/emir/Desktop/Projeler/mind/.venv/bin/activate`. Eğitimler `--throttle 0.3 --batch_size 16` (sessiz mod).

---

## Task 1: code_corpus.py — lokal Python kodu topla

**Files:** Create `mindllm/code_corpus.py`, `tests/test_code_corpus.py`

- [ ] **Step 1: Test yaz** — `tests/test_code_corpus.py`:

```python
from mindllm.code_corpus import gather_py_sources


def test_gathers_py_and_skips_non_py(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "b.py").write_text("x = 2\n")
    (tmp_path / "c.txt").write_text("not python")
    srcs = gather_py_sources(roots=[str(tmp_path)], max_chars=10000, max_files=100)
    joined = "\n".join(srcs)
    assert "def foo" in joined and "x = 2" in joined
    assert "not python" not in joined


def test_respects_max_files(tmp_path):
    for i in range(10):
        (tmp_path / f"f{i}.py").write_text(f"a = {i}\n")
    srcs = gather_py_sources(roots=[str(tmp_path)], max_chars=10_000_000, max_files=3)
    assert len(srcs) == 3


def test_default_roots_finds_stdlib():
    srcs = gather_py_sources(max_chars=50000, max_files=20)
    assert len(srcs) > 0
    assert any("def " in s or "import " in s for s in srcs)
```

- [ ] **Step 2: Çalıştır, FAIL gör** — `python -m pytest tests/test_code_corpus.py -v` → ModuleNotFoundError

- [ ] **Step 3: Implementasyon** — `mindllm/code_corpus.py`:

```python
import os
import sysconfig


def _default_roots():
    roots = []
    for key in ("stdlib", "purelib"):
        p = sysconfig.get_path(key)
        if p and os.path.isdir(p):
            roots.append(p)
    return roots


def gather_py_sources(roots=None, max_chars=5_000_000, max_files=4000):
    """Kök dizinlerdeki .py dosyalarını okur, içerik string listesi döner.
    roots None ise Python stdlib + site-packages otomatik bulunur."""
    if roots is None:
        roots = _default_roots()
    sources = []
    total = 0
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for fn in sorted(files):
                if not fn.endswith(".py"):
                    continue
                try:
                    with open(os.path.join(dirpath, fn), "r", encoding="utf-8") as f:
                        text = f.read()
                except (UnicodeDecodeError, OSError):
                    continue
                if not text.strip():
                    continue
                sources.append(text)
                total += len(text)
                if total >= max_chars or len(sources) >= max_files:
                    return sources
    return sources


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="out/code_corpus.txt")
    p.add_argument("--max_chars", type=int, default=5_000_000)
    p.add_argument("--max_files", type=int, default=4000)
    a = p.parse_args()
    srcs = gather_py_sources(max_chars=a.max_chars, max_files=a.max_files)
    text = "\n\n".join(srcs)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"kod korpusu: {len(srcs)} dosya, {len(text)} karakter -> {a.out}")
```

- [ ] **Step 4: Çalıştır, PASS gör** — `python -m pytest tests/test_code_corpus.py -v` (3 test)
- [ ] **Step 5: Commit** — `git add mindllm/code_corpus.py tests/test_code_corpus.py && git commit -m "feat: code_corpus — lokal Python kodu toplama"`

---

## Task 2: bpe.py — --text_file ile herhangi bir korpustan eğit

**Files:** Modify `mindllm/bpe.py` (sadece `__main__`), Test: `tests/test_bpe.py`

- [ ] **Step 1: Test ekle** — `tests/test_bpe.py` SONUNA:

```python
def test_train_on_code_text():
    from mindllm.bpe import BPETokenizer
    code = ("def foo():\n    return 1\n" * 100
            + "import os\nfor i in range(10):\n    print(i)\n" * 100)
    tok = BPETokenizer().train(code, vocab_size=320)
    assert tok.vocab_size == 320
    assert tok.decode(tok.encode("def foo():")) == "def foo():"
```

- [ ] **Step 2: Çalıştır, PASS** (bu zaten geçmeli — BPE genel; doğrulama testi) — `python -m pytest tests/test_bpe.py::test_train_on_code_text -v`

- [ ] **Step 3: `bpe.py` `__main__` bloğunu genelleştir** — mevcut `__main__` içinde TinyStories yükleyen kısmı `--text_file` desteğiyle değiştir. Tam yeni `__main__`:

```python
if __name__ == "__main__":
    import argparse
    import os
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="out/tokenizer.json")
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument("--max_chars", type=int, default=2_000_000)
    p.add_argument("--max_docs", type=int, default=30000)
    p.add_argument("--text_file", default=None, help="verilirse TinyStories yerine bu dosyadan eğit")
    a = p.parse_args()
    if a.text_file:
        with open(a.text_file, "r", encoding="utf-8") as f:
            text = f.read()[:a.max_chars]
    else:
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

- [ ] **Step 4: Tüm testler PASS** — `python -m pytest -q`
- [ ] **Step 5: Commit** — `git add mindllm/bpe.py tests/test_bpe.py && git commit -m "feat: bpe --text_file (herhangi bir korpustan eğit)"`

---

## Task 3: data.py — metin dosyasından tokenize

**Files:** Modify `mindllm/data.py`, Test: `tests/test_data.py`

- [ ] **Step 1: Test ekle** — `tests/test_data.py` SONUNA:

```python
def test_prepare_from_textfile(tmp_path):
    from mindllm.data import prepare_from_textfile, load_bin
    from mindllm.bpe import BPETokenizer
    tf = tmp_path / "corpus.txt"
    tf.write_text("def f():\n    return 1\n\nimport os\nx = os.getcwd()\n" * 50)
    tok = BPETokenizer().train(tf.read_text(), vocab_size=300)
    out = tmp_path / "d"
    prepare_from_textfile(str(tf), str(out), tok, val_frac=0.1)
    tr = load_bin(str(out / "train.bin"))
    va = load_bin(str(out / "val.bin"))
    assert len(tr) > 0 and len(va) > 0
    assert tr.max() < tok.vocab_size
```

- [ ] **Step 2: Çalıştır, FAIL** — `python -m pytest tests/test_data.py::test_prepare_from_textfile -v` → ImportError

- [ ] **Step 3: `data.py`'a fonksiyon + CLI ekle** — `encode_texts`'ten SONRA şu fonksiyonu ekle:

```python
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
```

Ve `data.py` `__main__` bloğunu, `--text_file` verilirse bu yolu kullanacak şekilde değiştir:

```python
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
```

- [ ] **Step 4: Tüm testler PASS** — `python -m pytest -q` (25 test bekleniyor: 22 + code_corpus 3; bpe/data eklemeleri dahil → say ve raporla)
- [ ] **Step 5: Commit** — `git add mindllm/data.py tests/test_data.py && git commit -m "feat: data prepare_from_textfile (--text_file)"`

---

## Task 4: Gerçek koşu (manuel, sessiz mod)

- [ ] **Step 1:** Kod korpusu topla — `python -m mindllm.code_corpus --out out/code_corpus.txt --max_chars 5000000`
- [ ] **Step 2:** Kod BPE'si eğit — `python -u -m mindllm.bpe --out out/tokenizer_code.json --vocab_size 4096 --text_file out/code_corpus.txt --max_chars 5000000`
- [ ] **Step 3:** Tokenize — `python -u -m mindllm.data --out_dir mindllm/data_code --tokenizer out/tokenizer_code.json --text_file out/code_corpus.txt`
- [ ] **Step 4:** Eğit (sessiz) — `VECLIB_MAXIMUM_THREADS=2 nice -n 10 python -u -m mindllm.train --data_dir mindllm/data_code --out_dir out_code --tokenizer out/tokenizer_code.json --max_iters 5000 --batch_size 16 --eval_interval 100 --throttle 0.3`
- [ ] **Step 5:** Örnek — `python -m mindllm.sample --out_dir out_code --ckpt out_code/ckpt.safetensors --tokenizer out/tokenizer_code.json --prompt "def " --max_new_tokens 200`
- [ ] **Step 6:** `Notlar/kod-deney-sonuc.md` yaz, commit.

---

## Başarı Kriteri
- [ ] `gather_py_sources` lokal `.py` topluyor (Task 1)
- [ ] bpe & data `--text_file` ile çalışıyor (Task 2-3)
- [ ] Tüm testler geçiyor
- [ ] Model kod üstünde eğitiliyor, kod-benzeri üretiyor (Task 4)
- [ ] Sonuç belgelendi
