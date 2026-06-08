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
