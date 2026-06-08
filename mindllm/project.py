"""Mind proje bağlamı — bir kod tabanını tarayıp yapı/özet çıkarır.

Agent'a "proje bağlam kurma" yeteneği verir: dosya ağacı + anahtar dosyaların
başlıkları, beyne (Qwen) verilecek bir bağlam metnine dönüştürülür.
"""
import os

CODE_EXT = {".py", ".js", ".ts", ".tsx", ".swift", ".java", ".go", ".rs",
            ".c", ".cpp", ".h", ".md", ".json", ".toml", ".yaml", ".yml", ".sh"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".obsidian",
             ".superpowers", "out", "out_bpe", "out_code", "out_tr", "out_124m",
             "out_distill", "dist", "build"}


def scan_project(root, max_files=200):
    """Kök dizindeki kod dosyalarını tarar; [(rel_path, size), ...] döner."""
    files = []
    for dirpath, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for n in sorted(names):
            if os.path.splitext(n)[1] in CODE_EXT:
                full = os.path.join(dirpath, n)
                try:
                    size = os.path.getsize(full)
                except OSError:
                    continue
                files.append((os.path.relpath(full, root), size))
                if len(files) >= max_files:
                    return files
    return files


def file_head(path, n_lines=15):
    """Bir dosyanın ilk n satırını döner (docstring/başlık için)."""
    try:
        with open(path, encoding="utf-8") as f:
            return "".join(f.readline() for _ in range(n_lines))
    except OSError:
        return ""


def project_context(root, max_files=120, head_files=5):
    """Beyne verilecek bağlam metni: dosya ağacı + ilk birkaç dosyanın başlığı."""
    files = scan_project(root, max_files=max_files)
    name = os.path.basename(os.path.abspath(root))
    lines = [f"# Proje: {name} ({len(files)} kod dosyası)", "", "## Dosyalar"]
    for rel, size in files:
        lines.append(f"- {rel} ({size} B)")
    # Birkaç önemli dosyanın başlığını ekle (README/ana dosyalar)
    priority = [f for f in files if os.path.basename(f[0]).lower()
                in ("readme.md", "__init__.py", "main.py", "index.md")][:head_files]
    if priority:
        lines += ["", "## Anahtar dosya başlıkları"]
        for rel, _ in priority:
            head = file_head(os.path.join(root, rel)).strip()
            if head:
                lines += [f"\n### {rel}", head[:500]]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default=".")
    p.add_argument("--max_files", type=int, default=120)
    a = p.parse_args()
    print(project_context(a.root, max_files=a.max_files))
