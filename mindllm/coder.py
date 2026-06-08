"""Mind kod agent'ı — Qwen beyni + proje bağlamı + uzun hafıza + iskelet.

İstenen dört yeteneği tek yerde birleştirir:
- kodlama:        Qwen ile kod sorularını cevaplar
- proje başlatma: scaffold() ile proje iskeleti oluşturur
- proje bağlamı:  project_context ile kod tabanını okuyup beyne verir
- uzun hatırlama: Memory ile geçmişi hatırlar ve yeni bilgileri saklar
"""
import os

from mindllm.distill import ollama_generate
from mindllm.memory import Memory
from mindllm.project import project_context

_PY_FILES = {
    "README.md": "# {name}\n\n_(yeni proje — Mind tarafından oluşturuldu)_\n",
    "main.py": ('"""{name} ana modülü."""\n\n\ndef main():\n'
                '    print("hello from {name}")\n\n\n'
                'if __name__ == "__main__":\n    main()\n'),
    ".gitignore": "__pycache__/\n*.pyc\n.venv/\n",
    "requirements.txt": "",
}
_WEB_FILES = {
    "index.html": ('<!doctype html>\n<html lang="tr"><head><meta charset="utf-8">\n'
                   "<title>{name}</title><link rel=\"stylesheet\" href=\"style.css\">\n"
                   "</head>\n<body>\n  <h1>{name}</h1>\n  <script src=\"app.js\"></script>\n"
                   "</body></html>\n"),
    "app.js": "// {name}\nconsole.log('hello from {name}');\n",
    "style.css": "body {{ font-family: sans-serif; margin: 2rem; }}\n",
}


def scaffold(name, kind="python", base="."):
    """Proje iskeleti oluşturur. Oluşturulan dosya yollarını (base'e göre) döner."""
    root = os.path.join(base, name)
    os.makedirs(root, exist_ok=True)
    templates = {"python": _PY_FILES, "web": _WEB_FILES}.get(kind, {"README.md": "# {name}\n"})
    created = []
    for rel, tmpl in templates.items():
        path = os.path.join(root, rel)
        with open(path, "w", encoding="utf-8") as f:
            f.write(tmpl.format(name=name))
        created.append(os.path.relpath(path, base))
    return created


def code_assist(question, root=None, model="qwen2.5:3b", memory=None,
                remember=True, generate_fn=None):
    """Kodlama sorusunu cevaplar: hafızadan hatırlar + proje bağlamı kullanır +
    etkileşimi hafızaya yazar. generate_fn enjekte edilebilir (test için)."""
    gen = generate_fn or ollama_generate
    mem = memory if memory is not None else Memory()
    parts = []
    recalled = mem.recall(question, k=3)
    if recalled:
        parts.append("## Hatırlanan bilgiler\n"
                     + "\n".join(f"- {r['text']}" for r in recalled))
    if root:
        parts.append("## Proje bağlamı\n" + project_context(root, max_files=80))
    context = "\n\n".join(parts)
    prompt = (
        "Sen yardımcı bir kodlama asistanısın. Aşağıdaki bağlamı kullanarak soruyu "
        "kısa, pratik ve doğru cevapla; gerekiyorsa kod ver.\n\n"
        + (f"{context}\n\n" if context else "")
        + f"=== SORU ===\n{question}\n\n=== CEVAP ==="
    )
    answer = gen(prompt, model=model, num_predict=500, temperature=0.3).strip()
    if remember:
        mem.add(f"S: {question} | C: {answer[:200]}", tags=["kod"])
    return {"question": question, "answer": answer,
            "recalled": [r["text"] for r in recalled]}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Mind kod agent'ı")
    sub = p.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("new"); ps.add_argument("name")
    ps.add_argument("--kind", default="python", choices=["python", "web"])
    ps.add_argument("--base", default=".")
    pa = sub.add_parser("ask"); pa.add_argument("question")
    pa.add_argument("--root", default=None)
    a = p.parse_args()
    if a.cmd == "new":
        for f in scaffold(a.name, kind=a.kind, base=a.base):
            print(f"oluşturuldu: {f}")
    elif a.cmd == "ask":
        res = code_assist(a.question, root=a.root)
        if res["recalled"]:
            print(f"(hatırlandı: {len(res['recalled'])} anı)")
        print(f"\n🧠 {res['answer']}\n")
