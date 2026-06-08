"""Mind uzun hafıza — kalıcı, oturumlar arası bellek deposu.

Agent'ın hatırlaması gereken gerçekleri/notları JSON dosyada saklar ve basit
kelime-örtüşmesiyle alakalı anıları geri çağırır (sıfır bağımlılık, embedding yok).
Bu, "uzun hatırlama" yeteneğinin temelidir; agent buraya yazar ve buradan okur.
"""
import json
import os
import re


def _tokenize(text):
    return set(re.findall(r"\w+", text.lower()))


class Memory:
    def __init__(self, path="out/mind_memory.json"):
        self.path = path
        self.items = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.items = json.load(f)

    def add(self, text, tags=None):
        """Bir anı ekler ve diske yazar. Eklenen kaydı döner."""
        entry = {"id": len(self.items) + 1, "text": text, "tags": list(tags or [])}
        self.items.append(entry)
        self._save()
        return entry

    def recall(self, query, k=3):
        """Sorguyla en alakalı k anıyı döner (kelime örtüşmesi skoruyla)."""
        q = _tokenize(query)
        if not q:
            return []
        scored = []
        for it in self.items:
            words = _tokenize(it["text"]) | _tokenize(" ".join(it.get("tags", [])))
            overlap = len(q & words)
            if overlap:
                scored.append((overlap, it))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [it for _, it in scored[:k]]

    def all(self):
        return list(self.items)

    def clear(self):
        self.items = []
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.items, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Mind uzun hafıza")
    sub = p.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("add"); pa.add_argument("text"); pa.add_argument("--tags", nargs="*")
    pr = sub.add_parser("recall"); pr.add_argument("query"); pr.add_argument("--k", type=int, default=3)
    sub.add_parser("list")
    p.add_argument("--path", default="out/mind_memory.json")
    a = p.parse_args()
    mem = Memory(a.path)
    if a.cmd == "add":
        e = mem.add(a.text, a.tags)
        print(f"eklendi #{e['id']}: {e['text']}")
    elif a.cmd == "recall":
        for it in mem.recall(a.query, a.k):
            print(f"#{it['id']}: {it['text']}")
    elif a.cmd == "list":
        for it in mem.all():
            print(f"#{it['id']}: {it['text']}  {it.get('tags')}")
