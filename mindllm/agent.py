"""Mind web agent — canlı internet (Wikipedia) + yetenekli beyin (Qwen) = RAG.

Tiny from-scratch Mind internete erişemez (talimat takip / araç çağırma
kapasitesi yok). Bu katman, internet yeteneğini lokal Qwen2.5:3b'ye verir:
Wikipedia'dan canlı bilgi çeker, Qwen o bilgiyle soruyu cevaplar.

Kullanım:
    python -m mindllm.agent "İstanbul kaç nüfuslu?"
    python -m mindllm.agent --lang en "Who founded Turkey?"
"""
import json
import urllib.parse
import urllib.request

from mindllm.distill import ollama_generate

_UA = {"User-Agent": "MindAgent/0.1 (https://github.com/feonder/mind)"}


def _get_json(url, timeout=10):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wiki_search(query, lang="tr", limit=3, timeout=10):
    """Wikipedia'da arar, en alakalı başlıkları döner."""
    params = urllib.parse.urlencode({
        "action": "query", "list": "search", "srsearch": query,
        "format": "json", "srlimit": limit,
    })
    url = f"https://{lang}.wikipedia.org/w/api.php?{params}"
    data = _get_json(url, timeout=timeout)
    return [hit["title"] for hit in data.get("query", {}).get("search", [])]


def wiki_summary(title, lang="tr", timeout=10):
    """Bir Wikipedia başlığının özet metnini döner."""
    url = (f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/"
           f"{urllib.parse.quote(title)}")
    try:
        data = _get_json(url, timeout=timeout)
    except Exception:
        return ""
    return data.get("extract", "")


def web_context(query, lang="tr", n=2, timeout=10):
    """Sorgu için Wikipedia'dan bağlam toplar. (titles, context_text) döner."""
    titles = wiki_search(query, lang=lang, limit=n, timeout=timeout)
    parts, used = [], []
    for t in titles:
        s = wiki_summary(t, lang=lang, timeout=timeout)
        if s:
            parts.append(f"## {t}\n{s}")
            used.append(t)
    return used, "\n\n".join(parts)


def build_rag_prompt(question, context, lang="tr"):
    """Bağlam + soru'dan Qwen prompt'u kurar (saf fonksiyon, test edilebilir)."""
    if not context:
        if lang == "tr":
            return f"Soru: {question}\nKısa ve net bir cevap ver."
        return f"Question: {question}\nGive a short, clear answer."
    if lang == "tr":
        instr = ("Aşağıdaki Wikipedia bilgisini kullanarak soruyu kısa, doğru ve "
                 "Türkçe cevapla. Bilgi yetersizse bunu belirt.")
        q_lbl, a_lbl = "SORU", "CEVAP"
    else:
        instr = ("Using the Wikipedia information below, answer the question "
                 "concisely and accurately. If it is insufficient, say so.")
        q_lbl, a_lbl = "QUESTION", "ANSWER"
    return (f"{instr}\n\n=== WIKIPEDIA ===\n{context}\n\n"
            f"=== {q_lbl} ===\n{question}\n\n=== {a_lbl} ===")


def ask(question, lang="tr", model="qwen2.5:3b", n=2):
    """İnternetten (Wikipedia) bilgi çekip Qwen ile cevaplar. dict döner."""
    used, context = web_context(question, lang=lang, n=n)
    prompt = build_rag_prompt(question, context, lang=lang)
    answer = ollama_generate(prompt, model=model, num_predict=350, temperature=0.3)
    return {"question": question, "answer": answer.strip(), "sources": used}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("question", help="sorulacak soru")
    p.add_argument("--lang", default="tr", choices=["tr", "en"])
    p.add_argument("--model", default="qwen2.5:3b")
    a = p.parse_args()
    res = ask(a.question, lang=a.lang, model=a.model)
    print(f"\n🧠 {res['answer']}\n")
    if res["sources"]:
        print(f"📚 Kaynak (Wikipedia): {', '.join(res['sources'])}")
    else:
        print("📚 (Wikipedia'da bağlam bulunamadı — modelin kendi bilgisi)")
