"""Mind brain — genel araç-kullanan agent.

Tiny Mind genel-yetkin OLAMAZ (kapasite sınırı). Yetkinlik, araç-kullanımına
eğitilmiş beyne (qwen2.5:3b) + bir ARAÇ KAYIT DEFTERİ'ne dayanır. Beyin, soruya
göre hangi aracı kullanacağına KENDİ karar verir; sonucu alıp cevaplar.

Yeni yetenek eklemek = TOOLS'a yeni bir araç eklemek. Per-tool if/else yok —
beyin ne zaman kullanacağını seçer (ReAct / function-calling mantığı).
"""
import json
import re
import urllib.parse
import urllib.request

from mindllm.distill import ollama_generate
from mindllm.agent import web_context


# ---- Araçlar (her biri: açıklama + fonksiyon) ----

def tool_weather(location, **_):
    """Bir şehrin güncel hava durumu (wttr.in, key'siz)."""
    url = f"https://wttr.in/{urllib.parse.quote(location)}?format=j1"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.loads(r.read().decode("utf-8"))
    c = d["current_condition"][0]
    return (f"{location}: {c['temp_C']}°C, {c['weatherDesc'][0]['value']}, "
            f"nem %{c['humidity']}, rüzgar {c['windspeedKmph']} km/s")


def tool_wikipedia(query, lang="tr", **_):
    """Bir konu/kişi/yer hakkında ansiklopedik bilgi (Wikipedia)."""
    _titles, ctx = web_context(query, lang=lang, n=2)
    return ctx


TOOLS = {
    "weather": {
        "desc": 'Bir şehrin GÜNCEL/CANLI hava durumu. args: {"location": "şehir"}',
        "fn": tool_weather,
    },
    "wikipedia": {
        "desc": 'Bir konu, kişi, yer hakkında ansiklopedik bilgi. args: {"query": "konu"}',
        "fn": tool_wikipedia,
    },
    "none": {
        "desc": "Araç gerekmez; beyin doğrudan cevaplar. args: {}",
        "fn": None,
    },
}


def _tool_menu():
    return "\n".join(f"- {name}: {t['desc']}" for name, t in TOOLS.items())


def _parse_choice(raw):
    """Beynin çıktısından {"tool":..,"args":..} JSON'unu ayıklar."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"tool": "none", "args": {}}
    try:
        d = json.loads(m.group(0))
        tool = d.get("tool", "none")
        args = d.get("args", {}) or {}
        if tool not in TOOLS:
            tool = "none"
        return {"tool": tool, "args": args if isinstance(args, dict) else {}}
    except Exception:
        return {"tool": "none", "args": {}}


def choose_tool(question, generate_fn, model="qwen2.5:3b"):
    """Beyne araçları gösterir, hangisini kullanacağını seçtirir."""
    prompt = (
        "Aşağıdaki soruya en iyi cevabı vermek için hangi aracı kullanırsın?\n"
        f"Araçlar:\n{_tool_menu()}\n\n"
        f"Soru: {question}\n\n"
        'SADECE şu JSON ile cevap ver, başka hiçbir şey yazma: '
        '{"tool":"<araç>","args":{...}}'
    )
    raw = generate_fn(prompt, model=model, num_predict=120, temperature=0.1)
    return _parse_choice(raw)


def run(question, model="qwen2.5:3b", generate_fn=None):
    """Genel agent döngüsü: araç seç → çalıştır → sonuçla cevapla."""
    gen = generate_fn or ollama_generate
    choice = choose_tool(question, gen, model=model)
    tool, args = choice["tool"], choice["args"]
    result = ""
    if tool in TOOLS and TOOLS[tool]["fn"] is not None:
        try:
            result = TOOLS[tool]["fn"](**args)
        except Exception as e:
            result = f"(araç hatası: {e})"
    if result:
        prompt = (f"'{tool}' aracı şu sonucu verdi:\n{result}\n\n"
                  f"Bu bilgiyi kullanarak soruyu kısa ve Türkçe cevapla.\n"
                  f"Soru: {question}\nCevap:")
    else:
        prompt = f"Soruyu kısa ve Türkçe cevapla.\nSoru: {question}\nCevap:"
    answer = gen(prompt, model=model, num_predict=300, temperature=0.3).strip()
    return {"question": question, "tool": tool, "args": args,
            "result": result, "answer": answer}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Mind brain — araç-kullanan agent")
    p.add_argument("question")
    p.add_argument("--model", default="qwen2.5:3b")
    a = p.parse_args()
    res = run(a.question, model=a.model)
    print(f"\n[araç: {res['tool']}]")
    print(f"\n🧠 {res['answer']}\n")
