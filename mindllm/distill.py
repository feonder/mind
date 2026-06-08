"""F3 distillation: bir öğretmen modelden (Ollama, örn. qwen2.5:3b) sentetik
eğitim verisi üretir. Mind bu veriyle eğitilir (data-seviyesi distillation)."""
import json
import os
import time
import urllib.request

TOPICS = [
    "a lost puppy", "a magic tree", "sharing toys", "a rainy day",
    "a brave little mouse", "making a new friend", "a trip to the beach",
    "a kind dragon", "learning to share", "a birthday surprise",
    "helping mom in the kitchen", "a star that fell down", "a curious cat",
    "a garden of flowers", "a snowy morning", "a little red boat",
    "finding a tiny treasure", "a sleepy bear", "the moon and the sea",
    "a singing bird", "a broken toy", "a walk in the park",
]

NAMES = ["Tom", "Lily", "Ben", "Mia", "Sam", "Anna", "Leo", "Zoe", "Max", "Ella"]

TR_TOPICS = [
    "kaybolan bir kedi", "sihirli bir ağaç", "oyuncak paylaşmak", "yağmurlu bir gün",
    "cesur küçük bir fare", "yeni bir arkadaş edinmek", "deniz kenarına gezi",
    "nazik bir ejderha", "paylaşmayı öğrenmek", "doğum günü sürprizi",
    "annesine mutfakta yardım etmek", "gökten düşen bir yıldız", "meraklı bir köpek",
    "rengarenk bir çiçek bahçesi", "karlı bir sabah", "küçük kırmızı bir tekne",
    "küçük bir hazine bulmak", "uykucu bir ayı", "ay ve deniz",
    "şarkı söyleyen bir kuş", "kırık bir oyuncak", "parkta bir yürüyüş",
]

TR_NAMES = ["Ayşe", "Mehmet", "Elif", "Can", "Zeynep", "Ali", "Defne", "Emir", "Mavi", "Deniz"]


def build_prompt(i, lang="en"):
    """index'e göre çeşitli (karakter + konu) hikâye prompt'u üretir. lang: en|tr."""
    if lang == "tr":
        topic = TR_TOPICS[i % len(TR_TOPICS)]
        name = TR_NAMES[i % len(TR_NAMES)]
        return (
            "4 yaşındaki bir çocuk için çok kısa, basit bir Türkçe masal yaz "
            "(yaklaşık 90 kelime), sadece kolay ve günlük kelimeler kullan. "
            f"Ana kahramanın adı {name}. Masal {topic} hakkında. "
            "Sadece masal metnini yaz, başlık veya ek açıklama yazma."
        )
    topic = TOPICS[i % len(TOPICS)]
    name = NAMES[i % len(NAMES)]
    return (
        "Write a very short, simple children's story (about 90 words) for a "
        f"4-year-old, using only easy common words. The main character is named {name}. "
        f"The story is about {topic}. Write only the story text, no title, no commentary."
    )


def ollama_generate(prompt, model="qwen2.5:3b", host="http://localhost:11434",
                    num_predict=220, temperature=0.9, timeout=300):
    payload = {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"num_predict": num_predict, "temperature": temperature},
    }
    req = urllib.request.Request(
        host + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))["response"]


def generate_corpus(out_file, n=400, model="qwen2.5:3b", sleep=1.5, temperature=0.9,
                    lang="en"):
    """n adet sentetik hikâye üretip out_file'a yazar. Çağrılar arası `sleep`
    saniye bekler (GPU'ya nefes → daha az ısı/fan). lang: en|tr."""
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    written = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for i in range(n):
            try:
                text = ollama_generate(build_prompt(i, lang=lang), model=model,
                                       temperature=temperature)
            except Exception as e:
                print(f"skip {i}: {e}")
                time.sleep(sleep)
                continue
            text = text.strip()
            if text:
                f.write(text + "\n\n")
                f.flush()
                written += 1
                print(f"{written}/{n} ({len(text)} char)")
            time.sleep(sleep)
    print(f"distill korpusu: {written} hikâye -> {out_file}")
    return written


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="out/distill_corpus.txt")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--model", default="qwen2.5:3b")
    p.add_argument("--sleep", type=float, default=1.5)
    p.add_argument("--lang", default="en", choices=["en", "tr"])
    a = p.parse_args()
    generate_corpus(a.out, n=a.n, model=a.model, sleep=a.sleep, lang=a.lang)
