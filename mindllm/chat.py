"""Mind ile interaktif sohbet (REPL).

Mind bir *metin-devam* modelidir: yazdığın metni öğrendiği tarzda sürdürür
(soru-cevap / chat DEĞİL). En iyi sonuç için İngilizce, hikâye tarzı başlat.

Kullanım:
    python -m mindllm.chat                  # varsayılan: story (F2/out_bpe)
    python -m mindllm.chat --model code     # kod modeli
    python -m mindllm.chat --model big      # F4 91M
"""
import argparse
import json

import mlx.core as mx

from mindllm.model import GPT, GPTConfig
from mindllm.sample import generate
from mindllm.tokenizer import ByteTokenizer

# (out_dir, tokenizer_path) — tokenizer_path None ise byte-level
PRESETS = {
    "story": ("out_bpe_big", "out/tokenizer.json"),   # en iyi İngilizce (60k hikâye, 3x veri)
    "story1": ("out_bpe", "out/tokenizer.json"),      # F2 (20k hikâye)
    "tr": ("out_tr2", "out/tokenizer_tr2.json"),      # Türkçe model v2 (2300 hikâye)
    "tr1": ("out_tr", "out/tokenizer_tr.json"),       # Türkçe v1 (800 hikâye)
    "code": ("out_code", "out/tokenizer_code.json"),  # Python kodu
    "big": ("out_124m", "out/tokenizer.json"),        # F4 91M (ham)
    "distill": ("out_distill", "out/tokenizer.json"),  # F3 Qwen-distilled
    "f1": ("out", None),                              # byte-level 10M
}


def load_model(out_dir, tokenizer_path):
    """Bir checkpoint dizininden model + tokenizer + config yükler."""
    with open(f"{out_dir}/meta.json") as f:
        meta = json.load(f)
    cfg = GPTConfig(**meta["config"])
    model = GPT(cfg)
    model.load_weights(f"{out_dir}/ckpt.safetensors")
    mx.eval(model.parameters())
    if tokenizer_path:
        from mindllm.bpe import BPETokenizer
        tok = BPETokenizer.load(tokenizer_path)
    else:
        tok = ByteTokenizer()
    return model, tok, cfg


def respond(model, tok, cfg, prompt, max_new_tokens=120, temperature=0.8):
    """Bir prompt'tan üretilen tam metni (prompt dahil) döner."""
    ids = tok.encode(prompt)
    if not ids:
        return ""
    idx = mx.array([ids], dtype=mx.int32)
    out = generate(model, idx, max_new_tokens, cfg.block_size, temperature)
    return tok.decode([int(t) for t in out[0].tolist()])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="story", choices=list(PRESETS),
                   help="hazır model seçimi (varsayılan: story = F2)")
    p.add_argument("--out_dir", default=None, help="elle checkpoint dizini (preset'i ezer)")
    p.add_argument("--tokenizer", default=None, help="elle tokenizer.json (preset'i ezer)")
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    a = p.parse_args()

    out_dir, tok_path = PRESETS[a.model]
    if a.out_dir:
        out_dir = a.out_dir
    if a.tokenizer:
        tok_path = a.tokenizer

    print(f"Mind yükleniyor ({a.model}: {out_dir}) ...")
    model, tok, cfg = load_model(out_dir, tok_path)
    print(f"Hazır — vocab {cfg.vocab_size}, context {cfg.block_size} token.")
    print("Bir şeyler yaz, Mind devam ettirsin. Çıkış: boş satır / 'exit' / Ctrl-C.\n")

    while True:
        try:
            prompt = input("sen> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\ngörüşürüz 👋")
            break
        if not prompt or prompt.lower() in ("exit", "quit", "q", "çık"):
            print("görüşürüz 👋")
            break
        text = respond(model, tok, cfg, prompt,
                       max_new_tokens=a.max_new_tokens, temperature=a.temperature)
        print(f"\nMind> {text}\n")


if __name__ == "__main__":
    main()
