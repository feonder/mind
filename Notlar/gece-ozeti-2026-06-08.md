# 🌙 Gece Özeti — 2026-06-08 (otonom çalışma)

Emir uyurken Mind LLM'in 4 fazı + 1 deney otonom olarak tamamlandı. Hepsi `main`'de, GitHub'da (feonder/mind).

## Tamamlananlar

| İş | Sonuç | Not |
|---|---|---|
| **F2 — Sıfırdan BPE** | ✅ | vocab 4096, ~5x sıkışma; çıktı F1'den **daha tutarlı** |
| **Kod deneyi** | ✅ | Lokal Python kodunda eğitim; `def/class/import/try` idiomları öğrenildi |
| **F3 — Distillation** | ✅ | qwen2.5:3b → 400 sentetik hikâye → Mind; öğretmen→öğrenci pipeline kanıtı |
| **F4 — 91M model** | ✅ | Eğitildi ama **veri-aç** → 10M'den ham çıktı (öğretici ölçek dersi) |

## En önemli üç öğrenme

1. **Tokenizer kalitesi büyük fark yaratıyor (F2):** byte-level → BPE geçişi, aynı model boyutunda belirgin daha tutarlı metin. Sebep: BPE ile context'te ~5x fazla bilgi.

2. **Distillation gerçek (F3):** Lokal Qwen, Mind'a veri üretip onu eğitebiliyor. Çıktıda Qwen'in karakterleri/temaları görüldü. Küçük ölçekte ama kanıtlandı.

3. **Büyük ≠ iyi, veri sabitken (F4):** 91M model, 10M'den KÖTÜ çıktı çünkü ~%0.6 oranında veri aldı (Chinchilla: 1.8B token gerek, 11M verildi). "Daha büyük model" ancak orantılı veri+compute ile işe yarar — spec'teki compute duvarının canlı kanıtı.

## Loss karşılaştırması (kıyas dikkatli yorumlanmalı, farklı vocab/veri)
| Model | Param | Veri | Final loss | Çıktı |
|---|---|---|---|---|
| F1 (byte) | 10.9M | 44.7M byte-token | 0.86 | tutarlı-ish |
| F2 (BPE) | 13.7M | 3.49M BPE-token | ~3.25 | **en tutarlı** |
| Kod | 13.7M | 1.1M token | 1.64 | kod-şeklinde |
| F3 (distill) | 13.7M | 22k token | 0.14 (ezber) | Qwen stili |
| F4 (91M) | 91.5M | 3.49M token | 4.86 | ham (veri-aç) |

## Teknik altyapı (gece eklenenler)
- `mindllm/bpe.py` — sıfırdan BPE tokenizer
- `mindllm/code_corpus.py` — lokal Python kodu toplama
- `mindllm/distill.py` — Ollama öğretmenden sentetik veri
- `train.py` — `--throttle` (ısı/fan kontrolü) + `--n_layer/--n_head/--n_embd` (build_config)
- `bpe.py`/`data.py` — `--text_file` (herhangi bir korpustan eğit/tokenize)
- 32 birim testi, hepsi geçiyor

## Operasyonel notlar
- **Sessiz mod:** Fan sesi için tüm eğitimlere throttle (adım başına 0.3sn) + nice + sınırlı thread uygulandı.
- **Şarj kesintisi:** Mac gece şarj bitince uyudu, fişe takılınca F4 checkpoint'ten otomatik devam etti — veri kaybı yok.
- **Disk:** ~5GB (4.4GB'ı HuggingFace TinyStories cache'i — `rm -rf ~/.cache/huggingface` ile geri kazanılabilir).

## Sıradaki gerçek adımlar (öneri)
1. **F4'ü hak ettiği gibi eğit:** tüm TinyStories (~500M token) + çok daha fazla iter (günler/haftalar veya daha güçlü donanım).
2. **F3'ü ölçekle:** çok-öğretmen (DeepSeek, Gemma `ollama pull`) + saatlerce sentetik üretim.
3. **Vizyonun sonraki parçaları:** dashboard, web/tool katmanı, terminal (her biri ayrı spec).
