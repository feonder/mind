# F3 Sonuç — Distillation (öğretmen → öğrenci) 🎓

**Tarih:** 2026-06-08
**Durum:** ✅ Pipeline kanıtlandı (küçük ölçek)

## Ne yapıldı
Knowledge distillation'ın **data-seviyesi** biçimi: öğretmen model (Ollama **qwen2.5:3b**) sentetik eğitim verisi üretti, Mind o veriyle eğitildi. Yani Mind, Qwen'in "öğrettiği" içeriği/stili öğrendi.

## Pipeline
1. `mindllm/distill.py` → Qwen2.5:3b ile **400 sentetik çocuk hikâyesi** üretti (çeşitli karakter+konu prompt'ları, çağrı arası mola = sessiz). Üretim hızı ~3.5sn/hikâye.
2. Korpus: 400 hikâye, 22k kelime (`out/distill_corpus.txt`)
3. Tokenize: TinyStories BPE (F2 ile kıyaslanabilir olsun diye aynı tokenizer) → 22k token
4. Mind eğitildi: 2000 iter, sessiz mod → `out_distill`

## Sayılar
| | Değer |
|---|---|
| Öğretmen | qwen2.5:3b (Ollama, lokal) |
| Sentetik korpus | 400 hikâye / 22k kelime / ~22k BPE token |
| Eğitim | 2000 iter, sessiz mod |
| **Loss** | **8.48 → 0.14** (küçük korpus → ezberleme) |

## Üretilen örnek (Qwen-distilled Mind)
> Once upon a time... Anna loved to play in her backyard. One sunny day, she found a big shiny leaves... Max saw tiny fairies. Everyone sang happily as she became best friends forever... Mia found a magic tree outside her house...

**Gözlem:** Çıktıda Qwen'in damgası net — karakter isimleri (Anna, Max, Mia, Lily, Ben) ve prompt temaları (backyard, magic tree, fairies, shells, "best friends forever", ice cream) doğrudan öğretmenin ürettiği hikâyelerden geliyor. **Distillation çalıştı:** öğretmenin bilgisi öğrenciye aktı.

## Dürüst sınırlar
- Korpus çok küçük (22k token, F2'nin 3.49M'inin ~%0.6'sı) → model ezberledi (loss 0.14), çıktı F2'den daha bozuk.
- Bu bir **pipeline kanıtı**, kalite sıçraması değil. Gerçek F3 değeri için: çok daha fazla sentetik veri (saatlerce/günlerce Qwen üretimi) + çok-öğretmen (DeepSeek, Gemma de eklenir) + daha büyük model.
- Tek öğretmen kullanıldı (sadece qwen2.5:3b mevcuttu). Çok-öğretmen için `ollama pull` ile model eklenip `distill.py --model` ile karıştırılabilir.

## Kanıtlanan
Öğretmen→öğrenci veri akışı uçtan uca çalışıyor: **lokal LLM (Qwen) → sentetik müfredat → Mind eğitimi → Mind öğretmenin içeriğini üretiyor.** Vizyondaki "Qwen Mind'ı eğitebilir mi?" sorusunun yanıtı pratikte: evet.

## Yeni kod
- `mindllm/distill.py` (Ollama öğretmenden sentetik veri üretimi)

## Çıktı dosyaları
- `out/distill_corpus.txt`, `out_distill/ckpt.safetensors`, `mindllm/data_distill/`
