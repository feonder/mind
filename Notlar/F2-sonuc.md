# F2 Sonuç — Sıfırdan BPE Tokenizer ✅

**Tarih:** 2026-06-08
**Faz:** F2 (kendi BPE tokenizer'ımız + tokenizer-agnostik pipeline)
**Durum:** ✅ Başarılı

## Ne yapıldı
Byte-level (vocab 256) tokenizer yerine sıfırdan yazılmış **BPE tokenizer** (`mindllm/bpe.py`, GPT-2 tarzı byte-level BPE) eklendi. Pipeline tokenizer-agnostik yapıldı (`--tokenizer` ile byte veya BPE). Model BPE ile yeniden eğitilip F1 ile karşılaştırıldı.

## Sayılar
| | F1 (byte) | F2 (BPE) |
|---|---|---|
| Vocab | 256 | **4096** |
| Model param | 10.9M | ~13.7M (büyük embedding) |
| Hikâye başına token | ~894 | ~175 |
| **Sıkışma** | — | **~5x** |
| 256-token context'te metin | 256 karakter | **~1300 karakter (~5x)** |
| Eğitim | 5000 iter, byte | 5000 iter, BPE (sessiz mod: batch 16 + throttle) |

> Loss sayıları doğrudan kıyaslanamaz (256 vs 4096 sınıf üstünde cross-entropy). Anlamlı kıyas: **context'te görülen metin** ve **çıktı tutarlılığı**.

## BPE tokenizer detayı
- `BPETokenizer`: train (en sık çifti tekrar tekrar birleştir) / encode / decode / save / load — tamamen sıfırdan, sıfır bağımlılık
- 2MB TinyStories alt kümesinde eğitildi, vocab 4096 → `out/tokenizer.json`
- `"Once upon a time, there was a little cat."` = 41 byte → **4 token** (çok sık geçen ifade tek parçaya birleşmiş)
- Tüm UTF-8 işler (Türkçe dahil), round-trip testleri geçiyor

## Çıktı karşılaştırması (prompt: "Once upon a time")
**F1 (byte):**
> ...Timmy's mom made him a big bit stand too. It was the family looked for him to read it.

**F2 (BPE):**
> ...a little bird named Bob... they would play in the warm sand and they were having so much fun together. When they arrived at the beach, Sophia looked...

**Gözlem:** F2 belirgin daha tutarlı — daha uzun akıcı cümleler, korunan olay örgüsü. Sebep: BPE ile model aynı context'te ~5x daha fazla metin gördüğü için uzun menzilli yapıyı öğrendi. Hâlâ kusurlu (isimler kayıyor, uydurma kelimeler) ama F1'den net ileri.

## Doğrulama
- 22 birim testi geçiyor (F1'in 14'ü + BPE 4 + data 2 + train 1 + sample 1)
- İki aşamalı review geçti: spec uyumu ✅ (padding hile'si tespit edilip kaldırıldı) + kod kalitesi ✅ (BPE algoritması bağımsız referansla doğrulandı)
- Bonus: `--throttle` eklendi (ısı/fan kontrolü için adım başına sleep)

## Çıktı dosyaları
- `out/tokenizer.json` — eğitilmiş BPE (vocab 4096)
- `out_bpe/ckpt.safetensors` — BPE ile eğitilmiş model
- `mindllm/data_bpe/` — BPE token'lanmış veri (gitignore)

## Sonraki
- **Kod deneyi:** lokal Python kodu üstünde model (devam ediyor)
- **F3:** qwen2.5:3b ile distillation kanıtı
- **F4:** 124M model
