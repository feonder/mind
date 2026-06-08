# Daha İyi İngilizce Mind — Sonuç 🇬🇧

**Tarih:** 2026-06-08 | **Durum:** ✅ En tutarlı Mind (şimdiye kadar)

## Ne yapıldı
F2 (20k hikâye, 3.49M token) → **60k hikâye, ~10M token (~3x)**. Aynı 13.9M model, aynı BPE; sadece çok daha fazla veri + 6000 iter.

## Sayılar — aynı model, daha çok veri
| | F2 | İngilizce-big |
|---|---|---|
| Hikâye | 20k | **60k** |
| Token | 3.49M | **~10M** |
| Eğitim | 5000 iter | 6000 iter (~5 epoch) |
| Loss | ~3.25 | **2.82** |
| Çıktı | tutarlı | **en akıcı** |

> Kritik: loss 2.82 **ezber değil** (5 epoch, sağlıklı). F3/TR'deki 0.0x ezberin aksine, burada model gerçekten genelliyor.

## Örnek (prompt "Once upon a time")
> ...his family went on a big adventure. When the sun began to set, it began to rain... He went to them and said, "Let's go for a drive!" ... Everyone cheered. The driver was very proud of his work. He thanked his family...

Çok-paragraflı, tutarlı olay örgüsü, diyalog, karakter tutarlılığı. Küçük-model kaymaları var ama en akıcı Mind.

## Ders — F4'ün tam tersi
- **F4:** büyük model (91M) + az veri → KÖTÜ (veri-aç)
- **Bu:** aynı model + 3x veri → DAHA İYİ
- **Sonuç:** küçük donanımda doğru ölçekleme = **daha çok/kaliteli veri**, model boyutu değil. (Chinchilla dengesinin pratik kanıtı.)

## Konuşma
`python -m mindllm.chat --model story` (artık bu en iyi model) · `--model story1` (eski F2)
