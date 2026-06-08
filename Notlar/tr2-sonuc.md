# Türkçe Mind v2 — Sonuç 🇹🇷

**Tarih:** 2026-06-08 | **Durum:** ✅ Eğitildi (v1'den daha çok veri)

## Ne değişti
v1 (800 hikâye, 57k token) → **v2: 2300 hikâye, 168k token** (~3x). Qwen ile 1500 yeni Türkçe masal üretilip eskiyle birleştirildi; yeni Türkçe BPE + model eğitildi.

## Sayılar
| | v1 | v2 |
|---|---|---|
| Hikâye | 800 | **2300** |
| Token | 57k | **168k** |
| Eğitim | 3000 iter | 5000 iter |
| Loss | 0.075 (ağır ezber) | **0.18** (daha az ezber, daha çeşitli) |

## Örnek (prompt "Bir varmış bir yokmuş")
> Bir varmış bir yokmuş. Defne'nin evinde küçük bir kedi için... Ejderha unu kaçırmaz... Emir çocuğunu ayağa kalktı... Masal: Deniz odaya uzanan... Ayşe o yıl... Sonunda güzel bir...

## Değerlendirme
v2 daha çeşitli (daha az kelimesi kelimesine ezber) ama **hâlâ cümle düzeyinde dağınık.** Türkçe kelimeler, isimler (Defne, Emir, Ayşe, Ejderha), masal kalıpları doğru; anlamsal tutarlılık zayıf.

**Neden hâlâ ham:** (1) 13.9M model küçük, (2) Qwen2.5:3b'nin Türkçesi mükemmel değil (öğretmen sınırı), (3) 168k token hâlâ az. Üçü birden tavanı belirliyor.

**Gerçekten iyi Türkçe için:** çok daha fazla veri (gerçek Türkçe korpus — Wikipedia-tr/OSCAR-tr, milyonlarca token) + daha büyük model + daha iyi öğretmen.

## Konuşma
`python -m mindllm.chat --model tr` (v2) · `--model tr1` (v1)
