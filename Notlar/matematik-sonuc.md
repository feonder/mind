# Matematik — İki Yaklaşım, Net Ders 🔢

**Tarih:** 2026-06-08 | **Durum:** ✅ İkisi de yapıldı (paralel: GPU matematik + CPU İngilizce tokenize)

## Track 2 — Tiny matematik Mind (eğitim)
Qwen 500 matematik örneği üretti (`distill.py --mode math`) → kod-BPE → 13.8M model eğitildi (loss 8.5 → **0.017 = ezber**).

**Tanıdık problemlerde iyi görünüyor (ezberden doğru):**
> Find half of 8 → 4 ✓ · 5+3=8 ✓ · 24 rounded → 20 ✓ · multiples of 5: 10,15,20 → 45 ✓

**Ama YENİ problemde çuvallıyor:**
> "What is 17 + 28?" → (45 yerine) ezberlediği alakasız bir perimeter problemini kustu ❌

**Ders:** Tiny model matematiği **ezberliyor, hesaplamıyor.** Format mükemmel, tanıdık girdide doğru, ama yeni girdide başarısız — gerçek hesaplama/akıl yürütme yok.

## Track 1 — Calculator aracı (gerçek matematik) ✅
`brain.py`'a güvenli aritmetik aracı (`tool_calculator`, ast tabanlı — kod çalıştırmaz) eklendi. Beyin (Qwen) matematik sorusu görünce bu aracı seçer.

> "What is 17 + 28?" → **[araç: calculator] → 17 + 28 = 45** ✓ (gerçek, her sayı için doğru)

## Net sonuç
| | Tiny math Mind | Brain + calculator |
|---|---|---|
| Tanıdık problem | ✓ (ezber) | ✓ |
| **Yeni problem** | ❌ çöp | ✅ doğru |
| Yöntem | model eğitimi | araç |

**Genel kural (tüm yeteneklerde aynı):** Yetkinlik tiny modeli eğitmekten değil, **araç-kullanan beyinden** gelir. Matematik = calculator, hava = weather, bilgi = wikipedia. Yeni yetenek = yeni araç; beyin seçer.

## Yeni kod
- `distill.py --mode math` (matematik verisi üretimi)
- `brain.py` calculator aracı (TOOLS'a eklendi) — artık beyin: calculator/weather/wikipedia/none
