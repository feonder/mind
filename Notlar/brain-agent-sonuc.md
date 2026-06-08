# Mind Brain — Genel Araç-Kullanan Agent 🧠🔧

**Tarih:** 2026-06-08
**Durum:** ✅ Çalışıyor — "her şeye yetkin" mimarisinin doğru hali

## Asıl içgörü
Tiny Mind'ı "her şeye yetkin" eğitemeyiz (kapasite sınırı — F4 kanıtı). Ama araçları
**tek tek hardcode** da etmeyiz. Doğru çözüm: **genel araç-kullanan agent**.
- Yetkinlik zaten **Qwen'de eğitilmiş** (Alibaba onu araç kullanımına/akıl yürütmeye eğitti).
- Bizim işimiz: bir **araç kayıt defteri** + agent döngüsü. Beyin soruya göre **hangi aracı kullanacağına KENDİ karar verir.**
- **Yeni yetenek = TOOLS'a yeni araç eklemek.** Per-tool if/else yok.

## Mimari (`mindllm/brain.py`)
```
soru → choose_tool (Qwen JSON seçer) → aracı çalıştır → Qwen sonuçla cevaplar
```
- `TOOLS` kayıt defteri: her araç = {açıklama, fonksiyon}
- `choose_tool`: beyne araç menüsünü gösterir, `{"tool":..,"args":..}` JSON seçtirir
- `run`: aracı çalıştırır, sonucu beyne verir, nihai cevabı alır
- `generate_fn` enjekte edilebilir (test için, GPU'suz)

## Mevcut araçlar
| Araç | Kaynak | Örnek |
|---|---|---|
| `weather` | wttr.in (key'siz, canlı) | "İstanbul'da hava?" |
| `wikipedia` | Wikipedia API (TR/EN) | "Atatürk kimdir?" |
| `none` | doğrudan beyin | genel sohbet |

## Canlı kanıt (beyin aracı kendi seçti)
- "İstanbul'da hava bugün nasıl?" → **[weather]** → "güneşli, 27°C, nem %54, rüzgar 13 km/s" ✅
- "Atatürk kimdir?" → **[wikipedia]** → "Mustafa Kemal Atatürk, Türkiye Cumhuriyeti'nin kurucusu..." ✅
- "Ankara'da hava kaç derece?" → **[weather]** → "25°C" ✅

## Kullanım
```bash
python -m mindllm.brain "İstanbul'da hava nasıl?"
python -m mindllm.brain "Mona Lisa'yı kim yaptı?"
```

## Genişletmek için (gelecek — sadece araç ekle)
TOOLS'a ekle, beyin otomatik kullanır:
- `calculator` (hesap), `time` (saat/tarih), `web_search` (genel arama, key'li API),
  `code_run` (kod çalıştır), `files` (dosya oku/yaz), `memory` (uzun hafıza recall)
- Çok adımlı görevler için: agent döngüsünü "araç → sonuç → tekrar araç seç" şeklinde
  birden fazla tura çıkar (ReAct loop).

## Neden bu "doğru" yaklaşım
Kullanıcının sezgisi haklıydı: araç-bazlı değil, genel yetkinlik. Bunu tiny Mind'ı
eğiterek değil, **araç-eğitimli beyni (Qwen) bir araç ekosistemine koyarak** sağladık.
Mind sistemi büyüdükçe yetenekler araç olarak eklenir, beyin orkestra eder.
