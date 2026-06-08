# Mind Yetenekleri — Kodlama, Proje, Hafıza 🛠️

**Tarih:** 2026-06-08
**Durum:** ✅ İki track de tamam

İstek: "kendi verinle eğit, kodlama geliştir, proje başlatma, proje bağlam kurma, uzun hatırlama" → ikisi birden.

## Track 1 — Gerçek agent yetenekleri (Qwen beyni + araçlar)
Tiny Mind bunları yapamaz; yetenek, yetenekli beyne (Qwen) + araçlara verildi.

| Yetenek | Modül | Ne yapar |
|---|---|---|
| **Uzun hatırlama** | `mindllm/memory.py` | Kalıcı JSON bellek; oturumlar arası hatırlar. Kök eşleşmeli recall (Türkçe çekimleri yakalar: "fonksiyon"↔"fonksiyona") |
| **Proje bağlamı** | `mindllm/project.py` | Kod tabanını tarar → dosya ağacı + anahtar dosya başlıkları → beyne bağlam |
| **Proje başlatma** | `mindllm/coder.py` `scaffold()` | python/web proje iskeleti oluşturur |
| **Kodlama** | `mindllm/coder.py` `code_assist()` | Qwen + hafıza + proje bağlamı → kod sorusu cevaplar, etkileşimi hatırlar |

**Kullanım:**
```bash
python -m mindllm.coder new projem --kind python   # proje başlat
python -m mindllm.coder ask "ortalama fonksiyonu yaz" --root .
python -m mindllm.memory add "Bu projede MLX kullanıyoruz" --tags kural
python -m mindllm.memory recall "framework"
```

**Canlı demo:** coder agent `calculate_average` fonksiyonunu doğru yazıp Türkçe açıkladı. Hafıza, geçmiş kuralları hatırlayıp prompt'a ekliyor.

## Track 2 — Tiny Mind'ı kod verisiyle eğit ("kendi verinle")
| | Değer |
|---|---|
| Öğretmen | qwen2.5:3b → 500 kodlama örneği (`distill.py --mode code`) |
| Korpus | 67KB, temiz Python fonksiyonları |
| Tokenizer | kod-instruct BPE (vocab 4050) |
| Eğitim | erken durduruldu (loss 8.62 → 0.03, ezber) → `out_codeinstruct` |

**Çıktı (prompt "# "):** geçerli Python fonksiyonları — `factorial`, `binary_search`, `is_prime`, `reverse_string`, `filter_even_numbers`... Temiz veriyle eğitildiği için kod modelinden (stdlib) çok daha "doğru" görünüyor (ama ezber: eğitim örneklerini tekrarlıyor).

## Dürüst sınırlar
- **Tiny Mind** kod *üretiyor* ama gerçek kodlama asistanı değil — küçük korpusu ezberledi. Gerçek yetenek Track 1'de (Qwen).
- **Hafıza recall** kök eşleşmeli ama yine de basit (embedding yok) — anlamsal eşleşmede sınırlı.
- **code_assist** proje bağlamı + hafıza kullanıyor; daha güçlü olması için dosya yazma/komut çalıştırma araçları + iteratif agent döngüsü eklenebilir (gelecek).

## Yeni kod
- `mindllm/memory.py`, `mindllm/project.py`, `mindllm/coder.py` (+ `distill.py --mode code`)
- 54+ test (memory 6, project 3, coder 4, distill 6, ...)
