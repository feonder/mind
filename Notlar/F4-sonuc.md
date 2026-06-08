# F4 Sonuç — 91M Model + Önemli Ölçek Dersi 📊

**Tarih:** 2026-06-08
**Durum:** ✅ Eğitildi — ve beklenmedik ama öğretici bir sonuç verdi

## Ne yapıldı
GPT-2-small sınıfı **~91M parametreli** model (12 katman / 12 head / 768 embedding, vocab 4096) TinyStories BPE verisinde eğitildi. `train.py`'a model-boyutu CLI argümanları (`--n_layer/--n_head/--n_embd`, `build_config`) eklendi.

> Not: Spec'teki hedef "124M" idi; vocab 4096 ile bu boyut ~91M'e denk gelir (124M, ~50k'lık GPT-2 vocab'ıyla olur). İsim olarak "F4/124M" ama gerçek ~91M.

## Sayılar
| | F2 (10M) | F4 (91M) |
|---|---|---|
| Parametre | 13.7M | **91.5M** |
| Veri | 3.49M token | 3.49M token (aynı) |
| Eğitim | 5000 iter, batch 16 | 5000 iter, batch 8 |
| Görülen token | ~20M | ~11M |
| **Final loss** | **~3.25** | **~4.86** |
| Çıktı tutarlılığı | Daha iyi | **Daha ham** |

## ⚠️ Sonuç: büyük model DAHA KÖTÜ çıktı — ve bu doğru

F4 örneği (prompt "Once upon a time"):
> Once upon a time. Once upon a tim. She was a little girl who was very serious stage in a big forest. Timmy was very curious and asked, "What did you dove her?"... giggled and cross a limakes...

F2'nin (10M) aynı prompt'taki çıktısından **daha bozuk.** Sebep tek kelimeyle: **veri açlığı (undertraining).**

### Neden? Chinchilla matematiği
- Optimal eğitim ≈ **20 token / parametre**.
- 91M model için gereken: ~**1.8 milyar token**.
- Verdiğimiz: ~**11M token** (gerekenin ~%0.6'sı).
- Sonuç: 91M model kapasitesinin çok altında kaldı; loss 4.86'da takıldı (F2 10M, aynı veriyle 3.25'e indi çünkü küçük model o veriye "doydu").

### Ders
**"Daha büyük model = daha iyi" YANLIŞ — veri/compute sabitken.** Büyük model ancak orantılı veri + eğitim ile parlar. Bu, spec'in 2. bölümündeki compute duvarının canlı kanıtı: M4'te 91M'i doğru eğitmek için 1.8B token × birkaç epoch = günler/haftalar gerekir.

## Doğru yol (gelecek)
91M'i hak ettiği gibi eğitmek için:
1. Çok daha fazla veri (tüm TinyStories ~500M token + sentetik distillation verisi)
2. Çok daha fazla iterasyon (on binler/yüz binler)
3. Sabır (M4'te günler) veya daha güçlü donanım

Yani F4 "başarısız" değil — **doğru kurulmuş bir deneyin dürüst sonucu:** küçük donanımda büyük modelin sınırını gösterdi.

## Yeni kod
- `train.py`: `build_config` + `--n_layer/--n_head/--n_embd/--block_size` CLI

## Çıktı dosyaları
- `out_124m/ckpt.safetensors` (349MB), `out_124m/train.log`

## Şarj kesintisi notu
Eğitim gece şarj bitince Mac uykuya geçti, fişe takılınca **kaldığı yerden otomatik devam etti** (checkpoint + uyku/uyanma). Veri kaybı olmadı.
