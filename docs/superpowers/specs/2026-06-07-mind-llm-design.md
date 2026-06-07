# Mind LLM — Tasarım Dokümanı (Spec)

**Tarih:** 2026-06-07
**Durum:** Onay bekliyor
**Kapsam:** Sadece *Mind LLM* alt-projesi. Dashboard, terminal, web/tool katmanı ayrı spec'ler olacak.

---

## 1. Amaç ve Vizyon

**Mind**, lokal-öncelikli, tek-UI'lı, AI-sürücülü kişisel bir bilgi sisteminin uzun vadeli vizyonudur. Bu spec, o vizyonun **kalbini** tanımlar: sıfırdan yazılmış, M4'te çalışan, eğitilebilir bir dil modeli (LLM) ve onun uçtan uca eğitim pipeline'ı.

**Asıl hedef öğrenme ve sahiplik:** tokenizer'dan eğitim döngüsüne kadar her parçayı kendimiz yazarız. Rekabetçi kalite hedef değil; "tamamen benim modelim" hissi ve çalışan, anlaşılır bir sistem hedeftir.

**Kalite sıçraması için strateji — distillation:** Küçük model ham web verisiyle eğitilirse "aptal" kalır. Bunun yerine büyük açık modelleri (Qwen, DeepSeek, Gemma) **öğretmen** olarak kullanırız; onların ürettiği yüksek kaliteli sentetik veriyle modeli eğitiriz. Precedent: Microsoft Phi ("Textbooks Are All You Need") — küçük model + kaliteli sentetik veri, kendinden kat kat büyük modelleri geçti.

### Vizyondaki yeri (bağlam, bu spec'in kapsamı dışında)
| Sıra | Parça | Rol |
|---|---|---|
| 0 (hazır) | Bilgi tabanı (bu Obsidian vault) | Notlar/projeler organize |
| **1 (BU SPEC)** | **Mind LLM** | Sıfırdan eğitilen, öğretmen-damıtmalı model |
| 2 | Generative Dashboard | Tek UI, beyne göre şekillenir (pluggable beyin) |
| 3 | Web/tool katmanı | "x videosu" → oynar; sonuç gösterir |
| 4 | Kendini geliştiren terminal | Agent'ın eli-ayağı |

Dashboard'un beyni **pluggable** olacak: bugün Qwen sürükler (gerçekten çalışır), Mind olgunlaştıkça yerine takılır.

---

## 2. Donanım Gerçeği (kararları belirleyen kısıt)

- **Apple M4 (base), 16GB unified memory, 10 çekirdek.**
- Öğretmen modeli çalıştırma: 7B'ye kadar quantize rahat (Qwen2.5-7B-Q4 ≈ 4.5GB).
- Sıfırdan eğitim gerçekçi tavanı: **~124M–350M parametre.** 1B, 16GB'de gerçekçi değil (optimizer state'leri RAM'i doldurur) → "ileride daha büyük makinede" rafında.

### Compute gerçeği (neden küçük?)
Eğitim maliyeti ≈ `6 × parametre × token`. Chinchilla-optimal ≈ 20 token/parametre.
| Model | Token | M4'te kaba süre | Sonuç |
|---|---|---|---|
| 10M | ~300M | saatler | Tutarlı-ish, dil yapısını öğrenir |
| 124M | ~2.5B | birkaç gün | Tanınabilir GPT-2 seviyesi |
| 1B | ~20B | aylar + bellek baskısı | 16GB'de gerçekçi değil |
| 8B | ~15T | ~2000+ yıl | Evde imkânsız |

---

## 3. Teknik Yığın

- **Dil:** Python (temiz `venv`; MLX uyumu için gerekirse Python 3.12).
- **Framework:** **MLX** — Apple'ın native ML framework'ü. M4 GPU'sunu doğrudan kullanır, PyTorch+MPS'ten hızlı ve sürtünmesiz.
- **Felsefe:** az dosya, okunabilir, "magic" yok. nanoGPT ruhu, MLX'te.
- **Checkpoint formatı:** safetensors.

---

## 4. Mimari

Decoder-only transformer (GPT mimarisi). Her birim tek işlikli ve bağımsız test edilebilir.

```
mind-llm/
├── tokenizer/   metin ↔ token (F1: hazır BPE; F2: kendi BPE'miz)
├── data/        indir → tokenize → train.bin/val.bin shard → batch üret
├── model/       embedding → N× transformer blok (attention + MLP) → çıkış head
├── train/       eğitim döngüsü: loss, AdamW, checkpoint, log, resume
├── sample/      checkpoint → metin üret (test + eğlence)
├── configs/     boyut tarifleri (10M.yaml, 124M.yaml)
└── tests/       her birim için birim testleri
```

### Birim sözleşmeleri
| Birim | Girdi | Çıktı | Bağımlılık |
|---|---|---|---|
| tokenizer | metin | token id listesi (ve tersi) | — |
| data | ham metin + tokenizer | train.bin / val.bin, batch iterator | tokenizer |
| model | token batch | logits | — |
| train | model + data + config | checkpoint + log | model, data |
| sample | checkpoint + tokenizer + prompt | üretilen metin | model, tokenizer |

### Veri akışı
```
ham metin → tokenizer → token id → data(train.bin/val.bin)
   → train(batch → model → loss → backprop → checkpoint)
   → sample(checkpoint → prompt → üretilen metin)
```

### 10M referans konfigürasyonu (F1)
~6 katman, ~6 head, ~384 embedding boyutu, ~256 context uzunluğu, vocab ~8k. 16GB'ye fazlasıyla sığar.

---

## 5. Fazlar (yol haritası)

| Faz | Hedef | Veri | Tamamlanma kriteri |
|---|---|---|---|
| **F1** | 10M model uçtan uca çalışsın | TinyStories (hazır, indirilebilir) | Loss düşüyor; model tutarlı-ish İngilizce hikâye üretiyor |
| **F2** | Kendi BPE tokenizer'ımız | — | Kendi tokenizer'ımızla F1 tekrar çalışıyor (round-trip testi geçiyor) |
| **F3** | Çok-öğretmenli sentetik veri | Qwen/DeepSeek/Gemma üretir | "Bizim" müfredatımız hazır; modelin kalitesi ham veriye göre ölçülebilir şekilde iyi |
| **F4** | 124M'ye büyüt | F3 verisi | "Mind v1": tanınabilir kaliteli üretim |

**Neden TinyStories ile başlıyoruz (F1):** 10M gibi minik modellerin bile tutarlı metin üretmesini sağlayan kanıtlanmış set; kendisi de GPT ile üretilmiş sentetik veri (felsefemize uygun). Pipeline'ı hızlı ve ucuz doğrular. F3'te kendi öğretmenlerimize geçeriz.

### Çok-öğretmenli distillation (F3 detayı)
- **Veri seviyesinde birleştirme (kullanacağımız yöntem — kolay):** her öğretmen ayrı sentetik veri üretir, karıştırılır, model bu karışık müfredatla eğitilir. Sadece metin olduğu için tokenizer uyuşmazlığı sorun değil.
- **Logit seviyesinde birleştirme (kullanmayacağız — zor):** farklı tokenizer'lar nedeniyle vocab hizalama baş ağrısı.
- **Uzmanına göre görev dağıtımı:** DeepSeek → akıl yürütme; Qwen → Türkçe + genel; Gemma → kısa-öz.

---

## 6. Hata Yönetimi

- **Checkpoint + resume:** eğitim her N adımda checkpoint yazar; kaldığı yerden devam edebilir (uzun eğitimler için şart).
- **OOM koruması:** bellek dolarsa batch boyutunu otomatik küçült / net hata mesajı ver.
- **Deterministik tohum (seed):** tekrarlanabilir eğitim.
- **Veri doğrulama:** shard'lar bozuksa/eksikse eğitim başlamadan yakala.
- **Net loglar:** loss, hız (token/s), bellek kullanımı periyodik raporlanır.

---

## 7. Test Stratejisi

TDD (superpowers:test-driven-development). Her birim için:
- **tokenizer:** round-trip testi (metin → token → metin aynı kalmalı).
- **data:** shard bütünlüğü; batch şekli doğru.
- **model:** forward geçiş doğru şekil/boyut üretir.
- **train:** tek batch'e overfit edilince loss düşmeli (öğrenme kanıtı).
- **sample:** checkpoint'ten geçerli token üretiyor.

---

## 8. Başarı Kriteri (F1 — ilk somut hedef)

1. `mind-llm/` pipeline'ı uçtan uca çalışıyor: indir → tokenize → eğit → örnekle.
2. 10M model TinyStories üstünde eğitiliyor; **loss eğrisi düşüyor.**
3. Eğitilmiş modelden örnek alındığında **tutarlı-ish İngilizce hikâye** çıkıyor (mükemmel değil, ama rastgele değil — öğrendiği belli).
4. Tüm birim testleri geçiyor.
5. Bir checkpoint'ten eğitim durdurulup devam ettirilebiliyor.

---

## 9. Kapsam Dışı (YAGNI)

- Dashboard, terminal, web/tool katmanı (ayrı spec'ler).
- 1B+ model (donanım yetmiyor).
- RLHF / fine-tuning UI / dağıtık eğitim.
- Logit-seviyesi multi-teacher distillation (gereksiz karmaşık).
