# Mind Web Agent — İnternet Erişimi 🌐

**Tarih:** 2026-06-08
**Durum:** ✅ Çalışıyor (MVP) — vizyonun 3. parçası (web/tool katmanı)

## Önemli: kim internete eriştir?
Tiny from-scratch Mind (14M, masal-eğitimli) internete **erişemez** — talimat takip / araç çağırma kapasitesi yok, sadece metin devam ettirir. İnternet yeteneği **yetenekli beyne (lokal qwen2.5:3b)** verildi. Bu, vizyondaki "pluggable beyin" fikrinin ta kendisi: bugün Qwen ajan, ileride daha güçlü model.

## Nasıl çalışıyor (RAG)
`mindllm/agent.py`:
1. **Ara:** Soru → Wikipedia arama API'si (key gerektirmez, TR+EN)
2. **Oku:** En alakalı başlıkların özetlerini çek (canlı internet)
3. **Cevapla:** Qwen, bu bağlamı kullanarak soruyu cevaplar (grounded)

```bash
python -m mindllm.agent "İstanbul'un nüfusu kaç?"
python -m mindllm.agent --lang en "What is a black hole?"
```

## Çalışan örnekler
**"İstanbul'un nüfusu kaç ve hangi ülkede?"**
> 🧠 İstanbul'un nüfusu yaklaşık 15 milyondan fazla ve Türkiye'de yer alır.
> 📚 Kaynak: İstanbul, 1831 Osmanlı nüfus sayımı

**"What is a black hole?" (EN)**
> 🧠 A black hole is a region in space where the gravitational pull is so strong that nothing, not even light, can escape...
> 📚 Kaynak: Hawking radiation, Black hole information paradox

## Dürüst sınırlar
- **Arama kalitesi ifadeye bağlı:** tam cümle bazen Wikipedia'da eşleşmez. "Mona Lisa tablosunu kim yaptı ve nerede sergileniyor?" → bağlam bulunamadı, Qwen kendi bilgisine düştü ve sergilendiği yeri **yanlış** söyledi (halüsinasyon).
- Bağlam bulununca güvenilir; bulunamayınca model bilgisi (hatalı olabilir).
- Sadece Wikipedia (DuckDuckGo flaky 202 döndü). Genel web araması için ileride bir arama API'si (Brave/SerpAPI key) eklenebilir.

## İyileştirme fikirleri (gelecek)
- Sorudan anahtar kelime çıkarımı (tam cümle yerine) → daha iyi Wikipedia eşleşmesi
- "Bağlam yoksa cevaplama / bilmiyorum de" modu (halüsinasyonu kes)
- Çok kaynak (Wikipedia + haber + genel arama)

## Dashboard ile birleşim
Bu backend; dashboard (diğer oturum) sonucu gösterir. Dashboard `from mindllm.agent import ask; ask("soru")` çağırıp cevabı + kaynakları render edebilir.

## Yeni kod
- `mindllm/agent.py` (wiki_search, wiki_summary, web_context, build_rag_prompt, ask) — `distill.py`'ın `ollama_generate`'ini yeniden kullanır
