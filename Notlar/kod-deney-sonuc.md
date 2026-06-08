# Kod Modeli Deneyi — Sonuç 🐍

**Tarih:** 2026-06-08
**Durum:** ✅ Başarılı (kanıt/öğrenme amaçlı)

## Ne yapıldı
Mind modeli **gerçek Python kodu** üstünde eğitildi. Kod korpusu makinedeki `.py` dosyalarından toplandı (Python stdlib + venv site-packages) — internet/dataset indirme yok, tamamen offline.

## Pipeline
1. `mindllm/code_corpus.py` → 203 `.py` dosyası, 5.02M karakter topladı (`out/code_corpus.txt`)
2. Kod-BPE'si eğitildi (vocab 4096) → `out/tokenizer_code.json` (`"def foo(x): return x+1"` = 29 byte → 8 token)
3. Tokenize → `mindllm/data_code` (train 1.1M token, val 58k)
4. Model eğitildi: 5000 iter, sessiz mod (batch 16 + throttle 0.3) → `out_code`

## Sayılar
| | Değer |
|---|---|
| Korpus | 203 dosya, 5.02M karakter (stdlib + site-packages) |
| Tokenizer | kod-BPE, vocab 4096 |
| Model | ~13.7M param (vocab 4096) |
| Eğitim | 5000 iter, sessiz mod |
| **Loss** | **8.51 → 1.64** (kod tekrarlı/yapısal → prozadan düşük) |

## Üretilen kod örnekleri

**prompt: `def `**
```python
def user_link(self, user_site=None):
        """Run my method to occur in its own on password (if any),
        otherwise return default)."""
        return user_function, user_user_site)
    def user_sitepackages(self, user_site):
        ...
        _CONFIG_VARS['userbase'] = 'stdlibwin32'
    return os.environ
```

**prompt: `class `**
```python
class _mboxMMDFMessage(archive):
        if isinstance(extension, str):
            self._file_size = self._file_length
        else:
            ...
        self._file = os.path.abspath(file)
        except:
            if self._file_length & 1: os.stat(file_size & 0x0)
        self._file_size   = 0xffff
        self._locked = True
```

## Değerlendirme
Model gerçek Python idiomlarını öğrendi: `def ... self, ...=None`, docstring, girinti, `return`, `import`, `__all__ = [...]`, `class X(...)`, `if isinstance`, `try/except`, `self._attr`, `os.path.abspath`, hex literal (`0xffff`). Eğitildiği stdlib modüllerinin (shutil, mailbox, sysconfig, site) izleri net görülüyor.

**Sınırlar (beklenen):** Çalışan program yazamaz — mantık bozuk, parantez dengesiz, tekrara giriyor (10M model + düşük temperature). "Kod gibi görünen ama bozuk." Gerçek kod yazma yeteneği F4+ ve kod-güçlü öğretmenlerden distillation ile gelir.

## Çıktı dosyaları
- `out/code_corpus.txt`, `out/tokenizer_code.json`
- `out_code/ckpt.safetensors`, `mindllm/data_code/` (gitignore)

## Yeni kod
- `mindllm/code_corpus.py` (lokal `.py` toplama)
- `bpe.py` & `data.py`: `--text_file` (herhangi bir korpustan eğit/tokenize)
