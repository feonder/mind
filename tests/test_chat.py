import json

import mlx.core as mx

from mindllm.model import GPT, GPTConfig
from mindllm.bpe import BPETokenizer
from mindllm.chat import load_model, respond, PRESETS


def _make_checkpoint(tmp_path):
    """Test için minik bir model + meta + BPE tokenizer kurar."""
    tok = BPETokenizer().train("once upon a time there was a cat " * 80, vocab_size=300)
    tp = str(tmp_path / "tok.json")
    tok.save(tp)
    out = tmp_path / "out"
    out.mkdir()
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=64)
    m = GPT(cfg)
    mx.eval(m.parameters())
    m.save_weights(str(out / "ckpt.safetensors"))
    json.dump({"iter": 1, "config": vars(cfg)}, open(out / "meta.json", "w"))
    return str(out), tp, tok.vocab_size


def test_presets_well_formed():
    assert "story" in PRESETS
    for name, (out_dir, tok_path) in PRESETS.items():
        assert isinstance(out_dir, str)
        assert tok_path is None or isinstance(tok_path, str)


def test_load_model_returns_model_tok_cfg(tmp_path):
    out_dir, tp, vocab = _make_checkpoint(tmp_path)
    model, tok, cfg = load_model(out_dir, tp)
    assert cfg.vocab_size == vocab
    assert tok.vocab_size == vocab
    assert model is not None


def test_respond_returns_text(tmp_path):
    out_dir, tp, _ = _make_checkpoint(tmp_path)
    model, tok, cfg = load_model(out_dir, tp)
    text = respond(model, tok, cfg, "once", max_new_tokens=5, temperature=0.8)
    assert isinstance(text, str) and len(text) >= 1


def test_respond_empty_prompt_returns_empty(tmp_path):
    out_dir, tp, _ = _make_checkpoint(tmp_path)
    model, tok, cfg = load_model(out_dir, tp)
    assert respond(model, tok, cfg, "") == ""
