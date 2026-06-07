import mlx.core as mx
from mindllm.model import GPT, GPTConfig
from mindllm.sample import generate


def test_generate_extends_sequence():
    cfg = GPTConfig(vocab_size=256, block_size=16, n_layer=2, n_head=2, n_embd=64)
    model = GPT(cfg)
    mx.eval(model.parameters())
    idx = mx.array([[1, 2, 3]], dtype=mx.int32)
    out = generate(model, idx, max_new_tokens=10, block_size=16)
    assert out.shape == (1, 13)


def test_generate_respects_block_size():
    # prompt block_size'dan uzun olsa bile çalışmalı (pencere kayar)
    cfg = GPTConfig(vocab_size=256, block_size=8, n_layer=2, n_head=2, n_embd=64)
    model = GPT(cfg)
    mx.eval(model.parameters())
    idx = mx.zeros((1, 20), dtype=mx.int32)
    out = generate(model, idx, max_new_tokens=5, block_size=8)
    assert out.shape == (1, 25)


def test_generated_tokens_in_vocab():
    cfg = GPTConfig(vocab_size=256, block_size=16, n_layer=2, n_head=2, n_embd=64)
    model = GPT(cfg)
    mx.eval(model.parameters())
    idx = mx.array([[5]], dtype=mx.int32)
    out = generate(model, idx, max_new_tokens=8, block_size=16)
    assert mx.all(out >= 0).item() and mx.all(out < 256).item()
