import mlx.core as mx
from mindllm.model import GPT, GPTConfig, count_params


def test_forward_shape():
    cfg = GPTConfig(vocab_size=256, block_size=32, n_layer=2, n_head=2, n_embd=64)
    model = GPT(cfg)
    mx.eval(model.parameters())
    x = mx.zeros((2, 16), dtype=mx.int32)
    logits = model(x)
    assert logits.shape == (2, 16, 256)


def test_default_config_is_about_10m():
    cfg = GPTConfig()  # varsayılan = ~10M
    model = GPT(cfg)
    n = count_params(model)
    assert 8_000_000 < n < 13_000_000, f"param sayısı {n}"


def test_checkpoint_roundtrip(tmp_path):
    cfg = GPTConfig(vocab_size=256, block_size=16, n_layer=2, n_head=2, n_embd=64)
    m1 = GPT(cfg)
    mx.eval(m1.parameters())
    p = str(tmp_path / "ckpt.safetensors")
    m1.save_weights(p)
    m2 = GPT(cfg)
    m2.load_weights(p)
    x = mx.zeros((1, 8), dtype=mx.int32)
    assert mx.allclose(m1(x), m2(x)).item()
