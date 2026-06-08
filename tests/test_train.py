import numpy as np
import mlx.core as mx
import mlx.optimizers as optim
from mindllm.model import GPT, GPTConfig
from mindllm.train import make_step, train_main


def test_overfit_single_batch_drops_loss():
    cfg = GPTConfig(vocab_size=256, block_size=16, n_layer=2, n_head=2, n_embd=64)
    model = GPT(cfg)
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=1e-3)
    step = make_step(model, opt)
    rng = np.random.default_rng(0)
    d = rng.integers(0, 256, size=(4, 17)).astype(np.int32)
    x, y = mx.array(d[:, :16]), mx.array(d[:, 1:17])
    losses = [float(step(x, y)) for _ in range(200)]
    assert losses[-1] < losses[0] * 0.5


def test_end_to_end_tiny(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Öğrenilebilir tekrarlı desen → loss düşmeli
    arr = np.tile(np.arange(50, dtype=np.uint16), 3000)
    arr.tofile(str(data_dir / "train.bin"))
    arr.tofile(str(data_dir / "val.bin"))
    cfg = GPTConfig(vocab_size=256, block_size=32, n_layer=2, n_head=2, n_embd=64)
    hist = train_main(str(data_dir), str(tmp_path / "out"), cfg=cfg,
                      max_iters=300, batch_size=16, eval_interval=50,
                      status_path=str(tmp_path / "st.json"))
    assert hist[-1][1] < hist[0][1]
    assert (tmp_path / "out" / "ckpt.safetensors").exists()
    assert (tmp_path / "out" / "meta.json").exists()


def test_resume_continues_iteration(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    arr = np.tile(np.arange(50, dtype=np.uint16), 3000)
    arr.tofile(str(data_dir / "train.bin"))
    arr.tofile(str(data_dir / "val.bin"))
    cfg = GPTConfig(vocab_size=256, block_size=32, n_layer=2, n_head=2, n_embd=64)
    out = str(tmp_path / "out")
    train_main(str(data_dir), out, cfg=cfg, max_iters=100, batch_size=16, eval_interval=50,
               status_path=str(tmp_path / "st.json"))
    import json
    meta = json.load(open(tmp_path / "out" / "meta.json"))
    assert meta["iter"] == 100
    # resume: 100'den 150'ye devam
    hist = train_main(str(data_dir), out, cfg=cfg, max_iters=150, batch_size=16,
                      eval_interval=50, resume=True, status_path=str(tmp_path / "st.json"))
    assert hist[0][0] >= 100


def test_train_with_custom_vocab(tmp_path):
    import numpy as np
    from mindllm.model import GPTConfig
    from mindllm.train import train_main
    d = tmp_path / "data"
    d.mkdir()
    arr = np.tile(np.arange(60, dtype=np.uint16), 3000)  # id'ler < 300
    arr.tofile(str(d / "train.bin"))
    arr.tofile(str(d / "val.bin"))
    cfg = GPTConfig(vocab_size=300, block_size=32, n_layer=2, n_head=2, n_embd=64)
    hist = train_main(str(d), str(tmp_path / "out"), cfg=cfg,
                      max_iters=200, batch_size=16, eval_interval=50,
                      status_path=str(tmp_path / "st.json"))
    assert hist[-1][1] < hist[0][1]


def test_build_config_overrides_dims():
    from mindllm.train import build_config
    cfg = build_config(n_layer=12, n_head=12, n_embd=768)
    assert cfg.n_layer == 12 and cfg.n_head == 12 and cfg.n_embd == 768
    assert cfg.vocab_size == 256  # tokenizer verilmedi → default


def test_build_config_defaults():
    from mindllm.train import build_config
    cfg = build_config()
    assert cfg.n_layer == 6 and cfg.n_embd == 384 and cfg.vocab_size == 256


def test_write_status_file(tmp_path):
    import json
    from mindllm.train import _write_status
    p = str(tmp_path / "status.json")
    _write_status(p, "out_demo", 50, 200, 1.234, running=True)
    d = json.load(open(p))
    assert d["percent"] == 25.0
    assert d["iter"] == 50 and d["max_iters"] == 200
    assert d["model"] == "out_demo" and d["running"] is True


def test_decode_sample_bytes():
    import mlx.core as mx
    from mindllm.train import _decode_sample
    x = mx.array([[72, 105, 32, 77, 105, 110, 100]], dtype=mx.int32)  # "Hi Mind"
    assert "Hi" in _decode_sample(None, x)
