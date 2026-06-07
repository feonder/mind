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
                      max_iters=300, batch_size=16, eval_interval=50)
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
    train_main(str(data_dir), out, cfg=cfg, max_iters=100, batch_size=16, eval_interval=50)
    import json
    meta = json.load(open(tmp_path / "out" / "meta.json"))
    assert meta["iter"] == 100
    # resume: 100'den 150'ye devam
    hist = train_main(str(data_dir), out, cfg=cfg, max_iters=150, batch_size=16,
                      eval_interval=50, resume=True)
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
                      max_iters=200, batch_size=16, eval_interval=50)
    assert hist[-1][1] < hist[0][1]
