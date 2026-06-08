import os
import json
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mindllm.model import GPT, GPTConfig, count_params
from mindllm.data import get_batch, load_bin


def build_config(tokenizer_path=None, n_layer=None, n_head=None, n_embd=None,
                 block_size=None):
    """GPTConfig'i opsiyonel override'larla kurar. tokenizer verilirse vocab_size
    oradan alınır; n_layer/n_head/n_embd/block_size None değilse override edilir."""
    overrides = {}
    if tokenizer_path:
        from mindllm.bpe import BPETokenizer
        overrides["vocab_size"] = BPETokenizer.load(tokenizer_path).vocab_size
    for key, val in (("n_layer", n_layer), ("n_head", n_head),
                     ("n_embd", n_embd), ("block_size", block_size)):
        if val is not None:
            overrides[key] = val
    return GPTConfig(**overrides)


def loss_fn(model, x, y):
    logits = model(x)
    B, T, V = logits.shape
    return nn.losses.cross_entropy(
        logits.reshape(B * T, V), y.reshape(B * T), reduction="mean"
    )


def make_step(model, optimizer):
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        return loss

    return step


def _decode_sample(tok, x, n=48):
    """O anki batch'in ilk örneğinden ~n token'ı metne çevirir (code rain içeriği)."""
    try:
        ids = [int(t) for t in x[0][:n].tolist()]
        if tok is not None:
            text = tok.decode(ids)
        else:
            text = bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")
        return text.replace("\n", " ").strip()[:200]
    except Exception:
        return ""


def _write_status(path, out_dir, it, max_iters, loss, running,
                  tok_per_sec=0, loss_history=None, sample=""):
    """Dashboard için canlı eğitim durumu: yüzde, loss geçmişi, hız, o anki metin örneği."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump({
        "model": os.path.basename(out_dir.rstrip("/")),
        "iter": it, "max_iters": max_iters,
        "percent": round(100.0 * it / max_iters, 1) if max_iters else 0.0,
        "loss": round(loss, 4), "running": running,
        "tok_per_sec": tok_per_sec,
        "loss_history": loss_history or [],
        "sample": sample,
    }, open(path, "w"))


def train_main(data_dir, out_dir, cfg=None, max_iters=2000, batch_size=32,
               eval_interval=200, lr=3e-4, resume=False, throttle=0.0,
               status_path="out/train_status.json", live_interval=5, tokenizer_path=None):
    """throttle: adım başına bekleme (ısı/fan). status_path: canlı durum dosyası.
    live_interval: kaç adımda bir canlı telemetri (loss geçmişi/hız/örnek metin) yaz.
    tokenizer_path: 'code rain' için o anki batch'i metne çevirmekte kullanılır."""
    cfg = cfg or GPTConfig()
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "ckpt.safetensors")
    meta_path = os.path.join(out_dir, "meta.json")

    train_data = load_bin(os.path.join(data_dir, "train.bin"))

    model = GPT(cfg)
    start_iter = 0
    if resume and os.path.exists(ckpt):
        model.load_weights(ckpt)
        start_iter = json.load(open(meta_path))["iter"]
        print(f"resume: iter {start_iter}'den devam")
    mx.eval(model.parameters())

    opt = optim.AdamW(learning_rate=lr)
    step = make_step(model, opt)
    rng = np.random.default_rng(1337)
    history = []

    # canlı telemetri (code rain) kurulumu
    from collections import deque
    sample_tok = None
    if tokenizer_path:
        try:
            from mindllm.bpe import BPETokenizer
            sample_tok = BPETokenizer.load(tokenizer_path)
        except Exception:
            sample_tok = None
    loss_hist = deque(maxlen=40)
    tps_per_step = batch_size * cfg.block_size
    t0 = time.time()
    seen = 0

    for it in range(start_iter, max_iters):
        x, y = get_batch(train_data, cfg.block_size, batch_size, rng)
        loss = step(x, y)
        if throttle:
            time.sleep(throttle)
        seen += tps_per_step
        if it % live_interval == 0:
            l = float(loss)
            loss_hist.append(round(l, 3))
            dt = time.time() - t0
            tps = int(seen / dt) if dt > 0 else 0
            sample = _decode_sample(sample_tok, x)
            _write_status(status_path, out_dir, it + 1, max_iters, l, True,
                          tok_per_sec=tps, loss_history=list(loss_hist), sample=sample)
        if it % eval_interval == 0 or it == max_iters - 1:
            l = float(loss)
            history.append((it, l))
            print(f"iter {it}: loss {l:.4f}")
            model.save_weights(ckpt)
            json.dump({"iter": it + 1, "config": vars(cfg)}, open(meta_path, "w"))

    final_loss = history[-1][1] if history else 0.0
    _write_status(status_path, out_dir, max_iters, max_iters, final_loss, running=False)
    return history


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="mindllm/data")
    p.add_argument("--out_dir", default="out")
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--tokenizer", default=None, help="BPE tokenizer.json yolu; vocab_size buradan alınır")
    p.add_argument("--throttle", type=float, default=0.0, help="adım başına saniye bekleme (fan/ısı düşürür)")
    p.add_argument("--n_layer", type=int, default=None)
    p.add_argument("--n_head", type=int, default=None)
    p.add_argument("--n_embd", type=int, default=None)
    p.add_argument("--block_size", type=int, default=None)
    p.add_argument("--live_interval", type=int, default=5, help="canlı telemetri sıklığı (adım)")
    a = p.parse_args()
    cfg = build_config(tokenizer_path=a.tokenizer, n_layer=a.n_layer,
                       n_head=a.n_head, n_embd=a.n_embd, block_size=a.block_size)
    print(f"model: vocab {cfg.vocab_size}, {cfg.n_layer}L/{cfg.n_head}H/{cfg.n_embd}d, "
          f"~{count_params(GPT(cfg)) / 1e6:.1f}M param")
    train_main(a.data_dir, a.out_dir, cfg=cfg, max_iters=a.max_iters,
               batch_size=a.batch_size, eval_interval=a.eval_interval,
               lr=a.lr, resume=a.resume, throttle=a.throttle,
               live_interval=a.live_interval, tokenizer_path=a.tokenizer)
