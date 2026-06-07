import os
import json

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mindllm.model import GPT, GPTConfig
from mindllm.data import get_batch, load_bin


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


def train_main(data_dir, out_dir, cfg=None, max_iters=2000, batch_size=32,
               eval_interval=200, lr=3e-4, resume=False):
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

    for it in range(start_iter, max_iters):
        x, y = get_batch(train_data, cfg.block_size, batch_size, rng)
        loss = step(x, y)
        if it % eval_interval == 0 or it == max_iters - 1:
            l = float(loss)
            history.append((it, l))
            print(f"iter {it}: loss {l:.4f}")
            model.save_weights(ckpt)
            json.dump({"iter": it + 1, "config": vars(cfg)}, open(meta_path, "w"))

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
    a = p.parse_args()
    train_main(a.data_dir, a.out_dir, max_iters=a.max_iters, batch_size=a.batch_size,
               eval_interval=a.eval_interval, lr=a.lr, resume=a.resume)
