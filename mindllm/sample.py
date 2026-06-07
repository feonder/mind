import mlx.core as mx

from mindllm.model import GPT, GPTConfig
from mindllm.tokenizer import ByteTokenizer


def generate(model, idx, max_new_tokens: int, block_size: int, temperature: float = 1.0):
    """idx: (B, T) int32. Autoregressive olarak max_new_tokens token ekler."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] * (1.0 / max(temperature, 1e-6))
        next_id = mx.random.categorical(logits).reshape(idx.shape[0], 1)
        idx = mx.concatenate([idx, next_id.astype(idx.dtype)], axis=1)
        mx.eval(idx)
    return idx


def sample_main(ckpt, out_dir, prompt="Once upon a time", max_new_tokens=200,
                temperature=0.8, tokenizer_path=None):
    import json
    meta = json.load(open(f"{out_dir}/meta.json"))
    cfg = GPTConfig(**meta["config"])
    model = GPT(cfg)
    model.load_weights(ckpt)
    mx.eval(model.parameters())
    if tokenizer_path:
        from mindllm.bpe import BPETokenizer
        tok = BPETokenizer.load(tokenizer_path)
    else:
        tok = ByteTokenizer()
    idx = mx.array([tok.encode(prompt)], dtype=mx.int32)
    out = generate(model, idx, max_new_tokens, cfg.block_size, temperature)
    text = tok.decode([int(t) for t in out[0].tolist()])
    print(text)
    return text


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="out")
    p.add_argument("--ckpt", default="out/ckpt.safetensors")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--tokenizer", default=None, help="BPE tokenizer.json yolu; yoksa byte-level")
    a = p.parse_args()
    sample_main(a.ckpt, a.out_dir, a.prompt, a.max_new_tokens, a.temperature,
                tokenizer_path=a.tokenizer)
