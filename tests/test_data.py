import numpy as np
import mlx.core as mx
from mindllm.data import get_batch


def test_get_batch_shapes():
    data = np.arange(1000, dtype=np.uint16)
    x, y = get_batch(data, block_size=16, batch_size=8)
    assert x.shape == (8, 16)
    assert y.shape == (8, 16)


def test_y_is_x_shifted_by_one():
    # arange verisinde y[k] her zaman x[k] + 1 olmalı (sonraki token)
    data = np.arange(1000, dtype=np.uint16)
    x, y = get_batch(data, block_size=8, batch_size=4)
    assert mx.all((y - x) == 1).item()


def test_encode_texts_byte():
    from mindllm.data import encode_texts
    from mindllm.tokenizer import ByteTokenizer
    arr = encode_texts(["AB", "C"], ByteTokenizer())
    # "AB"+"\n"+"C"+"\n" => [65, 66, 10, 67, 10]
    assert list(arr) == [65, 66, 10, 67, 10]
    assert arr.dtype == np.uint16


def test_encode_texts_bpe_ids_in_range():
    from mindllm.data import encode_texts
    from mindllm.bpe import BPETokenizer
    tok = BPETokenizer().train("hello world " * 100, vocab_size=300)
    arr = encode_texts(["hello world"], tok)
    assert arr.dtype == np.uint16
    assert arr.max() < tok.vocab_size
