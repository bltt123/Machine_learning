"""Toy 复制/翻转 + CLS 分类 + LM 计数 + 去噪 + 词表（05 复用 04 的 S=6 设定，便于与 04-03 对比）。"""
import random
import numpy as np

__all__ = ["make_copy_data", "make_reverse_data", "build_vocab", "make_cls_data", "make_lm_data", "make_denoise_data"]


def make_copy_data(n=600, seq_len=6, vocab_size=8, seed=0):
    rng = random.Random(seed)
    _ = np.array(0)
    X = [[rng.randint(0, vocab_size - 1) for _ in range(seq_len)] for _ in range(n)]
    return X, [row[:] for row in X]


def make_reverse_data(n=600, seq_len=6, vocab_size=8, seed=1):
    rng = random.Random(seed)
    X = [[rng.randint(0, vocab_size - 1) for _ in range(seq_len)] for _ in range(n)]
    Y = [row[::-1] for row in X]
    return X, Y


def build_vocab(vocab_size):
    """id 词表（toy 用整数 id，无需字符串）。"""
    return list(range(vocab_size))


def make_cls_data(n=800, seq_len=8, vocab_size=12, seed=0):
    """Binary CLS toy：y=1 当且仅当序列含 token 0。约 49% 正例，平衡。"""
    rng = random.Random(seed)
    X = [[rng.randint(0, vocab_size - 1) for _ in range(seq_len)] for _ in range(n)]
    y = [1 if 0 in row else 0 for row in X]
    return X, y


def make_lm_data(n=1000, seq_len=8, vocab_size=8, seed=0):
    """Deterministic cyclic LM：seq[t] = (start + t) % vocab。next-token 可完美学习，用于演示因果生成。"""
    rng = random.Random(seed)
    X = []
    for _ in range(n):
        start = rng.randint(0, vocab_size - 1)
        seq = [(start + t) % vocab_size for t in range(seq_len)]
        X.append(seq)
    return X


def make_denoise_data(n=800, seq_len=8, vocab_size=8, mask_id=None, p_mask=0.25, seed=0):
    """BART-style 随机掩码去噪：src 把 25% 位置换成 [MASK]，tgt 为原句。"""
    rng = random.Random(seed)
    if mask_id is None:
        mask_id = vocab_size
    X = [[rng.randint(0, vocab_size - 1) for _ in range(seq_len)] for _ in range(n)]
    X_noisy = []
    for row in X:
        noisy = [mask_id if rng.random() < p_mask else tok for tok in row]
        X_noisy.append(noisy)
    return X_noisy, X, mask_id
