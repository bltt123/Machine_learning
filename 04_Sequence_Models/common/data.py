"""04 序列家族数据：Toy 中文 NER（字符级 BIO）+ 首尾比较的序敏感任务。

设计目标：无外网、无重依赖，足以讲清 HMM/CRF 差异与 RNN 梯度消失。
"""
from collections import Counter
import numpy as np

# —— 语料：字符已分好，BIO 标注 ——
# 人名 PER、地名 LOC、组织 ORG
RAW_SENTENCES = [
    # 训练 8 句
    (["张", "三", "在", "北", "京", "工", "作"], ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "O", "O"]),
    (["李", "四", "去", "上", "海", "出", "差"], ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "O", "O"]),
    (["王", "五", "和", "赵", "六", "在", "深", "圳", "开", "会"], ["B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-LOC", "I-LOC", "O", "O"]),
    (["陈", "明", "来", "自", "华", "为", "公", "司"], ["B-PER", "I-PER", "O", "O", "B-ORG", "I-ORG", "I-ORG", "I-ORG"]),
    (["小", "明", "在", "清", "华", "大", "学", "读", "书"], ["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "O"]),
    (["北", "京", "大", "学", "的", "李", "华", "教", "授"], ["B-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "B-PER", "I-PER", "O", "O"]),
    (["阿", "里", "巴", "巴", "在", "杭", "州"], ["B-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "B-LOC", "I-LOC"]),
    (["张", "三", "和", "李", "四", "去", "广", "州"], ["B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-LOC", "I-LOC"]),
    # 测试 4 句（字符/组合在训练出现过，但序列是新组合）
    (["赵", "六", "在", "上", "海", "工", "作"], ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "O", "O"]),
    (["王", "五", "来", "自", "杭", "州"], ["B-PER", "I-PER", "O", "O", "B-LOC", "I-LOC"]),
    (["李", "华", "在", "北", "京", "大", "学"], ["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "I-ORG", "I-ORG"]),
    (["华", "为", "的", "陈", "明"], ["B-ORG", "I-ORG", "O", "B-PER", "I-PER"]),
]

TAGS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
ID2TAG = {i: t for t, i in TAG2ID.items()}


def get_splits():
    """返回 (train_X, train_Y, test_X, test_Y)，各为 List[List[str]]。"""
    train = RAW_SENTENCES[:8]
    test = RAW_SENTENCES[8:]
    train_X, train_Y = zip(*train)
    test_X, test_Y = zip(*test)
    return list(train_X), list(train_Y), list(test_X), list(test_Y)


def build_vocab(sentences):
    """从所有句子建字符词表，含 <UNK>。"""
    chars = Counter(ch for s in sentences for ch in s)
    vocab = {"<UNK>": 0, "<PAD>": 1}
    for ch in sorted(chars):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def tag_distribution(tag_seqs):
    """统计标签分布 Counter。"""
    return Counter(t for seq in tag_seqs for t in seq)


# —— 首尾比较任务：y=1 当首字<尾字，中间为噪音 ——
def make_first_last_data(n_train=800, n_test=200, seq_len=30, vocab_size=12, seed=0):
    """生成 toy 序敏感数据：首<尾为 1，其余随机。"""
    rng = np.random.default_rng(seed)
    def _gen(n):
        X = rng.integers(0, vocab_size, size=(n, seq_len)).tolist()
        y = [1 if seq[0] < seq[-1] else 0 for seq in X]
        return X, y
    train_X, train_y = _gen(n_train)
    test_X, test_y = _gen(n_test)
    return train_X, train_y, test_X, test_y


# —— Seq2Seq toy：复制与翻转 ——
def make_copy_data(n=600, seq_len=6, vocab_size=8, seed=0):
    """复制任务：src==tgt，适合可视化对角 Attention。"""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, vocab_size, size=(n, seq_len)).tolist()
    return X, [row[:] for row in X]


def make_reverse_data(n=600, seq_len=6, vocab_size=8, seed=1):
    """翻转任务：tgt 是 src 翻转，需长程重排。"""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, vocab_size, size=(n, seq_len)).tolist()
    Y = [row[::-1] for row in X]
    return X, Y
