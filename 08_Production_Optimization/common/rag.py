"""08-08：中文 TF-IDF 检索 + 抽取式问答（零新包：jieba + sklearn）。

设计： taught-why 不用向量库——TF-IDF 让“词项权重×余弦”全程可见；
生产切 dense embedding 只换向量器，流程（建库→检索→抽取→拒答）不变。
"""
import re

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOPWORDS = set("的了是在和与或有为这那个一不也都而及把被对就可要能会可以应该如果但是因为所以 thereof"
                .replace("thereof", "")) | {"，", "。", "、", "？", "！", "：", "；", "（", "）", "(", ")", " "}


def tokenize_zh(text):
    """jieba 分词 + 去停用词/空白，返回空格拼接串（给 TfidfVectorizer 按空格切）。"""
    toks = [t.strip() for t in jieba.cut(text) if t.strip() and t.strip() not in STOPWORDS]
    return " ".join(toks)


def split_sentences(text):
    """按中文句界切分，保留句号以便直接展示答案。"""
    parts = re.split(r"(?<=[。！？；])", text)
    return [p.strip() for p in parts if p.strip()]


class ZhRAG:
    """最小 RAG：fit 建库 → retrieve TopK → answer 抽取最佳句（低分拒答）。"""

    def __init__(self, docs, threshold=0.08):
        self.docs = list(docs)
        self.threshold = threshold
        self.vec = TfidfVectorizer(tokenizer=str.split, preprocessor=None, lowercase=False,
                                   token_pattern=None)
        self.mat = self.vec.fit_transform([tokenize_zh(d) for d in self.docs])

    def retrieve(self, query, top_k=3):
        q = self.vec.transform([tokenize_zh(query)])
        scores = cosine_similarity(q, self.mat).flatten()
        order = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in order]

    def answer(self, query, top_k=3):
        ranked = self.retrieve(query, top_k=top_k)
        best_id, best_score = ranked[0]
        if best_score < self.threshold:
            return {"doc_id": -1, "score": best_score, "text": "不知道（库外问题，已拒答）",
                    "known": False, "ranked": ranked}
        q_terms = set(tokenize_zh(query).split())
        sents = split_sentences(self.docs[best_id])
        scored = [(s, len(set(tokenize_zh(s).split()) & q_terms)) for s in sents]
        best_sent = max(scored, key=lambda t: (t[1], len(t[0])))[0]
        return {"doc_id": best_id, "score": best_score, "text": best_sent,
                "known": True, "ranked": ranked}

    def evaluate(self, pairs, top_k=3):
        """pairs: [(query, gold_doc_id)] → recall@1/@k + 抽取命中率。"""
        r1 = rk = hit = 0
        for q, gold in pairs:
            ranked = self.retrieve(q, top_k=top_k)
            ids = [i for i, _ in ranked]
            r1 += ids[0] == gold
            rk += gold in ids
            ans = self.answer(q, top_k=top_k)
            hit += ans["doc_id"] == gold
        n = len(pairs)
        out = {"recall@1": r1 / n, f"recall@{top_k}": rk / n, "answer_hit": hit / n}
        for k in (2, 5):
            out.setdefault(f"recall@{k}", out.get(f"recall@{top_k}", 0.0))
        return out
