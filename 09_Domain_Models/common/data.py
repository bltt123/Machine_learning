"""09-02：MovieLens-100K 预处理与负采样。

固定使用 u1.base/u1.test：按用户最后一条训练交互构造验证，
测试用官方 u1.test；训练负样本从用户未看电影中随机采 3 个。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_movielens(root, seed=0, negatives=3):
    """返回 train/val/test DataFrame、电影 genre 多热矩阵和字段规模。"""
    root = Path(root)
    names = ["user_id", "item_id", "rating", "timestamp"]
    train = pd.read_csv(root / "u1.base", sep="\t", names=names)
    test = pd.read_csv(root / "u1.test", sep="\t", names=names)
    train["label"] = (train.rating >= 4).astype(np.float32)
    test["label"] = (test.rating >= 4).astype(np.float32)
    rng = np.random.default_rng(seed)
    val_idx = train.groupby("user_id", sort=False).tail(1).index
    val = train.loc[val_idx].copy()
    train = train.drop(val_idx).copy()
    all_items = np.arange(1, int(max(train.item_id.max(), test.item_id.max())) + 1)
    seen = train.groupby("user_id")["item_id"].apply(set).to_dict()

    def add_negatives(df, n):
        rows = [df]
        neg = []
        for row in df.itertuples(index=False):
            if row.label < 0.5:
                continue
            candidates = np.setdiff1d(all_items, list(seen.get(row.user_id, set())))
            picks = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
            for item in picks:
                neg.append({"user_id": row.user_id, "item_id": int(item), "rating": 0,
                            "timestamp": row.timestamp, "label": 0.0})
        if neg:
            rows.append(pd.DataFrame(neg))
        out = pd.concat(rows, ignore_index=True)
        return out.sample(frac=1, random_state=seed).reset_index(drop=True)

    train = add_negatives(train, negatives)
    val = add_negatives(val, negatives)
    genres = pd.read_csv(root / "u.item", sep="|", header=None, encoding="latin-1")
    genre_cols = genres.columns[5:]
    genre = genres.set_index(0)[genre_cols].astype(np.float32)
    max_user = int(max(train.user_id.max(), val.user_id.max(), test.user_id.max()))
    max_item = int(max(train.item_id.max(), val.item_id.max(), test.item_id.max()))
    sizes = {"users": max_user, "items": max_item, "genres": len(genre_cols)}
    return train, val, test, genre, sizes


def tensor_frame(df, genre, device="cpu"):
    """DataFrame→(user,item,genre,label) tensors，电影不存在 genre 时补零。"""
    g = genre.reindex(df.item_id).fillna(0).values
    return (torch.tensor(df.user_id.values - 1, dtype=torch.long, device=device),
            torch.tensor(df.item_id.values - 1, dtype=torch.long, device=device),
            torch.tensor(g, dtype=torch.float32, device=device),
            torch.tensor(df.label.values, dtype=torch.float32, device=device))
