"""评估与训练引擎：NER 实体 F1 + 文本分类/Seq2Seq 训练循环。"""
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── NER 评估 ──

def token_accuracy(pred, gold):
    correct = sum(p == g for ps, gs in zip(pred, gold) for p, g in zip(ps, gs))
    total = sum(len(s) for s in gold)
    return correct / total if total else 0.0


def _spans(tags):
    spans = set()
    i = 0
    while i < len(tags):
        t = tags[i]
        if t.startswith("B-"):
            typ = t[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{typ}":
                j += 1
            spans.add((typ, i, j - 1))
            i = j
        else:
            i += 1
    return spans


def entity_f1(pred_tags, gold_tags):
    pred_all, gold_all = set(), set()
    for idx, (ps, gs) in enumerate(zip(pred_tags, gold_tags)):
        for sp in _spans(ps):
            pred_all.add((idx,) + sp)
        for sp in _spans(gs):
            gold_all.add((idx,) + sp)
    tp = len(pred_all & gold_all)
    prec = tp / len(pred_all) if pred_all else (1.0 if not gold_all else 0.0)
    rec = tp / len(gold_all) if gold_all else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def confusion_tags(pred, gold, tag_list):
    cnt = Counter()
    for ps, gs in zip(pred, gold):
        for p, g in zip(ps, gs):
            if p != g:
                cnt[(g, p)] += 1
    return cnt

# ── 文本分类训练 ──

def fit_text_classifier(model, train_X, train_y, test_X, test_y, epochs=30, batch_size=32, lr=5e-3, device="cpu", verbose=True):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    train_ds = TensorDataset(torch.tensor(train_X, dtype=torch.long), torch.tensor(train_y, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(test_X, dtype=torch.long), torch.tensor(test_y, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    def _acc(loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        return correct / total

    hist = {"train_acc": [], "test_acc": []}
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        tr = _acc(train_loader)
        te = _acc(test_loader)
        hist["train_acc"].append(tr)
        hist["test_acc"].append(te)
        if verbose and (ep <= 3 or ep % 5 == 0 or ep == epochs):
            print(f"epoch {ep:02d}/{epochs}  train {tr:.3f}  test {te:.3f}", flush=True)
    return hist


def fit_seq2seq(model, data_fn, epochs=60, batch_size=32, lr=8e-3, device="cpu", verbose=True):
    """data_fn 返回 (src, tgt) 均为 (N, S) LongTensor，task 为复制或翻转。"""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    src, tgt = data_fn()
    ds = TensorDataset(torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    def _acc():
        model.eval()
        with torch.no_grad():
            logits, _ = model(torch.tensor(src, dtype=torch.long, device=device),
                              torch.tensor(tgt, dtype=torch.long, device=device))
            pred = logits.argmax(-1).cpu().numpy()
            correct = (pred == tgt).all(axis=1).sum() if hasattr(tgt, 'all') else 0
            # need numpy tgt
            import numpy as np
            tgt_np = np.array(tgt)
            acc = (pred == tgt_np).all(axis=1).mean()
            return float(acc)

    for ep in range(1, epochs + 1):
        model.train()
        for s, t in loader:
            s, t = s.to(device), t.to(device)
            opt.zero_grad()
            logits, _ = model(s, t)
            loss = crit(logits.reshape(-1, logits.size(-1)), t.reshape(-1))
            loss.backward()
            opt.step()
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"epoch {ep:02d}/{epochs}  seq-acc {_acc():.3f}", flush=True)
    return _acc()
