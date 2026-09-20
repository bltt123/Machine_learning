"""09-02 推荐训练/评估。"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def batch_tensors(df, genre, device="cpu"):
    from .data import tensor_frame
    return tensor_frame(df, genre, device)


def make_history(df, max_len=20):
    hist = {}
    for u, g in df.sort_values("timestamp").groupby("user_id"):
        hist[u] = list(g.loc[g.label > 0.5, "item_id"].astype(int))[-max_len:]
    return hist


def add_hist(df, history, max_len=20):
    rows = []
    for r in df.itertuples(index=False):
        h = history.get(r.user_id, [])[-max_len:]
        rows.append(([0] * (max_len - len(h))) + h or [0] * max_len)
    return rows


def train_rec(model, train_df, genre, history=None, epochs=5, lr=1e-3, batch=2048):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()
    for ep in range(1, epochs + 1):
        order = np.random.permutation(len(train_df)); total = 0.0
        model.train()
        for start in range(0, len(order), batch):
            d = train_df.iloc[order[start:start + batch]]
            u, i, g, y = batch_tensors(d, genre)
            kw = {}
            if history is not None:
                kw["hist_i"] = torch.tensor([[max(0, x - 1) for x in h] for h in add_hist(d, history)], dtype=torch.long)
            loss = bce(model(u, i, g, **kw), y)
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item() * len(d)
        print(f"{type(model).__name__} ep {ep:02d} loss {total/len(train_df):.4f}", flush=True)
    return model


@torch.no_grad()
def predict_rec(model, df, genre, history=None):
    model.eval(); out = []
    for start in range(0, len(df), 4096):
        d = df.iloc[start:start + 4096]
        u, i, g, _ = batch_tensors(d, genre)
        kw = {}
        if history is not None:
            kw["hist_i"] = torch.tensor([[max(0, x - 1) for x in h] for h in add_hist(d, history)], dtype=torch.long)
        out.extend(torch.sigmoid(model(u, i, g, **kw)).cpu().numpy())
    return np.asarray(out)


def auc(model, df, genre, history=None):
    return float(roc_auc_score(df.label.values, predict_rec(model, df, genre, history)))


def fit_reg(model, tr_loader, va_loader, epochs=15, lr=1e-3):
    """ETT 回归训练：保留 09-01 的统一接口。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.MSELoss(); hist = []
    for ep in range(1, epochs + 1):
        model.train(); total = 0.0; n = 0
        for xb, yb in tr_loader:
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(xb); n += len(xb)
        model.eval(); val_total = 0.0; val_n = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                val_total += crit(model(xb), yb).item() * len(xb); val_n += len(xb)
        hist.append((total / n, val_total / val_n))
        if ep in (1, 5, 10, 15, 20, 25, 30):
            print(f"ep {ep:02d} train {total/n:.4f} val {val_total/val_n:.4f}", flush=True)
    return hist


@torch.no_grad()
def test_mse(model, te_loader):
    """ETT 标准化域测试 MSE。"""
    model.eval(); crit = torch.nn.MSELoss(); total = 0.0; n = 0
    for xb, yb in te_loader:
        total += crit(model(xb), yb).item() * len(xb); n += len(xb)
    return total / n
