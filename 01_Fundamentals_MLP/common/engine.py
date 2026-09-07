"""01 家族通用训练/评估循环（PyTorch）。

约定：模型输出 logits（未过 softmax），标签为 {0..C-1} 的 int64，
内部统一用 CrossEntropyLoss。03 项目的优化器对比通过 optimizer_cls 注入。
"""
import torch
import torch.nn as nn


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    """跑一个 epoch。传入 optimizer 即训练模式，否则纯评估（无梯度）。"""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        n += yb.size(0)
    return total_loss / n, correct / n


def fit(model, train_loader, val_loader, epochs=5, lr=1e-3,
        optimizer_cls=torch.optim.Adam, device="cpu", verbose=True,
        criterion=None, weight_decay=0.0, es_patience=None):
    """标准训练循环，返回曲线 hist：{train_loss, train_acc, val_loss, val_acc}。

    criterion: 可插拔损失（默认 CrossEntropyLoss）；label_smoothing 等由外部构造好传入。
    weight_decay: 透传给优化器构造（Adam 是 L2-in-grad，AdamW 是解耦——03 消融会用到差异）。
    es_patience: 早停容忍轮数（监控 val_acc，连续 N 轮无提升即停）；None 表示关闭。
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, bad = -1.0, 0
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device)
        for key, value in zip(hist, (tr_loss, tr_acc, va_loss, va_acc)):
            hist[key].append(value)
        if verbose:
            print(f"epoch {ep:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
                  f"| val loss {va_loss:.4f} acc {va_acc:.4f}")
        if es_patience is not None:
            if va_acc > best_acc:
                best_acc, bad = va_acc, 0
            else:
                bad += 1
                if bad >= es_patience:
                    if verbose:
                        print(f"early stop @ epoch {ep}（val_acc 连续 {es_patience} 轮无提升）")
                    break
    return hist
