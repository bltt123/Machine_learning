"""06 家族训练/评估循环（与 02 家族 engine 同源，接口一致）。"""
import torch
import torch.nn as nn


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = criterion(logits, yb)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        n += yb.size(0)
    return total_loss / n, correct / n


def fit(model, train_loader, val_loader, epochs=5, lr=3e-4,
        optimizer_cls=torch.optim.AdamW, device="cpu", verbose=True,
        criterion=None, weight_decay=0.05):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device)
        for key, value in zip(hist, (tr_loss, tr_acc, va_loss, va_acc)):
            hist[key].append(value)
        if verbose:
            print(f"epoch {ep:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
                  f"| val loss {va_loss:.4f} acc {va_acc:.4f}")
    return hist
