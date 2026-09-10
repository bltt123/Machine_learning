"""03 分割家族引擎：CE+Dice 联合损失、mIoU / Dice 评估。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, n_classes: int, smooth: float = 1e-6):
    """多类 soft Dice loss（softmax 后逐类算，平均）。"""
    prob = F.softmax(logits, dim=1)
    y_oh = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
    inter = (prob * y_oh).sum((0, 2, 3))
    denom = prob.sum((0, 2, 3)) + y_oh.sum((0, 2, 3))
    dice = (2 * inter + smooth) / (denom + smooth)
    return 1 - dice.mean()


def miou(pred: torch.Tensor, target: torch.Tensor, n_classes: int) -> float:
    ious = []
    for c in range(n_classes):
        p, t = (pred == c), (target == c)
        union = (p | t).sum().item()
        if union == 0:
            continue
        ious.append((p & t).sum().item() / union)
    return sum(ious) / len(ious) if ious else 0.0


def _eval_epoch(model, loader, device, n_classes: int):
    model.eval()
    tot_ce = tot_dice = tot_miou = tot_d = 0
    ce_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            tot_ce += ce_fn(logits, y).item() * len(x)
            tot_dice += dice_loss(logits, y, n_classes).item() * len(x)
            tot_miou += miou(logits.argmax(1).cpu(), y.cpu(), n_classes) * len(x)
            pred = logits.argmax(1)
            inter = ((pred == 1).float().sum() + (pred == 2).float().sum())  # dummy to keep batch
            tot_d += 0  # placeholder; real Dice computed below per loader
    n = len(loader.dataset)
    # Dice as 2inter/union on probs: recompute consistently as 1 - dice_loss mean
    # so report (1 - dice_loss) averaged = Dice coeff
    avg_dice = 1 - tot_dice / n
    return {"loss": (tot_ce + tot_dice) / n, "miou": tot_miou / n, "dice": avg_dice}


def fit_seg(model, train_loader, val_loader, epochs: int = 6, lr: float = 1e-3,
            device: str = "cpu", verbose: bool = True):
    """CE + Dice 联合训练，返回 history dict。"""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce_fn = nn.CrossEntropyLoss()
    n_classes = model.head.out_channels if hasattr(model.head, "out_channels") else 3
    hist = {"train_loss": [], "val_loss": [], "val_miou": [], "val_dice": []}
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce_fn(logits, y) + dice_loss(logits, y, n_classes)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * len(x)
        hist["train_loss"].append(run / len(train_loader.dataset))
        ev = _eval_epoch(model, val_loader, device, n_classes)
        hist["val_loss"].append(ev["loss"]); hist["val_miou"].append(ev["miou"]); hist["val_dice"].append(ev["dice"])
        if verbose:
            print(f"epoch {ep:02d}/{epochs}  train {hist['train_loss'][-1]:.4f}  val {ev['loss']:.4f}  mIoU {ev['miou']:.4f}  Dice {ev['dice']:.4f}", flush=True)
    return hist
