"""03 分割家族数据：Oxford-IIIT Pet（torchvision 版）—— 128×128 彩色图像 + trimap(0=背景 1=前景 2=边界)。"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_SIZE = 128
CLASS_NAMES = ["背景", "前景", "边界"]
N_CLASSES = 3
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _resolve_pet_root(root: str) -> Path:
    p = Path(root)
    if (p / "images").exists() and (p / "annotations").exists():
        return p
    if (p / "oxford-iiit-pet" / "images").exists():
        return p / "oxford-iiit-pet"
    return p


def _collect_pet_samples(root: Path):
    ann_file = root / "annotations" / "list.txt"
    if ann_file.exists():
        names = [l.split()[0] for l in ann_file.read_text().splitlines() if not l.startswith("#")]
    else:
        names = sorted(p.stem for p in (root / "images").glob("*.jpg"))
    return [n for n in names if (root / "images" / f"{n}.jpg").exists()
            and (root / "annotations" / "trimaps" / f"{n}.png").exists()]


def _load_pair(root: Path, name: str):
    img = Image.open(root / "images" / f"{name}.jpg").convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    m = Image.open(root / "annotations" / "trimaps" / f"{name}.png").convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    x = torch.from_numpy(np.array(img, dtype=np.float32).transpose(2, 0, 1)) / 255.0
    x = (x - MEAN) / STD
    y = torch.from_numpy(np.array(m, dtype=np.int64) - 1)  # 1/2/3 → 0/1/2
    y = y.clamp(0, 2)
    return x, y


def load_pet_segmentation(root: str, train_n: int = 500, val_n: int = 200):
    """返回 (Xtr,ytr,Xva,yva) 张量，形状 (N,3,128,128)/(N,128,128)，已标准化，seed=0 洗牌后切分。"""
    root = _resolve_pet_root(root)
    assert (root / "images").exists(), f"未找到 images 目录：{root}"
    names = _collect_pet_samples(root)
    assert len(names) >= train_n + val_n, f"可用样本 {len(names)} < {train_n+val_n}"
    rng = np.random.default_rng(0)
    rng.shuffle(names)
    tr_names, va_names = names[:train_n], names[train_n:train_n + val_n]
    xs = lambda ns: torch.stack([_load_pair(root, n)[0] for n in ns])
    ys = lambda ns: torch.stack([_load_pair(root, n)[1] for n in ns])
    return xs(tr_names), ys(tr_names), xs(va_names), ys(va_names)


class PetSegDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x * STD + MEAN).clamp(0, 1)
