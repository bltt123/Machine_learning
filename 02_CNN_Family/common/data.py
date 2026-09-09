"""02 CNN 家族数据加载。图像任务约定：一律 (N, C, H, W) 通道优先，float32 / int64。"""
import numpy as np
import torch

FASHION_CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def _load_image_dataset(root: str, dataset_cls, mean: float, std: float):
    train = dataset_cls(root, train=True, download=True)
    test = dataset_cls(root, train=False, download=True)
    Xtr = train.data.numpy().astype(np.float32) / 255.0
    Xte = test.data.numpy().astype(np.float32) / 255.0
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    ytr, yte = train.targets.numpy(), test.targets.numpy()
    return (
        torch.as_tensor(Xtr[:, None, :, :], dtype=torch.float32),
        torch.as_tensor(ytr, dtype=torch.int64),
        torch.as_tensor(Xte[:, None, :, :], dtype=torch.float32),
        torch.as_tensor(yte, dtype=torch.int64),
    )


def load_mnist_torch(root: str):
    """MNIST 下载与标准化（0.1307/0.3081，离线缓存于 root）。

    返回 (X_train, y_train, X_test, y_test)，图像为 (N, 1, 28, 28)。
    """
    from torchvision import datasets
    return _load_image_dataset(root, datasets.MNIST, 0.1307, 0.3081)


def load_fashion_mnist_torch(root: str):
    """Fashion-MNIST（比 MNIST 难得多：语义相近类别多，SOTA ~94%）。

    返回 (X_train, y_train, X_test, y_test)，图像为 (N, 1, 28, 28)；
    类别名见 FASHION_CLASSES。
    """
    from torchvision import datasets
    return _load_image_dataset(root, datasets.FashionMNIST, 0.2860, 0.3530)


def load_cifar10_torch(root: str):
    """CIFAR-10（32×32 彩色 3 通道，10 类，SOTA ~99.5%，ResNet 时代基准）。

    返回 (X_train, y_train, X_test, y_test)，图像为 (N, 3, 32, 32)；
    类别名见 CIFAR10_CLASSES。
    """
    from torchvision import datasets
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(3, 1, 1)
    train = datasets.CIFAR10(root, train=True, download=True)
    test = datasets.CIFAR10(root, train=False, download=True)
    Xtr = train.data.astype(np.float32) / 255.0          # (N,32,32,3) HWC
    Xte = test.data.astype(np.float32) / 255.0
    Xtr = ((Xtr.transpose(0, 3, 1, 2) - mean) / std)     # → (N,3,32,32) CHW
    Xte = ((Xte.transpose(0, 3, 1, 2) - mean) / std)
    ytr, yte = np.array(train.targets, dtype=np.int64), np.array(test.targets, dtype=np.int64)
    return (
        torch.as_tensor(Xtr, dtype=torch.float32),
        torch.as_tensor(ytr),
        torch.as_tensor(Xte, dtype=torch.float32),
        torch.as_tensor(yte),
    )


CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


class AugLoader:
    """通用批量增强数据迭代器（CPU 张量实现，等价 torchvision.transforms 的效果）。

    洗牌与增强共用固定种子的独立 generator——序列完全可复现；
    aug_fn(xb, g) 需是"保标签"的批量随机变换，传 None 则只做洗牌。
    """

    def __init__(self, x, y, batch_size: int = 128, aug_fn=None, seed: int = 0):
        self.x, self.y, self.bs, self.aug_fn = x, y, batch_size, aug_fn
        self.g = torch.Generator().manual_seed(seed)

    def __iter__(self):
        n = len(self.x)
        idx = torch.randperm(n, generator=self.g)
        for s in range(0, n, self.bs):
            b = idx[s:s + self.bs]
            xb = self.aug_fn(self.x[b], self.g) if self.aug_fn is not None else self.x[b]
            yield xb, self.y[b]

    def __len__(self):
        return (len(self.x) + self.bs - 1) // self.bs


def cifar10_aug(xb: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """CIFAR-10 标准增强两件套（保标签）：4 像素 padding 随机裁剪 + 50% 水平翻转。"""
    B, C, H, W = xb.shape
    if torch.rand(1, generator=g).item() < 0.5:
        xb = torch.flip(xb, dims=[3])
    pad = torch.nn.functional.pad(xb, (4, 4, 4, 4))
    i = torch.randint(0, 9, (1,), generator=g).item()
    j = torch.randint(0, 9, (1,), generator=g).item()
    return pad[:, :, i:i + H, j:j + W]
