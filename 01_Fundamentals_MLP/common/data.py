"""01 家族玩具/小数据集生成与加载。

家族统一约定：
- 数据一律为 numpy/torch 张量，标签为 {0,1} 或 {0..9}
- 玩具数据无划分，返回 (X, y, y)（第三项仅占位，与 y 相同）
- 真实数据集（MNIST 等）返回 (X_train, y_train, X_test, y_test)
- dtype 统一 float32 / int64
"""
import numpy as np
import torch


def make_linearly_separable(n: int = 200, margin: float = 0.8, seed: int = 0):
    """严格线性可分二维数据：两类中心 (-1,-1)/(+1,+1)，保证间隔 margin>0。

    构造方式：先生成一类高斯点，再镜像平移，确保任意点与决策面距离 >= margin。
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    base = rng.normal(loc=(-1.0, -1.0), scale=0.35, size=(half, 2))
    # 沿 (1,1) 方向把另一类推开足够远，保证两团点之间有硬间隔
    offset = np.array([2.0, 2.0]) + margin
    X = np.vstack([base, base + offset])
    y = np.concatenate([np.zeros(n - half, dtype=np.int64), np.ones(half, dtype=np.int64)])
    return to_tensors(X, y, y)


def make_xor(n: int = 200, noise: float = 0.15, seed: int = 0):
    """XOR 数据：异号象限为 1、同号象限为 0。线性不可分的经典反例。"""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = ((X[:, 0] * X[:, 1]) < 0).astype(np.int64)
    X = X + rng.normal(scale=noise, size=(n, 2))
    return to_tensors(X, y, y)


def load_mnist_torch(root: str, flatten: bool = True):
    """MNIST 下载与标准化（常用统计量，离线缓存于 root）。

    返回 (X_train, y_train, X_test, y_test) 四元组。
    """
    from torchvision import datasets

    train = datasets.MNIST(root, train=True, download=True)
    test = datasets.MNIST(root, train=False, download=True)

    Xtr = train.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    Xte = test.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    mean, std = 0.1307, 0.3081
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    ytr, yte = train.targets.numpy(), test.targets.numpy()
    if not flatten:
        Xtr = Xtr.reshape(-1, 1, 28, 28)
        Xte = Xte.reshape(-1, 1, 28, 28)
    return (
        torch.as_tensor(Xtr, dtype=torch.float32),
        torch.as_tensor(ytr, dtype=torch.int64),
        torch.as_tensor(Xte, dtype=torch.float32),
        torch.as_tensor(yte, dtype=torch.int64),
    )


def load_iris_binary():
    """Iris 二分类（setosa=+1, versicolor=-1），取花瓣长/宽两特征，共 100 条。"""
    from sklearn.datasets import load_iris

    iris = load_iris()
    mask = iris.target < 2
    X = iris.data[mask][:, 2:4].astype(np.float32)  # petal length / width
    y = np.where(iris.target[mask] == 0, 1, -1).astype(np.int64)
    return to_tensors(X, y, y)


def to_tensors(X, y, y_test=None):
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(y), dtype=torch.int64)
    y2 = torch.as_tensor(np.asarray(y_test), dtype=torch.int64)
    return X, y, y2


def train_test_split(X: torch.Tensor, y: torch.Tensor, test_ratio: float = 0.3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    idx = torch.randperm(n, generator=g)
    n_test = int(n * test_ratio)
    return X[idx[n_test:]], y[idx[n_test:]], X[idx[:n_test]], y[idx[:n_test]]
