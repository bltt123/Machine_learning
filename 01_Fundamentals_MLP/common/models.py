"""01 家族模型定义：感知机（01 项目）与 MLP（02/03 项目）。"""
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------
# 01 项目：手写感知机（纯 NumPy，教学用）
# ---------------------------------------------------------------
class Perceptron:
    """感知机（原始形式）。

    标签约定 y ∈ {-1, +1}。误分类判定：y_i * (w·x_i + b) <= 0，
    更新规则：w <- w + lr * y_i * x_i,  b <- b + lr * y_i
    """

    def __init__(self, lr: float = 1.0, max_epochs: int = 100, seed: int = 0):
        self.lr = lr
        self.max_epochs = max_epochs
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        rng = np.random.default_rng(self.seed)
        self.w_ = np.zeros(X.shape[1], dtype=np.float64)
        self.b_ = 0.0
        self.n_updates_ = 0
        self.n_epochs_ = 0
        self.converged_ = False
        for epoch in range(1, self.max_epochs + 1):
            mistakes = 0
            for i in rng.permutation(len(X)):
                if y[i] * (X[i] @ self.w_ + self.b_) <= 0:
                    self.w_ += self.lr * y[i] * X[i]
                    self.b_ += self.lr * y[i]
                    mistakes += 1
                    self.n_updates_ += 1
            self.n_epochs_ = epoch
            if mistakes == 0:
                self.converged_ = True
                break
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w_ + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


# ---------------------------------------------------------------
# 02/03 项目：MLP（PyTorch）
# ---------------------------------------------------------------
class MLP(nn.Module):
    """可插拔归一化/激活/Dropout 的多层感知机。

    norm: None | "bn" | "ln" | "rms"——作用在每个隐藏层的 Linear 之后、激活之前。
    """

    def __init__(self, in_dim: int, hidden_dims=(128, 64), out_dim: int = 10,
                 dropout: float = 0.0, activation=nn.ReLU, norm=None):
        super().__init__()
        self.dropout = dropout
        norm_map = {"bn": nn.BatchNorm1d, "ln": nn.LayerNorm, "rms": nn.RMSNorm}
        layers, d = [], in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            if norm is not None:
                layers.append(norm_map[norm](h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
