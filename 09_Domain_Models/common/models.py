"""09-02 推荐模型：LR / FM / DeepFM / DIN。"""
import torch
import torch.nn as nn


class RecBase(nn.Module):
    def __init__(self, users, items, genres, dim=16):
        super().__init__()
        self.user = nn.Embedding(users, dim)
        self.item = nn.Embedding(items, dim)
        self.genre = nn.Linear(genres, dim, bias=False)

    def fields(self, u, i, g):
        return self.user(u), self.item(i), self.genre(g)


class LRModel(nn.Module):
    def __init__(self, users, items, genres):
        super().__init__()
        self.user = nn.Embedding(users, 1); self.item = nn.Embedding(items, 1)
        self.genre = nn.Linear(genres, 1); self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u, i, g, **kwargs):
        return self.user(u).squeeze(-1) + self.item(i).squeeze(-1) + self.genre(g).squeeze(-1) + self.bias


class FMModel(RecBase):
    def __init__(self, users, items, genres, dim=16):
        super().__init__(users, items, genres, dim)
        self.user1 = nn.Embedding(users, 1); self.item1 = nn.Embedding(items, 1)
        self.genre1 = nn.Linear(genres, 1); self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u, i, g, **kwargs):
        f = torch.stack(self.fields(u, i, g), dim=1)
        second = 0.5 * ((f.sum(1) ** 2 - (f ** 2).sum(1)).sum(-1))
        first = self.user1(u).squeeze(-1) + self.item1(i).squeeze(-1) + self.genre1(g).squeeze(-1)
        return second + first + self.bias


class DeepFMModel(FMModel):
    def __init__(self, users, items, genres, dim=16):
        super().__init__(users, items, genres, dim)
        self.dnn = nn.Sequential(nn.Linear(dim * 3, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, u, i, g, **kwargs):
        f = torch.stack(self.fields(u, i, g), dim=1)
        fm = 0.5 * ((f.sum(1) ** 2 - (f ** 2).sum(1)).sum(-1))
        first = self.user1(u).squeeze(-1) + self.item1(i).squeeze(-1) + self.genre1(g).squeeze(-1)
        return fm + first + self.dnn(f.reshape(f.size(0), -1)).squeeze(-1) + self.bias


class DINModel(RecBase):
    def __init__(self, users, items, genres, dim=16):
        super().__init__(users, items, genres, dim)
        self.attn = nn.Sequential(nn.Linear(dim * 4, 32), nn.ReLU(), nn.Linear(32, 1))
        self.mlp = nn.Sequential(nn.Linear(dim * 3, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, u, i, g, hist_i=None, **kwargs):
        ue, ie, ge = self.fields(u, i, g)
        hist = ie.unsqueeze(1) if hist_i is None else self.item(hist_i)
        q = ie.unsqueeze(1).expand_as(hist)
        score = self.attn(torch.cat([hist, q, hist - q, hist * q], dim=-1)).squeeze(-1)
        h = (torch.softmax(score, dim=-1).unsqueeze(-1) * hist).sum(1)
        return self.mlp(torch.cat([ue, ie, ge + h], dim=-1)).squeeze(-1)
