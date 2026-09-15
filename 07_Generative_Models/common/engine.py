"""07-02 GAN/DCGAN 训练循环：非饱和 GAN + history + 固定噪声采样。"""
import torch
import torch.nn as nn


def fit_gan(G, D, loader, epochs=8, z_dim=64, lr=2e-4, device="cpu", seed=0, fixed_z=None):
    """训练一个 GAN，返回 G/D loss、固定噪声快照和每 epoch mode proxy。

    判别器用 BCEWithLogits；生成器用非饱和目标 ``-log D(G(z))``，比原始 minimax 梯度更稳定。
    """
    torch.manual_seed(seed)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    fixed_z = fixed_z if fixed_z is not None else torch.randn(64, z_dim, device=device)
    hist = {"g": [], "d": [], "snapshots": [], "real_score": [], "fake_score": []}
    for ep in range(1, epochs + 1):
        G.train(); D.train(); sum_g = sum_d = real_s = fake_s = n = 0
        for batch in loader:
            real = batch[0].to(device); b = real.size(0)
            ones = torch.ones(b, device=device); zeros = torch.zeros(b, device=device)
            # D：真图为1，假图为0
            z = torch.randn(b, z_dim, device=device)
            fake = G(z).detach()
            d_loss = criterion(D(real), ones) + criterion(D(fake), zeros)
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()
            # G：非饱和目标，骗过 D
            z = torch.randn(b, z_dim, device=device)
            g_loss = criterion(D(G(z)), ones)
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()
            sum_g += g_loss.item() * b; sum_d += d_loss.item() * b
            real_s += torch.sigmoid(D(real)).mean().item() * b
            fake_s += torch.sigmoid(D(G(z).detach())).mean().item() * b
            n += b
        with torch.no_grad(): snap = G(fixed_z).detach().cpu()
        hist["g"].append(sum_g / n); hist["d"].append(sum_d / n)
        hist["snapshots"].append(snap)
        hist["real_score"].append(real_s / n); hist["fake_score"].append(fake_s / n)
        print(f"epoch {ep:02d} | G {hist['g'][-1]:.3f} D {hist['d'][-1]:.3f} "
              f"D(real) {hist['real_score'][-1]:.2f} D(fake) {hist['fake_score'][-1]:.2f}", flush=True)
    return hist
