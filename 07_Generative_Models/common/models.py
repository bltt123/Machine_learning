"""ConvAE / ConvVAE（07-01）+ MLP-GAN / DCGAN（07-02）+ DDPM-Denoiser（07-03）+ GlowLite（07-04 拓展）。

07-01：28×28 → 32 维瓶颈 → 还原，VAE 加重参数采样。
07-02：z(64) → 28×28（MLP 生成器 / DCGAN 生成器），判别器二分类真假。
07-03：DDPM 噪声预测器（t-嵌入 + UNet-lite），x_t → ε̂，一步采样 x_T → x_0。
07-04：RealNVP 耦合层可逆映射 x ↔ z，精确对数似然 + 往返一致性验证。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorMLP(nn.Module):
    """MLP-GAN 生成器：64 维噪声 → 28×28 灰度图（sigmoid 输出）。"""

    def __init__(self, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Linear(512, 784), nn.Sigmoid(),
        )
        self.z_dim = z_dim

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class DiscriminatorMLP(nn.Module):
    """MLP-GAN 判别器：图像展平 → logits（配 BCEWithLogitsLoss）。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3), nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).flatten()


class GeneratorDCGAN(nn.Module):
    """DCGAN 生成器：z → 7×7 特征（MLP 257k 参）→ 两次反卷积到 28×28。"""

    def __init__(self, z_dim=64):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Sequential(nn.Linear(z_dim, 128 * 7 * 7), nn.BatchNorm1d(128 * 7 * 7), nn.ReLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 128, 7, 7))


class DiscriminatorDCGAN(nn.Module):
    """DCGAN 判别器：两次卷积下采样到真假 logits。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Flatten(), nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, x):
        return self.net(x).flatten()


class ConvAE(nn.Module):
    """卷积自编码器：编码 1→32→64 通道，瓶颈 32 维；解码转置卷积还原。"""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),   # 14
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),  # 7
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 64 * 7 * 7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),  # 14
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),              # 28
        )
        self.latent_dim = latent_dim

    def encode(self, x):
        return self.fc_mu(self.enc(x).flatten(1))

    def decode(self, z):
        return self.dec(self.fc_dec(z).view(-1, 64, 7, 7))

    def forward(self, x):
        return self.decode(self.encode(x))


class ConvVAE(nn.Module):
    """卷积 VAE：编码输出 (mu, logvar)，重参数 z = mu + eps·exp(0.5·logvar)。"""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),   # 14
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),  # 7
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 64 * 7 * 7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),  # 14
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),              # 28
        )
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.dec(self.fc_dec(z).view(-1, 64, 7, 7))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, device="cpu"):
        self.eval()
        z = torch.randn(n, self.latent_dim, device=device)
        return torch.sigmoid(self.decode(z))


def vae_loss(recon_logits, x, mu, logvar, beta=1.0):
    """ELBO：BCE 重建（logits 版防溢出）+ β·KL(q‖N(0,I))，按 batch 均值。

    x 是标准化图（0.1307/0.3081），先逆变换回 [0,1] 再当 BCE 目标——与采样显示口径一致。
    """
    target = (x * 0.3081 + 0.1307).clamp(0, 1)
    bce = F.binary_cross_entropy_with_logits(recon_logits, target, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    n = x.size(0)
    return (bce + beta * kl) / n, bce.item() / n, kl.item() / n


class SinusoidalTimeEmbedding(nn.Module):
    """把扩散步 t 编成 sin/cos 向量，告诉网络当前雾有多厚。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        scale = torch.log(torch.tensor(10000.0, device=t.device)) / max(half - 1, 1)
        freq = torch.exp(-scale * torch.arange(half, device=t.device))
        phase = t.float().unsqueeze(1) * freq.unsqueeze(0)
        return torch.cat([phase.sin(), phase.cos()], dim=1)


class DenoiseUNetLite(nn.Module):
    """轻量 DDPM 噪声预测器：U 形卷积骨干 + 时间条件，预测加入的 epsilon。"""

    def __init__(self, base=32, time_dim=32):
        super().__init__()
        self.time = nn.Sequential(SinusoidalTimeEmbedding(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU())
        self.down1 = nn.Sequential(nn.Conv2d(1, base, 3, padding=1), nn.GroupNorm(4, base), nn.SiLU())
        self.down2 = nn.Sequential(nn.Conv2d(base, base * 2, 4, stride=2, padding=1), nn.GroupNorm(8, base * 2), nn.SiLU())
        self.mid = nn.Sequential(nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.GroupNorm(8, base * 2), nn.SiLU())
        self.up = nn.Sequential(nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1), nn.GroupNorm(4, base), nn.SiLU())
        self.out = nn.Conv2d(base, 1, 3, padding=1)
        self.to_t1 = nn.Linear(time_dim, base)
        self.to_t2 = nn.Linear(time_dim, base * 2)

    def forward(self, x, t):
        temb = self.time(t)
        h1 = self.down1(x) + self.to_t1(temb)[:, :, None, None]
        h2 = self.down2(h1) + self.to_t2(temb)[:, :, None, None]
        h = self.mid(h2)
        return self.out(self.up(h) + h1)


class DDPM(nn.Module):
    """DDPM 前向加噪 + 反向采样封装；模型学习 epsilon，不直接学习像素。"""

    def __init__(self, timesteps=50, base=32):
        super().__init__()
        self.timesteps = timesteps
        beta = torch.linspace(1e-4, 0.02, timesteps)
        alpha = 1.0 - beta
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("abar", torch.cumprod(alpha, dim=0))
        self.net = DenoiseUNetLite(base=base)

    def q_sample(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        a = self.abar[t].sqrt()[:, None, None, None]
        b = (1 - self.abar[t]).sqrt()[:, None, None, None]
        return a * x0 + b * noise, noise

    def loss(self, x0):
        t = torch.randint(0, self.timesteps, (x0.size(0),), device=x0.device)
        xt, noise = self.q_sample(x0, t)
        return F.mse_loss(self.net(xt, t), noise)

    @torch.no_grad()
    def sample(self, n, device="cpu", trajectory_steps=8):
        self.eval(); x = torch.randn(n, 1, 28, 28, device=device)
        trajectory = [x.cpu()]
        keep = set(torch.linspace(self.timesteps - 1, 0, trajectory_steps).long().tolist())
        for ti in reversed(range(self.timesteps)):
            t = torch.full((n,), ti, device=device, dtype=torch.long)
            eps = self.net(x, t)
            a, ab, b = self.alpha[ti], self.abar[ti], self.beta[ti]
            mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
            x = mean + (b.sqrt() * torch.randn_like(x) if ti > 0 else 0)
            if ti in keep: trajectory.append(x.cpu())
        return x.clamp(-1, 1), trajectory


class AffineCoupling(nn.Module):
    """RealNVP 仿射耦合层：一半通道当条件，另一半做 s/t 变换，log-det 是 s 的和。"""

    def __init__(self, dim, hidden=256, flip=False):
        super().__init__()
        self.flip = flip
        self.net = nn.Sequential(
            nn.Linear(dim // 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, (dim // 2) * 2),
        )
        self.scale = nn.Parameter(torch.zeros(1))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
                nn.init.xavier_uniform_(m.weight, gain=0.5)

    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        if self.flip:
            x1, x2 = x2, x1
        s_raw, t = self.net(x1).chunk(2, dim=1)
        s = torch.tanh(s_raw) * torch.exp(self.scale).clamp(max=2.0)
        if not reverse:
            y2 = x2 * torch.exp(s) + t
            logdet = s.sum(dim=1)
        else:
            y2 = (x2 - t) * torch.exp(-s)
            logdet = -s.sum(dim=1)
        y = torch.cat([x1, y2], dim=1)
        if self.flip:
            y = torch.cat(torch.chunk(y, 2, dim=1)[::-1], dim=1)
        return y, logdet


class GlowLite(nn.Module):
    """RealNVP 风格可逆流：784 维展平 + K 个耦合层 + ActNorm，x ↔ z 精确互逆。"""

    def __init__(self, dim=784, n_couplings=6, hidden=256):
        super().__init__()
        self.dim = dim
        self.couplings = nn.ModuleList(
            [AffineCoupling(dim, hidden, flip=(i % 2 == 1)) for i in range(n_couplings)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, reverse=False):
        layers = list(self.couplings) if not reverse else list(reversed(self.couplings))
        logdet = torch.zeros(x.size(0), device=x.device)
        for layer in layers:
            x, ld = layer(x, reverse=reverse)
            logdet = logdet + ld
        return x, logdet

    def nll(self, x):
        """负对数似然：先验 z~N(0,I) + 体积变化 log-det。

        LayerNorm 只在解码可视化/采样分支用思想，NLL 主干不用——可逆链必须纯耦合，
        否则往返 x→z→x̂ 会被归一化吃掉信息（FAQ1）。
        """
        z, logdet = self.forward(x)
        logpz = -0.5 * (z.pow(2).sum(dim=1) + self.dim * torch.log(torch.tensor(2 * torch.pi)))
        return (-(logpz + logdet)).mean()

    @torch.no_grad()
    def sample(self, n, device="cpu", temperature=1.0):
        self.eval()
        z = torch.randn(n, self.dim, device=device) * temperature
        x, _ = self.forward(z, reverse=True)
        return x.view(-1, 1, 28, 28)
