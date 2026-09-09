"""02 CNN 家族模型定义（随项目推进逐步扩充：LeNet → AlexNet/VGG/NiN → ResNet → DenseNet/Inception → ...）。

28×28 适配说明：AlexNet 原版 227×227、VGG 原版 224×224。本家族在 CPU 上训练，
保持架构精神（块的层数、通道比例、Dropout、池化次数）而缩小空间分辨率与通道数，
模型名统一以 Mini 标记缩通道版本。
"""
import torch
import torch.nn as nn


class LeNet(nn.Module):
    """LeNet-5（LeCun, 1998）现代复现版：tanh→ReLU、RBF 输出层→Linear+CrossEntropy。

    原版输入 32×32；此处为 MNIST 28×28 适配——conv1 加 padding=2 保持 28。
    输入 (N, 1, 28, 28)，可训练参数 61,706（对照：01 家族 MLP 同任务 235,146）。
    """

    def __init__(self, in_ch: int = 1, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 6, 5, padding=2), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class AlexNetMini(nn.Module):
    """AlexNet（Krizhevsky, 2012）精神复现版：5 卷积 + 3 全连接 + ReLU/Dropout/MaxPool。

    原版 227×227 大 stride 双 GPU；此处 28×28 缩通道 32→96→192→128→128，
    Dropout 0.5 保留（AlexNet 的招牌），参数量约 1.29M。
    """

    def __init__(self, in_ch: int = 1, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 14
            nn.Conv2d(32, 96, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),        # 7
            nn.Conv2d(96, 192, 3, padding=1), nn.ReLU(),
            nn.Conv2d(192, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # 3
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def vgg_block(in_ch: int, out_ch: int, n_convs: int):
    """VGG 基础块：n 个 3×3 卷积（同通道）+ 1 个 2×2 MaxPool。"""
    layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU()]
    for _ in range(n_convs - 1):
        layers += [nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU()]
    layers.append(nn.MaxPool2d(2))
    return layers


class VGGMini(nn.Module):
    """VGG（Simonyan, 2014）精神复现版：全 3×3 卷积堆叠。

    原版 224×224 下 5 个池化到 7×7；28×28 输入放 4 个池化（28→14→7→3→1），
    块结构 [2,2,3,2]（VGG-11 的 28×28 适配），无 BN——保留原版训练难点作为教学素材。
    """

    def __init__(self, in_ch: int = 1, num_classes: int = 10):
        super().__init__()
        feats = vgg_block(in_ch, 32, 2)      # 28→14
        feats += vgg_block(32, 64, 2)        # 14→7
        feats += vgg_block(64, 128, 3)       # 7→3
        feats += vgg_block(128, 256, 2)      # 3→1
        self.features = nn.Sequential(*feats)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class NiNBlock(nn.Module):
    """NiN 基础块：Conv → 两个 1×1 Conv（像素级全连接）→ MaxPool。"""

    def __init__(self, in_ch: int, out_ch: int, k: int, pad: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=pad), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 1), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 1), nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class NiNMini(nn.Module):
    """NiN（Lin, 2013）：1×1 卷积 + 全局平均池化（GAP）代替全连接头部。

    参数量极小（约 32 万）但 2013 年没有 BN，深层 1×1 堆叠在小 lr 下收敛很慢
    （原论文即报告调参敏感）——本项目的如实教学素材，也是 03 项目 BN 消融的伏笔。
    """

    def __init__(self, in_ch: int = 1, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            NiNBlock(in_ch, 48, 5, pad=2),      # 28→14
            NiNBlock(48, 96, 3, pad=1),         # 14→7
            NiNBlock(96, 192, 3, pad=1),        # 7→3
            nn.Conv2d(192, num_classes, 3, padding=1), nn.ReLU(),   # (C,3,3)
            nn.AdaptiveAvgPool2d(1),            # 全局平均池化 → (C,1,1)
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------
# 03 项目：ResNet / PlainNet（受控消融专用，残差与 BN 可开关）
# ---------------------------------------------------------------
class BasicBlock(nn.Module):
    """两层 3×3 的残差基础块（ResNet-18/34 同款，CIFAR 缩通道版）。

    residual=False 时退化为同参数量的 Plain 块（唯一差别：没有 x + F(x) 这条捷径）。
    stride>1 时捷径路径用 1×1 卷积做下采样对齐（ResNet 论文的 option B）。
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 residual: bool = True, use_bn: bool = True, dw1: bool = False):
        super().__init__()
        self.residual = residual

        def norm(ch):
            return nn.BatchNorm2d(ch) if use_bn else nn.Identity()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1,
                               groups=in_ch if dw1 else 1, bias=not use_bn)
        self.bn1 = norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=not use_bn)
        self.bn2 = norm(out_ch)
        self.act = nn.ReLU(inplace=True)

        if residual and (stride != 1 or in_ch != out_ch):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=not use_bn),
                norm(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + self.shortcut(x)
        return self.act(out)


class ResNetCIFAR(nn.Module):
    """He et al., 2015 的 CIFAR 风格主干，通道 16→32→64（原论文的缩放）。

    num_blocks=[3,3,3] 对应 ResNet-20（6n+2）；三阶段各含一个 stride=2
    下采样（32→16→8），GAP 头出分类。
    开关设计（消融的本体，保证各变体逐模块参数/前向路径一致）：
      residual=False → PlainNet（同深度同通道，无捷径，研究退化梯度）
      use_bn=False   → 无 BN 版（研究归一化的贡献，呼应 02 项目 NiN 的困境）
      se=True        → 每个残差块加 SE 通道注意力（04 项目选学，通往 Transformer 注意力的桥）
    """

    def __init__(self, num_blocks=(3, 3, 3), num_classes: int = 10,
                 residual: bool = True, use_bn: bool = True, se: bool = False):
        super().__init__()
        self.residual, self.use_bn, self.se = residual, use_bn, se
        self.conv_in = nn.Conv2d(3, 16, 3, padding=1, bias=not use_bn)
        self.bn_in = nn.BatchNorm2d(16) if use_bn else nn.Identity()

        blocks, in_ch = [], 16
        for stage, (nb, out_ch, stride) in enumerate(
                zip(num_blocks, (16, 32, 64), (1, 2, 2))):
            for b in range(nb):
                if se:
                    blocks.append(BasicBlockWithSE(in_ch, out_ch,
                                                   stride=stride if b == 0 else 1,
                                                   residual=residual, use_bn=use_bn))
                else:
                    blocks.append(BasicBlock(in_ch, out_ch,
                                             stride=stride if b == 0 else 1,
                                             residual=residual, use_bn=use_bn))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.bn_in(self.conv_in(x))
        out = self.blocks(out)
        out = self.gap(out).flatten(1)
        return self.fc(out)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力（Hu et al., 2018）。

    对每个通道：全局平均池化（Squeeze）→ 瓶颈两层全连接（Excitation，r=4）→
    sigmoid 权重 → 逐通道缩放。让网络自己学习"哪些通道重要"。
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = x.mean((2, 3))                     # (B, C)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class BasicBlockWithSE(BasicBlock):
    """带 SE 通道注意力的残差块（主特征加完残差后做通道重标定）。"""

    def __init__(self, *args, se: bool = True, **kw):
        super().__init__(*args, **kw)
        self.se = SEBlock(kw.get("out_ch", args[1])) if se else None

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + self.shortcut(x)
        if self.se is not None:
            out = self.se(out)
        return self.act(out)


# ---------------------------------------------------------------
# 04 项目：DenseNet / Inception（连接拓扑对比）
# ---------------------------------------------------------------
class DenseLayer(nn.Module):
    """DenseNet 单层：BN → ReLU → 3×3 Conv，输出 growth 个新通道（拼接旧特征）。"""

    def __init__(self, in_ch: int, growth: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, growth, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return torch.cat([x, self.layer(x)], 1)


class TransitionLayer(nn.Module):
    """DenseNet Transition：BN → ReLU → 1×1 压缩(θ=0.5) → AvgPool 减半分辨率。"""

    def __init__(self, in_ch: int, theta: float = 0.5):
        super().__init__()
        out_ch = int(in_ch * theta)
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        return self.layer(x)


class DenseNetCIFAR(nn.Module):
    """DenseNet（Huang et al., 2017）CIFAR 缩通道版。

    3 个 DenseBlock（各 6 层，growth=12）+ 2 个 Transition（θ=0.5）。
    通道推演见 notebook（stem 16 → 88 → 44 → 116 → 58 → 130 → GAP）。
    参数量约 17 万——凭"特征复用"以 ResNet20 约 63% 的参数量出分类。
    """

    def __init__(self, num_classes: int = 10, growth: int = 12,
                 n_layers: int = 6, theta: float = 0.5, in_ch: int = 16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_ch, 3, padding=1, bias=False), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True))

        blocks, ch = [], in_ch
        for stage in range(3):
            for _ in range(n_layers):
                blocks.append(DenseLayer(ch, growth))
                ch += growth
            if stage < 2:
                blocks.append(TransitionLayer(ch, theta))
                ch = int(ch * theta)
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(ch, num_classes)

    def forward(self, x):
        out = self.features(self.stem(x))
        return self.fc(self.head(out))


class InceptionBlock(nn.Module):
    """Inception 基础块（GoogLeNet, 2014）：四分支并行 → 通道拼接。

    1×1 | 1×1→3×3 | 1×1→5×5 | MaxPool→1×1。1×1 先降维（"降维再卷"）省参数。
    """

    def __init__(self, in_ch: int, b1: int, r3: int, b3: int, r5: int, b5: int, bp: int):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_ch, b1, 1), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, r3, 1), nn.ReLU(inplace=True),
            nn.Conv2d(r3, b3, 3, padding=1), nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, r5, 1), nn.ReLU(inplace=True),
            nn.Conv2d(r5, b5, 5, padding=2), nn.ReLU(inplace=True))
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, bp, 1), nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


class InceptionCIFAR(nn.Module):
    """GoogLeNet 精神复现版（CIFAR 缩通道）：stem + 2 个 Inception 块 + GAP。

    四分支 16/32/16/16 → 80 通道，参数量约 3.6 万（思想重于规模）。
    原版深度与通道更多，此处轻量化保证 CPU 可承受。
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(
            InceptionBlock(32, 16, 16, 32, 8, 16, 16),    # → 80
            InceptionBlock(80, 32, 24, 48, 12, 24, 16),   # →120
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(120, num_classes)

    def forward(self, x):
        out = self.blocks(self.stem(x))
        return self.fc(self.head(out))


# ---------------------------------------------------------------
# 05 项目：MobileNet / EfficientNet（效率时代）
# ---------------------------------------------------------------
class MobileBlock(nn.Module):
    """MobileNetV1 基本单元：Depthwise 3×3 → BN → ReLU → Pointwise 1×1 → BN → ReLU。

    标准 3×3 卷积的参数/FLOPs ≈ 9·Cin·Cout，分解后 ≈ 9·Cin + Cin·Cout——3×3 时约省 8~9 倍。
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MobileNetCIFAR(nn.Module):
    """MobileNetV1（Howard et al., 2017）CIFAR 精神复现版：全 DW+PW 分解。

    alpha 是原论文的宽度旋钮（统一缩放每层通道）；分辨率旋钮 ρ 在 32×32 上无下探空间，固定。
    FLOPs 约为同规模标准卷积网的 1/4（ResNet20 41M MACs → 约 10M）。
    """

    def __init__(self, num_classes: int = 10, alpha: float = 1.0):
        super().__init__()
        c = lambda x: max(4, int(x * alpha))
        self.stem = nn.Sequential(
            nn.Conv2d(3, c(32), 3, padding=1, bias=False), nn.BatchNorm2d(c(32)), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(
            MobileBlock(c(32), c(64), stride=1),     # 32×32
            MobileBlock(c(64), c(128), stride=2),    # 16×16
            MobileBlock(c(128), c(256), stride=2),   # 8×8
            MobileBlock(c(256), c(512), stride=2),   # 4×4
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(c(512), num_classes)

    def forward(self, x):
        return self.fc(self.head(self.blocks(self.stem(x))))


class MBConv(nn.Module):
    """MobileNetV2 倒残差块 + SE（即 EfficientNet 的 MBConv）。

    PW 扩张(expand 倍) → DW 3×3 → SE 通道注意力 → PW 投影（线性，无激活）；
    stride=1 且 in==out 时加残差。"先扩后缩"与 ResNet 瓶颈方向相反，故称倒残差。
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand: int = 2):
        super().__init__()
        mid = in_ch * expand
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.blocks = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            SEBlock(mid, reduction=4),
            nn.Conv2d(mid, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),  # 线性投影
        )

    def forward(self, x):
        out = self.blocks(x)
        return x + out if self.use_res else out


class EfficientNetCIFAR(nn.Module):
    """EfficientNet（Tan & Le, 2019）CIFAR mini 版：MBConv 堆叠 + 缩放旋钮。

    width_mult/depth_mult 对应原论文的 β/α（分辨率维度 γ 在 32×32 上无下探空间，固定）。
    复合缩放的故事：单轴放大遭遇边际递减，等比协同放大更优——notebook 用三臂对照演示。
    """

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0,
                 depth_mult: float = 1.0, expand: int = 2):
        super().__init__()
        c = lambda x: max(8, int(x * width_mult))
        n = lambda k: max(1, round(k * depth_mult))
        self.stem = nn.Sequential(
            nn.Conv2d(3, c(16), 3, padding=1, bias=False), nn.BatchNorm2d(c(16)), nn.ReLU(inplace=True))

        blocks, in_ch = [], c(16)
        for out_ch, k, stride in [(c(24), 1, 1), (c(40), 1, 2), (c(64), 1, 2)]:
            for b in range(n(k)):
                blocks.append(MBConv(in_ch if b == 0 else out_ch, out_ch,
                                     stride=stride if b == 0 else 1, expand=expand))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(c(64), num_classes)

    def forward(self, x):
        return self.fc(self.head(self.blocks(self.stem(x))))


# ---------------------------------------------------------------
# 06 项目：ConvNeXt——用 Transformer 的经验重造 CNN（现代化路径）
# ---------------------------------------------------------------
class LayerNorm2d(nn.Module):
    """(N,C,H,W) 上的逐位置 LayerNorm（等价 channels_last 的 nn.LayerNorm）。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt 块（Liu et al., 2022）：DW 7×7 → LN → PW×4 → GELU → PW → γ 缩放 → (+x)。

    与 ResNet BasicBlock 的五处差异（notebook 逐步消融）：大核 DW、LayerNorm、
    倒瓶颈（扩张 4×）、GELU、逐通道缩放 γ（大模型用 1e-6，mini 此处取 1.0 保短训活力）。
    """

    def __init__(self, dim: int, expand: int = 4, gamma_init: float = 1.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False)
        self.norm = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, dim * expand, 1, bias=False)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * expand, dim, 1, bias=False)
        self.gamma = nn.Parameter(torch.full((dim,), gamma_init))

    def forward(self, x):
        res = x
        out = self.pw2(self.act(self.pw1(self.norm(self.dwconv(x)))))
        return res + self.gamma.view(1, -1, 1, 1) * out


class ConvNeXtCIFAR(nn.Module):
    """ConvNeXt 精神复现版（CIFAR mini）：4×4 patchify stem + 3 阶段 + 分离下采样。

    现代化五件套全部内置：patchify stem、DW 7×7、LayerNorm、倒瓶颈、GELU。
    dims=(32,64,128)、depths=(2,2,2)、分辨率 32→8→4→2（patchify 4×4 直接到 8×8）。
    """

    def __init__(self, num_classes: int = 10, dims=(32, 64, 128), depths=(2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], 4, stride=4, bias=False), LayerNorm2d(dims[0]))

        layers, in_ch = [], dims[0]
        for si, (d, n) in enumerate(zip(dims, depths)):
            if si > 0:
                layers.append(nn.Sequential(LayerNorm2d(in_ch),
                                            nn.Conv2d(in_ch, d, 2, stride=2, bias=False)))
                in_ch = d
            for _ in range(n):
                layers.append(ConvNeXtBlock(d))
        self.stages = nn.Sequential(*layers)
        self.norm = LayerNorm2d(dims[-1])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        out = self.stages(self.stem(x))
        return self.fc(self.head(self.norm(out)))


class InvertedBlock(nn.Module):
    """倒瓶颈块（MobileNetV2/ConvNeXt 方向）：1×1 扩张×4 → DW k×k → 1×1 投影。

    ResNet 瓶颈"先缩后扩"，此块"先扩后缩"——中间层在低分辨率下做重计算。
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, kernel: int = 3):
        super().__init__()
        mid = in_ch * 4
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel, stride=stride, padding=kernel // 2,
                      groups=mid, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
        )
        self.use_res = stride == 1 and in_ch == out_ch

    def forward(self, x):
        out = self.net(x)
        return x + out if self.use_res else out


class ModernPathCIFAR(nn.Module):
    """现代化路径驱动器（ResNet20 骨架逐档换块）：path='dw'|'inv3'|'inv7'。

    'dw' = 3×3 卷积换深度可分离（ResNeXt-ify）；'inv3' = 换倒瓶颈；'inv7' = 倒瓶颈+大核 7×7。
    骨架（stem/stages 16→32→64/下采样/GAP）与 ResNetCIFAR 完全一致——每步只动"块内部"。
    """

    def __init__(self, num_classes: int = 10, path: str = "dw", num_blocks=(3, 3, 3)):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(16)
        blocks, in_ch = [], 16
        for nb, out_ch, stride in zip(num_blocks, (16, 32, 64), (1, 2, 2)):
            for b in range(nb):
                if path == "dw":
                    blocks.append(BasicBlock(in_ch, out_ch,
                                             stride=stride if b == 0 else 1, dw1=True))
                else:
                    blocks.append(InvertedBlock(in_ch, out_ch,
                                                stride=stride if b == 0 else 1,
                                                kernel=3 if path == "inv3" else 7))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = torch.relu(self.bn_in(self.conv_in(x)))
        out = self.blocks(out)
        return self.fc(self.gap(out).flatten(1))
