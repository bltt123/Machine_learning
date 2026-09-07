# 03 CNN 下游任务：分割与检测

> 分类之上的两个核心视觉任务。评估体系完全不同：分割看 IoU/Dice，检测看 mAP。

## 学习目标

1. 理解语义分割（FCN → U-Net）与目标检测（two-stage vs one-stage）两大技术路线
2. 掌握分割/检测的专用指标：mIoU、Dice、mAP、NMS、Anchor
3. 学会用 torchvision 自带预训练模型做推理与微调（生产实际做法）

## 算法清单（学习检查表）

### 分割
- [ ] FCN（全卷积、上采样）
- [ ] U-Net（编码器-解码器 + 跳跃连接）★重点

### 检测
- [ ] Faster R-CNN（two-stage 代表：RPN + RoI）
- [ ] YOLO 系列（one-stage 代表：grid 回归）
- [ ] RetinaNet（Focal Loss 解决类别不均衡）
- [ ] Mask R-CNN（实例分割 = 检测 + 分割头）

## 项目规划

| 编号 | 项目 | 覆盖算法 | 数据集 | 关键实验 |
|---|---|---|---|---|
| 01 | `01_FCN_UNet_Segmentation` | FCN, U-Net | Oxford-IIIT Pet（torchvision 自带分割掩码） | 自训 U-Net，IoU/Dice 评估 |
| 02 | `02_Detection_torchvision` | Faster R-CNN, RetinaNet, Mask R-CNN | 小型自选数据 | 预训练推理 + 可视化 + mAP 理解 |
| 03 | `03_YOLO_Concept` | YOLO 思想 | —（以原理+小实验为主） | NMS 手写实现 |

> CPU 提示：检测任务以"预训练模型推理 + 可视化 + 指标理解"为主，避免大规模训练。

## 与其他家族的关系

- **前置**：02 的 ResNet / U-Net 编码器思想
- **后续**：07 生成模型中的 Diffusion 也用 U-Net 骨干（可回头呼应）
