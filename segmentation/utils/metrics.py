# src/segmentation/utils/metrics.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """Dice Loss + BCE Loss 的组合损失函数"""

    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # BCE损失
        bce_loss = self.bce_loss(inputs, targets)

        # Dice损失
        inputs_sigmoid = torch.sigmoid(inputs)
        intersection = (inputs_sigmoid * targets).sum(dim=(1, 2, 3))
        union = inputs_sigmoid.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        # 组合损失
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return total_loss


class IoUScore:
    """IoU（交并比）计算"""

    def __init__(self, threshold=0.5, smooth=1e-6):
        self.threshold = threshold
        self.smooth = smooth

    def __call__(self, inputs, targets):
        if inputs.max() > 1 or inputs.min() < 0:
            inputs = torch.sigmoid(inputs)

        preds = (inputs > self.threshold).float()
        targets = (targets > self.threshold).float()

        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        return iou.mean().item()