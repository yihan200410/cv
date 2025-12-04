import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
import numpy as np
from typing import Optional, Tuple, Dict, Any


class DiceLoss(nn.Module):
    """Dice Loss - 专注于分割区域重叠"""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            inputs: 预测值 (logits), shape [B, 1, H, W]
            targets: 真实标签, shape [B, 1, H, W]

        Returns:
            Dice loss值
        """
        inputs = torch.sigmoid(inputs)

        # 展平张量
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (
                inputs_flat.sum() + targets_flat.sum() + self.smooth
        )

        return 1 - dice_score


class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)

        # 计算二元交叉熵
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # Focal loss权重
        pt = inputs * targets + (1 - inputs) * (1 - targets)
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()


class FocalDiceLoss(nn.Module):
    """Focal + Dice 组合损失 - 平衡难易样本并关注区域重叠"""

    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 dice_weight: float = 0.7,
                 focal_weight: float = 0.3,
                 smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Dice Loss
        inputs_sigmoid = torch.sigmoid(inputs)
        intersection = (inputs_sigmoid * targets).sum(dim=(1, 2, 3))
        union = inputs_sigmoid.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        # Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss.mean()

        # 组合损失
        total_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss

        return total_loss


class TverskyLoss(nn.Module):
    """Tversky Loss - 控制精确率和召回率的平衡"""

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
        """
        Args:
            alpha: 假阳性的权重 (控制精确率)
            beta: 假阴性的权重 (控制召回率)
            alpha=0.7, beta=0.3 更注重召回率
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)

        # True Positives, False Positives, False Negatives
        tp = (inputs * targets).sum(dim=(1, 2, 3))
        fp = (inputs * (1 - targets)).sum(dim=(1, 2, 3))
        fn = ((1 - inputs) * targets).sum(dim=(1, 2, 3))

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        tversky_loss = 1 - tversky.mean()

        return tversky_loss


class IoULoss(nn.Module):
    """IoU Loss - 直接优化IoU指标"""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)

        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        union = inputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        iou_loss = 1 - iou.mean()

        return iou_loss


class ComboLoss(nn.Module):
    """组合损失：Dice + BCE + L2正则化"""

    def __init__(self,
                 dice_weight: float = 0.7,
                 bce_weight: float = 0.3,
                 l2_weight: float = 0.01,
                 smooth: float = 1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.l2_weight = l2_weight
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Dice Loss
        inputs_sigmoid = torch.sigmoid(inputs)
        intersection = (inputs_sigmoid * targets).sum(dim=(1, 2, 3))
        union = inputs_sigmoid.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)

        # L2正则化（通过权重衰减实现，这里仅为展示）
        total_loss = (self.dice_weight * dice_loss +
                      self.bce_weight * bce_loss)

        return total_loss


class IoUScore:
    """IoU（交并比）计算器"""

    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        self.threshold = threshold
        self.smooth = smooth

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """计算IoU

        Args:
            inputs: 预测值 (logits), shape [B, 1, H, W]
            targets: 真实标签, shape [B, 1, H, W]

        Returns:
            IoU值
        """
        # 如果输入是logits，先进行sigmoid
        if inputs.max() > 1 or inputs.min() < 0:
            inputs = torch.sigmoid(inputs)

        preds = (inputs > self.threshold).float()
        targets = (targets > 0.5).float()

        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        return iou.mean().item()


class DiceScore:
    """Dice系数计算器"""

    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        self.threshold = threshold
        self.smooth = smooth

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        if inputs.max() > 1 or inputs.min() < 0:
            inputs = torch.sigmoid(inputs)

        preds = (inputs > self.threshold).float()
        targets = (targets > 0.5).float()

        intersection = (preds * targets).sum(dim=(1, 2, 3))
        dice = (2. * intersection + self.smooth) / (
                preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + self.smooth
        )

        return dice.mean().item()


class Trainer:
    """U-Net训练器 - 支持多种损失函数和优化策略"""

    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 device: str = 'cuda',
                 loss_type: str = 'focal_dice',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 patience: int = 15):
        """
        Args:
            model: U-Net模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 训练设备
            loss_type: 损失函数类型
            learning_rate: 学习率
            weight_decay: 权重衰减
            patience: 早停耐心值
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 选择损失函数
        self.loss_type = loss_type
        self.criterion = self._get_loss_function(loss_type)

        # 优化器配置
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
            last_epoch=-1
        )

        # 指标计算器
        self.iou_calculator = IoUScore()
        self.dice_calculator = DiceScore()

        # 训练状态
        self.best_val_iou = 0.0
        self.best_val_dice = 0.0
        self.patience_counter = 0
        self.patience = patience
        self.epoch_history = {
            'train_loss': [],
            'train_iou': [],
            'val_loss': [],
            'val_iou': [],
            'val_dice': [],
            'learning_rate': []
        }

        # 创建检查点目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = Path(f"checkpoints/unet_{loss_type}_{timestamp}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard日志目录
        self.log_dir = Path(f"logs/tensorboard_{loss_type}_{timestamp}")
        self.writer = SummaryWriter(str(self.log_dir))

        # 打印配置信息
        self._print_config(learning_rate, weight_decay)

    def _get_loss_function(self, loss_type: str) -> nn.Module:
        """获取损失函数"""
        loss_functions = {
            'bce': nn.BCEWithLogitsLoss(),
            'dice': DiceLoss(),
            'focal': FocalLoss(),
            'focal_dice': FocalDiceLoss(dice_weight=0.7, focal_weight=0.3),
            'tversky': TverskyLoss(alpha=0.7, beta=0.3),
            'iou': IoULoss(),
            'combo': ComboLoss(),
        }

        if loss_type not in loss_functions:
            raise ValueError(f"不支持的损失函数类型: {loss_type}")

        print(f"使用 {loss_type} 损失函数")
        return loss_functions[loss_type]

    def _print_config(self, lr: float, wd: float):
        """打印训练配置"""
        print("\n" + "=" * 50)
        print("训练配置")
        print("=" * 50)
        print(f"模型架构: {self.model.__class__.__name__}")
        print(f"设备: {self.device}")
        print(f"损失函数: {self.loss_type}")
        print(f"学习率: {lr}")
        print(f"权重衰减: {wd}")
        print(f"早停耐心值: {self.patience}")
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        print(f"验证样本数: {len(self.val_loader.dataset)}")
        print(f"检查点保存路径: {self.checkpoint_dir}")
        print(f"TensorBoard日志路径: {self.log_dir}")
        print("=" * 50 + "\n")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_iou = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'训练 Epoch {epoch + 1}',
                    leave=False, position=0)

        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            self.optimizer.step()

            # 计算指标
            with torch.no_grad():
                iou = self.iou_calculator(outputs, masks)
                dice = self.dice_calculator(outputs, masks)

            epoch_loss += loss.item()
            epoch_iou += iou
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{iou:.4f}',
                'Dice': f'{dice:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

            # TensorBoard记录（每100个batch记录一次）
            if batch_idx % 100 == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)
                self.writer.add_scalar('Train/Batch_IoU', iou, global_step)
                self.writer.add_scalar('Train/Batch_Dice', dice, global_step)

        return epoch_loss / num_batches, epoch_iou / num_batches

    def validate(self) -> Tuple[float, float, float]:
        """验证"""
        self.model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        num_batches = 0

        with torch.no_grad():
            val_pbar = tqdm(self.val_loader, desc='验证',
                            leave=False, position=0)

            for images, masks in val_pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                iou = self.iou_calculator(outputs, masks)
                dice = self.dice_calculator(outputs, masks)

                val_loss += loss.item()
                val_iou += iou
                val_dice += dice
                num_batches += 1

                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'IoU': f'{iou:.4f}',
                    'Dice': f'{dice:.4f}'
                })

        return (val_loss / num_batches,
                val_iou / num_batches,
                val_dice / num_batches)

    def save_checkpoint(self,
                        epoch: int,
                        val_iou: float,
                        val_dice: float,
                        is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_iou': val_iou,
            'val_dice': val_dice,
            'loss_type': self.loss_type,
            'epoch_history': self.epoch_history,
            'best_val_iou': self.best_val_iou,
            'best_val_dice': self.best_val_dice,
        }

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pth')
        else:
            torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth')

    def train(self, epochs: int = 50) -> Dict[str, Any]:
        """完整训练过程

        Args:
            epochs: 训练轮数

        Returns:
            训练结果字典
        """
        print(f"开始训练，共 {epochs} 轮...")

        for epoch in range(epochs):
            # 训练一个epoch
            train_loss, train_iou = self.train_epoch(epoch)

            # 验证
            val_loss, val_iou, val_dice = self.validate()

            # 更新学习率
            self.scheduler.step(val_iou)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 保存历史记录
            self.epoch_history['train_loss'].append(train_loss)
            self.epoch_history['train_iou'].append(train_iou)
            self.epoch_history['val_loss'].append(val_loss)
            self.epoch_history['val_iou'].append(val_iou)
            self.epoch_history['val_dice'].append(val_dice)
            self.epoch_history['learning_rate'].append(current_lr)

            # 打印epoch结果
            print(f"\nEpoch {epoch + 1}/{epochs}:")
            print(f"  训练 - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}")
            print(f"  验证 - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
            print(f"  学习率: {current_lr:.6f}")

            # TensorBoard记录
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('IoU/Train', train_iou, epoch)
            self.writer.add_scalar('IoU/Val', val_iou, epoch)
            self.writer.add_scalar('Dice/Val', val_dice, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)

            # 保存最佳模型
            if val_iou > self.best_val_iou:
                self.best_val_iou = val_iou
                self.best_val_dice = val_dice
                self.patience_counter = 0

                self.save_checkpoint(epoch, val_iou, val_dice, is_best=True)
                print(f"  ✅ 最佳模型保存，IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
            else:
                self.patience_counter += 1
                print(f"  ⏳ 无改善 ({self.patience_counter}/{self.patience})")

            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_iou, val_dice, is_best=False)
                print(f"  💾 检查点保存")

            # 早停检查
            if self.patience_counter >= self.patience:
                print(f"\n⚠️  早停触发！{self.patience}个epoch无改善")
                break

        # 训练完成
        self.writer.close()

        # 保存最终模型
        final_checkpoint = {
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_iou': val_iou,
            'val_dice': val_dice,
            'loss_type': self.loss_type,
            'epoch_history': self.epoch_history,
            'best_val_iou': self.best_val_iou,
            'best_val_dice': self.best_val_dice,
        }
        torch.save(final_checkpoint, self.checkpoint_dir / 'final_model.pth')

        # 打印总结
        print("\n" + "=" * 50)
        print("训练完成!")
        print("=" * 50)
        print(f"最佳验证IoU: {self.best_val_iou:.4f}")
        print(f"最佳验证Dice: {self.best_val_dice:.4f}")
        print(f"最终验证IoU: {val_iou:.4f}")
        print(f"最终验证Dice: {val_dice:.4f}")
        print(f"总训练轮数: {epoch + 1}")
        print(f"模型保存路径: {self.checkpoint_dir}")
        print("=" * 50)

        return {
            'best_iou': self.best_val_iou,
            'best_dice': self.best_val_dice,
            'final_iou': val_iou,
            'final_dice': val_dice,
            'epoch_history': self.epoch_history,
            'checkpoint_dir': str(self.checkpoint_dir)
        }

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_val_iou = checkpoint['best_val_iou']
        self.best_val_dice = checkpoint['best_val_dice']
        self.epoch_history = checkpoint['epoch_history']

        print(f"加载检查点: {checkpoint_path}")
        print(f"Epoch: {checkpoint['epoch']}")
        print(f"验证IoU: {checkpoint['val_iou']:.4f}")
        print(f"验证Dice: {checkpoint['val_dice']:.4f}")


# 使用示例
if __name__ == "__main__":
    from datetime import datetime
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    # 示例用法
    print("示例用法:")
    print("1. 准备数据集和数据加载器")
    print("2. 初始化U-Net模型")
    print("3. 创建Trainer实例")
    print("4. 开始训练")

    # 伪代码示例
    """
    # 假设你已经有了这些组件
    train_loader = DataLoader(...)
    val_loader = DataLoader(...)
    model = UNet(in_channels=3, out_channels=1)

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        loss_type='focal_dice',  # 可以选择: 'bce', 'dice', 'focal', 'focal_dice', 'tversky', 'iou', 'combo'
        learning_rate=1e-4,
        weight_decay=1e-4,
        patience=15
    )

    # 开始训练
    results = trainer.train(epochs=50)

    # 或者加载检查点继续训练
    # trainer.load_checkpoint('checkpoints/unet_focal_dice/best_model.pth')
    # results = trainer.train(epochs=30)
    """