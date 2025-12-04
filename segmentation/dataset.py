# src/segmentation/dataset.py
import os
import random
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class EnhancedDefectDataset(Dataset):
    """完全使用增强图像的缺陷分割数据集"""

    def __init__(self,
                 enhanced_image_dir: str,
                 label_dir: str,
                 phase: str = 'train',
                 image_size: tuple = (512, 512),
                 augment: bool = True):

        self.enhanced_image_dir = Path(enhanced_image_dir)
        self.label_dir = Path(label_dir)
        self.phase = phase
        self.image_size = image_size
        self.augment = augment and phase == 'train'

        # 获取所有增强图像
        self.image_files = sorted([
            f for f in self.enhanced_image_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])

        # 验证标签对应关系
        self.valid_pairs = []
        for img_file in self.image_files:
            label_file = self.label_dir / (img_file.stem + '.txt')
            if label_file.exists():
                self.valid_pairs.append((img_file, label_file))

        print(f"{phase}: Found {len(self.valid_pairs)} valid image-label pairs")

        # 数据增强
        self.transform = self.get_transforms()

    def get_transforms(self):
        """根据albumentations版本兼容性创建transform"""
        if self.augment:
            try:
                # 尝试新版本的参数格式 - 使用元组
                return A.Compose([
                    # 新版本RandomResizedCrop需要将height和width作为元组传递
                    A.RandomResizedCrop(
                        height=self.image_size[0],
                        width=self.image_size[1],
                        scale=(0.8, 1.0),
                        p=0.5
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.2,
                        scale_limit=0.2,
                        rotate_limit=30,
                        p=0.8
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=0.5
                    ),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                    A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
                    A.CoarseDropout(
                        max_holes=5,
                        max_height=24,
                        max_width=24,
                        fill_value=0,
                        p=0.3
                    ),
                    A.Resize(height=self.image_size[0], width=self.image_size[1]),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
            except Exception as e:
                # 如果新版本失败，尝试旧版本格式
                print(f"警告: 新版本参数失败，尝试旧版本: {e}")
                try:
                    return A.Compose([
                        # 旧版本RandomResizedCrop - 使用元组
                        A.RandomResizedCrop(self.image_size, scale=(0.8, 1.0), p=0.5),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        A.ShiftScaleRotate(
                            shift_limit=0.2,
                            scale_limit=0.2,
                            rotate_limit=30,
                            p=0.8
                        ),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2,
                            contrast_limit=0.2,
                            p=0.5
                        ),
                        A.Resize(self.image_size[0], self.image_size[1]),
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2(),
                    ])
                except Exception as e2:
                    # 如果两种格式都失败，使用最基本的变换
                    print(f"警告: 两种格式都失败，使用基本变换: {e2}")
                    return A.Compose([
                        A.Resize(height=self.image_size[0], width=self.image_size[1]),
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2(),
                    ])
        else:
            # 验证和测试时只做基本处理
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_path, label_path = self.valid_pairs[idx]

        try:
            # 加载增强图像
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"无法读取图像: {img_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 从YOLO标签创建掩码
            from .utils.yolo_to_mask import YOLOToMaskConverter
            mask = YOLOToMaskConverter.create_mask_from_yolo(str(label_path), image.shape)
            mask = (mask > 127).astype(np.uint8) * 255

            # 应用变换
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            else:
                # 如果没有transform，只做基本的resize
                image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
                mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]))
                # 转换为tensor
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                mask = torch.from_numpy(mask).float() / 255.0

            # 掩码归一化 - 添加通道维度
            mask = mask.float() / 255.0
            mask = mask.unsqueeze(0)  # 添加通道维度，从 [H,W] 变为 [1,H,W]

            return image, mask

        except Exception as e:
            print(f"处理数据时出错 {img_path}: {e}")
            # 返回空白数据作为占位符
            image = torch.zeros((3, self.image_size[0], self.image_size[1]))
            mask = torch.zeros((1, self.image_size[0], self.image_size[1]))
            return image, mask

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])  # [batch, 3, H, W]
        masks = torch.stack([item[1] for item in batch])  # [batch, 1, H, W]

        # 确保掩码有正确的维度
        if masks.dim() == 4:
            # 已经是 [batch, 1, H, W] 格式，不需要修改
            pass
        elif masks.dim() == 3:
            # 如果是 [batch, H, W]，添加通道维度
            masks = masks.unsqueeze(1)  # 变为 [batch, 1, H, W]

        return images, masks