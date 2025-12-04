# src/segmentation/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """U-Net中的双卷积块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """简化版U-Net - 修正版"""

    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器路径
        for feature in features:
            if len(self.downs) == 0:
                self.downs.append(DoubleConv(in_channels, feature))
            else:
                self.downs.append(DoubleConv(features[len(self.downs) - 1], feature))

        # 瓶颈层
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 解码器路径
        for idx in range(len(features) - 1, 0, -1):
            # 上采样
            self.ups.append(
                nn.ConvTranspose2d(
                    features[idx] * 2 if idx == len(features) - 1 else features[idx],
                    features[idx - 1],
                    kernel_size=2,
                    stride=2
                )
            )
            # 双卷积
            self.ups.append(
                DoubleConv(
                    features[idx] + features[idx - 1] if idx == len(features) - 1 else features[idx],
                    features[idx - 1]
                )
            )

        # 最终的1x1卷积
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # 编码器路径
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # 瓶颈层
        x = self.bottleneck(x)

        # 解码器路径
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            # 上采样
            x = self.ups[idx](x)

            # 获取对应的跳跃连接
            skip_idx = idx // 2
            skip_connection = skip_connections[skip_idx]

            # 调整尺寸（如果需要）
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            # 拼接跳跃连接
            x = torch.cat([skip_connection, x], dim=1)

            # 双卷积
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


# 更简单的版本，避免复杂逻辑
class SimpleUNet(nn.Module):
    """简化版U-Net，避免复杂的通道计算"""

    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()

        # 编码器
        self.enc1 = DoubleConv(in_channels, features[0])
        self.enc2 = DoubleConv(features[0], features[1])
        self.enc3 = DoubleConv(features[1], features[2])
        self.enc4 = DoubleConv(features[2], features[3])

        # 池化
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 瓶颈
        self.bottleneck = DoubleConv(features[3], features[3] * 2)

        # 解码器
        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(features[3] * 2, features[3])  # enc4 + up4

        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(features[2] * 2, features[2])  # enc3 + up3

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(features[1] * 2, features[1])  # enc2 + up2

        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(features[0] * 2, features[0])  # enc1 + up1

        # 输出
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # 瓶颈
        b = self.bottleneck(self.pool(e4))

        # 解码器
        d4 = self.up4(b)
        # 调整尺寸
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        # 输出
        return self.final_conv(d1)


# 测试UNet是否正确
def test_unet():
    """测试UNet模型"""
    # 测试SimpleUNet
    print("测试SimpleUNet...")
    model = SimpleUNet(in_channels=3, out_channels=1, features=[32, 64, 128, 256])

    # 创建测试输入
    x = torch.randn(1, 3, 256, 256)

    try:
        output = model(x)
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        print("✓ SimpleUNet测试通过!")

        # 测试TorchScript
        print("\n测试TorchScript导出...")
        traced_model = torch.jit.trace(model, x)
        traced_model.save("test_model.pt")
        print("✓ TorchScript导出成功!")

        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


# 使用SimpleUNet作为默认UNet
UNet = SimpleUNet

if __name__ == "__main__":
    test_unet()