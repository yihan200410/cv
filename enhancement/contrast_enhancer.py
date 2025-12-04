from typing import Tuple, List

import cv2
import numpy as np


class ContrastEnhancer:
    """对比度增强算法"""

    def __init__(self, config):
        self.config = config

    def apply_clahe(self, image: np.ndarray,
                    clip_limit: float = 2.0,
                    grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        应用CLAHE（限制对比度自适应直方图均衡化）

        Args:
            image: 输入图像
            clip_limit: 对比度限制阈值
            grid_size: 网格大小

        Returns:
            增强后的图像
        """
        if len(image.shape) == 3:
            # 处理彩色图像
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # 只对L通道应用CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            l_enhanced = clahe.apply(l)

            # 合并通道并转换回BGR
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        else:
            # 灰度图像
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            enhanced = clahe.apply(image)

        return enhanced

    def apply_histogram_equalization(self, image: np.ndarray,
                                     method: str = "adaptive") -> np.ndarray:
        """
        应用直方图均衡化

        Args:
            image: 输入图像
            method: 均衡化方法 ("global", "adaptive", "contrast")

        Returns:
            均衡化后的图像
        """
        if len(image.shape) == 3:
            # 彩色图像：转换到YUV空间，只对Y通道均衡化
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv)

            if method == "global":
                y_eq = cv2.equalizeHist(y)
            elif method == "adaptive":
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                y_eq = clahe.apply(y)
            else:  # contrast
                y_eq = self._contrast_limited_histogram_equalization(y)

            enhanced = cv2.merge([y_eq, u, v])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_YUV2BGR)
        else:
            if method == "global":
                enhanced = cv2.equalizeHist(image)
            elif method == "adaptive":
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(image)
            else:
                enhanced = self._contrast_limited_histogram_equalization(image)

        return enhanced

    def apply_retinex(self, image: np.ndarray,
                      scales: List[int] = None,
                      alpha: float = 125,
                      beta: float = 46,
                      gain: float = 1.0) -> np.ndarray:
        """
        应用Retinex算法（单尺度或多尺度）

        Args:
            image: 输入图像
            scales: 高斯核尺度列表
            alpha: 控制对数变换的参数
            beta: 控制对数变换的参数
            gain: 增益系数

        Returns:
            Retinex增强后的图像
        """
        if scales is None:
            scales = [15, 80, 250]

        if len(image.shape) == 3:
            # 分别处理每个通道
            channels = cv2.split(image)
            enhanced_channels = []

            for channel in channels:
                retinex = self._single_scale_retinex(channel.astype(np.float32),
                                                     scales[0], alpha, beta)
                enhanced_channels.append(retinex)

            enhanced = cv2.merge(enhanced_channels)
        else:
            enhanced = self._single_scale_retinex(image.astype(np.float32),
                                                  scales[0], alpha, beta)

        # 增益调整
        enhanced = np.clip(enhanced * gain, 0, 255).astype(np.uint8)

        return enhanced

    def _single_scale_retinex(self, image: np.ndarray,
                              sigma: int,
                              alpha: float,
                              beta: float) -> np.ndarray:
        """单尺度Retinex实现"""
        # 高斯模糊
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)

        # 避免除零
        blurred = np.maximum(blurred, 1)

        # Retinex公式
        retinex = alpha * np.log(image + 1) - beta * np.log(blurred)

        # 归一化到0-255
        retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)

        return retinex.astype(np.uint8)

    def apply_contrast_stretch(self, image: np.ndarray,
                               lower_percent: float = 2,
                               upper_percent: float = 98) -> np.ndarray:
        """
        应用对比度拉伸

        Args:
            image: 输入图像
            lower_percent: 下限百分位
            upper_percent: 上限百分位

        Returns:
            对比度拉伸后的图像
        """
        # 计算百分位
        lower = np.percentile(image, lower_percent)
        upper = np.percentile(image, upper_percent)

        # 线性拉伸
        stretched = np.clip((image - lower) * 255.0 / (upper - lower + 1e-10), 0, 255)

        return stretched.astype(np.uint8)

    def _contrast_limited_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """对比度限制的直方图均衡化"""
        # 计算直方图
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])

        # 计算累积分布函数
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()

        # 创建查找表
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')

        # 应用变换
        enhanced = cdf[image]

        return enhanced

    def apply_adaptive_histogram(self, image: np.ndarray,
                                 method: str = "clahe",
                                 clip_limit: float = 2.0,
                                 grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        应用自适应直方图均衡化

        Args:
            image: 输入图像
            method: 自适应方法 ("clahe", "local")
            clip_limit: 对比度限制阈值（仅对clahe有效）
            grid_size: 网格大小（仅对clahe有效）

        Returns:
            增强后的图像
        """
        if method.lower() == "clahe":
            # 使用CLAHE方法
            return self.apply_clahe(image, clip_limit, grid_size)
        else:
            # 使用局部直方图均衡化
            if len(image.shape) == 3:
                # 转换为灰度进行处理
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                enhanced_gray = self._local_histogram_equalization(gray)
                # 保持彩色信息
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                lab_enhanced = cv2.merge([enhanced_gray, a, b])
                enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            else:
                enhanced = self._local_histogram_equalization(image)

            return enhanced

    def _local_histogram_equalization(self, image: np.ndarray,
                                      window_size: int = 64) -> np.ndarray:
        """局部直方图均衡化"""
        h, w = image.shape
        enhanced = np.zeros_like(image)

        # 滑动窗口处理
        for i in range(0, h, window_size // 2):
            for j in range(0, w, window_size // 2):
                # 计算窗口边界
                top = max(0, i)
                bottom = min(h, i + window_size)
                left = max(0, j)
                right = min(w, j + window_size)

                # 提取窗口区域
                window = image[top:bottom, left:right]

                # 对窗口进行直方图均衡化
                if window.size > 0:
                    window_eq = cv2.equalizeHist(window)
                    enhanced[top:bottom, left:right] = window_eq

        return enhanced

    def apply_multi_scale_retinex(self, image: np.ndarray,
                                  scales: List[int] = None,
                                  gain: float = 1.0) -> np.ndarray:
        """
        应用多尺度Retinex

        Args:
            image: 输入图像
            scales: 高斯核尺度列表
            gain: 增益系数

        Returns:
            Retinex增强后的图像
        """
        if scales is None:
            scales = [15, 80, 250]

        if len(image.shape) == 3:
            # 分别处理每个通道
            channels = cv2.split(image)
            enhanced_channels = []

            for channel in channels:
                m_retinex = self._multi_scale_retinex_single(
                    channel.astype(np.float32), scales
                )
                enhanced_channels.append(m_retinex)

            enhanced = cv2.merge(enhanced_channels)
        else:
            enhanced = self._multi_scale_retinex_single(
                image.astype(np.float32), scales
            )

        # 增益调整
        enhanced = np.clip(enhanced * gain, 0, 255).astype(np.uint8)

        return enhanced

    def _multi_scale_retinex_single(self, image: np.ndarray,
                                    scales: List[int]) -> np.ndarray:
        """单通道多尺度Retinex"""
        retinex_sum = np.zeros_like(image)

        for sigma in scales:
            # 高斯模糊
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            blurred = np.maximum(blurred, 1)

            # 计算单尺度Retinex
            single_scale = np.log(image + 1) - np.log(blurred)
            retinex_sum += single_scale

        # 平均并归一化
        retinex_mean = retinex_sum / len(scales)
        retinex_mean = cv2.normalize(retinex_mean, None, 0, 255, cv2.NORM_MINMAX)

        return retinex_mean.astype(np.uint8)