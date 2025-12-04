"""
光照校正模块
处理低光照、阴影、过曝等问题
"""
import cv2
import numpy as np

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class IlluminationCorrector:
    """光照校正器"""

    def __init__(self, config):
        self.config = config

    def apply_gamma_correction(self, image: np.ndarray,
                               gamma: float = 1.2) -> np.ndarray:
        """
        应用伽马校正

        Args:
            image: 输入图像
            gamma: 伽马值
                gamma < 1: 变亮
                gamma > 1: 变暗
                gamma = 1: 无变化

        Returns:
            校正后的图像
        """
        # 确保图像为浮点类型
        if image.dtype != np.float32:
            image_float = image.astype(np.float32) / 255.0
        else:
            image_float = image.copy()

        # 应用伽马校正
        corrected = np.power(image_float, 1.0 / gamma)

        # 转换回原始类型
        if image.dtype != np.float32:
            corrected = (corrected * 255).astype(image.dtype)

        return corrected

    def adjust_brightness(self, image: np.ndarray,
                          delta: int = 30) -> np.ndarray:
        """
        调整亮度

        Args:
            image: 输入图像
            delta: 亮度调整值
                delta > 0: 增加亮度
                delta < 0: 减少亮度

        Returns:
            调整后的图像
        """
        # 使用OpenCV的亮度调整
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 调整V通道（亮度）
        if delta > 0:
            hsv[:, :, 2] = cv2.add(hsv[:, :, 2], delta)
        else:
            hsv[:, :, 2] = cv2.add(hsv[:, :, 2], delta)

        # 确保值在有效范围内
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

        # 转换回BGR
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return enhanced

    def apply_homomorphic_filtering(self, image: np.ndarray,
                                    gamma_low: float = 0.5,
                                    gamma_high: float = 2.0,
                                    cutoff: float = 30,
                                    order: float = 2) -> np.ndarray:
        """
        应用同态滤波（处理不均匀光照）

        Args:
            image: 输入图像
            gamma_low: 低频增益
            gamma_high: 高频增益
            cutoff: 截止频率
            order: 滤波器阶数

        Returns:
            滤波后的图像
        """
        if len(image.shape) == 3:
            # 分别处理每个通道
            channels = cv2.split(image)
            enhanced_channels = []

            for channel in channels:
                enhanced = self._homomorphic_filter_single(
                    channel.astype(np.float32),
                    gamma_low, gamma_high, cutoff, order
                )
                enhanced_channels.append(enhanced)

            enhanced = cv2.merge(enhanced_channels)
        else:
            enhanced = self._homomorphic_filter_single(
                image.astype(np.float32),
                gamma_low, gamma_high, cutoff, order
            )

        return enhanced.astype(image.dtype)

    def _homomorphic_filter_single(self, image: np.ndarray,
                                   gamma_low: float, gamma_high: float,
                                   cutoff: float, order: float) -> np.ndarray:
        """单通道同态滤波"""
        # 1. 对数变换
        image_log = np.log1p(image)

        # 2. 傅里叶变换
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        # 创建高斯高通滤波器
        u = np.arange(rows).reshape(-1, 1)
        v = np.arange(cols).reshape(1, -1)
        D = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
        H = (gamma_high - gamma_low) * (1 - np.exp(-(D ** 2) / (2 * cutoff ** 2))) + gamma_low

        # 3. 应用滤波器
        f = np.fft.fft2(image_log)
        fshift = np.fft.fftshift(f)
        fshift_filtered = fshift * H
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_filtered = np.fft.ifft2(f_ishift)

        # 4. 指数变换
        img_filtered = np.real(img_filtered)
        enhanced = np.expm1(np.abs(img_filtered))

        # 5. 归一化
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

        return enhanced

    def remove_shadows(self, image: np.ndarray,
                       kernel_size: int = 21,
                       method: str = "morphological") -> np.ndarray:
        """
        移除阴影

        Args:
            image: 输入图像
            kernel_size: 结构元素大小
            method: 方法选择
                "morphological": 形态学方法
                "illumination": 光照估计方法

        Returns:
            移除阴影后的图像
        """
        if method == "morphological":
            return self._remove_shadows_morphological(image, kernel_size)
        elif method == "illumination":
            return self._remove_shadows_illumination(image)
        else:
            raise ValueError(f"未知的阴影移除方法: {method}")

    def _remove_shadows_morphological(self, image: np.ndarray,
                                      kernel_size: int) -> np.ndarray:
        """形态学阴影移除"""
        if len(image.shape) == 3:
            # 转换为灰度图进行阴影检测
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. 计算背景光照估计（通过形态学开操作）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        background = cv2.medianBlur(background, kernel_size)

        # 2. 计算阴影掩码
        shadow_mask = (gray.astype(float) / (background.astype(float) + 1e-10) < 0.8).astype(np.uint8) * 255

        # 3. 对阴影区域进行亮度补偿
        if len(image.shape) == 3:
            enhanced = image.copy()
            for c in range(3):
                channel = image[:, :, c].astype(float)
                bg_channel = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, kernel)
                ratio = bg_channel / (channel + 1e-10)
                enhanced_channel = channel * ratio
                enhanced_channel = np.clip(enhanced_channel, 0, 255)
                enhanced[:, :, c] = enhanced_channel.astype(np.uint8)
        else:
            ratio = background.astype(float) / (gray.astype(float) + 1e-10)
            enhanced = gray.astype(float) * ratio
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        return enhanced

    def _remove_shadows_illumination(self, image: np.ndarray) -> np.ndarray:
        """基于光照估计的阴影移除"""
        # 使用Retinex理论估计光照分量
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
        else:
            l = image.copy()

        # 估计光照分量（通过高斯模糊）
        illumination = cv2.GaussianBlur(l, (0, 0), 80)

        # 计算反射分量
        reflection = l.astype(float) / (illumination.astype(float) + 1e-10)

        # 归一化反射分量
        reflection = cv2.normalize(reflection, None, 0, 255, cv2.NORM_MINMAX)

        if len(image.shape) == 3:
            # 合并LAB通道
            lab_enhanced = cv2.merge([reflection.astype(np.uint8), a, b])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        else:
            enhanced = reflection.astype(np.uint8)

        return enhanced

    def correct_exposure(self, image: np.ndarray,
                         target_mean: int = 128,
                         method: str = "histogram") -> np.ndarray:
        """
        曝光校正

        Args:
            image: 输入图像
            target_mean: 目标平均亮度
            method: 校正方法
                "histogram": 直方图匹配
                "linear": 线性拉伸
                "adaptive": 自适应校正

        Returns:
            曝光校正后的图像
        """
        if method == "linear":
            return self._correct_exposure_linear(image, target_mean)
        elif method == "histogram":
            return self._correct_exposure_histogram(image, target_mean)
        elif method == "adaptive":
            return self._correct_exposure_adaptive(image)
        else:
            raise ValueError(f"未知的曝光校正方法: {method}")

    def _correct_exposure_linear(self, image: np.ndarray,
                                 target_mean: int) -> np.ndarray:
        """线性曝光校正"""
        if len(image.shape) == 3:
            # 计算当前平均亮度（使用亮度通道）
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            current_mean = np.mean(hsv[:, :, 2])
        else:
            current_mean = np.mean(image)

        # 计算调整系数
        ratio = target_mean / (current_mean + 1e-10)

        # 应用线性调整
        enhanced = np.clip(image.astype(float) * ratio, 0, 255).astype(np.uint8)

        return enhanced

    def _correct_exposure_histogram(self, image: np.ndarray,
                                    target_mean: int) -> np.ndarray:
        """直方图曝光校正"""
        if len(image.shape) == 3:
            # 处理彩色图像
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv)

            # 计算当前和目标直方图
            current_hist, _ = np.histogram(y.flatten(), 256, [0, 256])
            target_hist = np.ones(256) * (y.size / 256)

            # 直方图匹配
            y_enhanced = self._histogram_matching(y, target_hist)

            # 合并通道
            yuv_enhanced = cv2.merge([y_enhanced, u, v])
            enhanced = cv2.cvtColor(yuv_enhanced, cv2.COLOR_YUV2BGR)
        else:
            # 灰度图像
            current_hist, _ = np.histogram(image.flatten(), 256, [0, 256])
            target_hist = np.ones(256) * (image.size / 256)
            enhanced = self._histogram_matching(image, target_hist)

        return enhanced

    def _correct_exposure_adaptive(self, image: np.ndarray) -> np.ndarray:
        """自适应曝光校正"""
        # 使用自适应伽马校正
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 计算图像统计
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)

        # 自适应计算伽马值
        if mean_intensity < 80:
            # 低光照：增加亮度
            gamma = 0.7
        elif mean_intensity > 180:
            # 过曝：减少亮度
            gamma = 1.5
        else:
            # 正常光照：轻微调整
            gamma = 1.0 + (128 - mean_intensity) / 256

        # 应用伽马校正
        enhanced = self.apply_gamma_correction(image, gamma)

        return enhanced

    def _histogram_matching(self, source: np.ndarray,
                            target_hist: np.ndarray) -> np.ndarray:
        """直方图匹配"""
        # 计算源图像的累积直方图
        source_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
        source_cdf = source_hist.cumsum()
        source_cdf_normalized = source_cdf / source_cdf[-1]

        # 计算目标累积直方图
        target_cdf = target_hist.cumsum()
        target_cdf_normalized = target_cdf / target_cdf[-1]

        # 创建映射函数
        mapping = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            j = 255
            while j >= 0 and source_cdf_normalized[i] <= target_cdf_normalized[j]:
                mapping[i] = j
                j -= 1

        # 应用映射
        matched = mapping[source]

        return matched

    def apply_tone_mapping(self, image: np.ndarray,
                           method: str = "reinhard") -> np.ndarray:
        """
        应用色调映射（处理高动态范围）

        Args:
            image: 输入图像
            method: 色调映射方法
                "reinhard": Reinhard方法
                "durand": Durand方法
                "drago": Drago方法

        Returns:
            色调映射后的图像
        """
        if method == "reinhard":
            return self._tone_mapping_reinhard(image)
        elif method == "durand":
            return self._tone_mapping_durand(image)
        elif method == "drago":
            return self._tone_mapping_drago(image)
        else:
            raise ValueError(f"未知的色调映射方法: {method}")

    def _tone_mapping_reinhard(self, image: np.ndarray) -> np.ndarray:
        """Reinhard色调映射"""
        # 转换为浮点型
        img_float = image.astype(np.float32) / 255.0

        # 计算对数平均亮度
        if len(img_float.shape) == 3:
            gray = 0.299 * img_float[:, :, 2] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 0]
        else:
            gray = img_float

        delta = 1e-6
        log_mean = np.exp(np.mean(np.log(gray + delta)))

        # Reinhard色调映射
        scaled = 0.18 / log_mean * gray
        mapped = scaled / (1 + scaled)

        if len(img_float.shape) == 3:
            # 应用于每个通道
            enhanced = img_float.copy()
            for c in range(3):
                ratio = img_float[:, :, c] / (gray + delta)
                enhanced[:, :, c] = mapped * ratio
        else:
            enhanced = mapped

        # 转换回8位
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)

        return enhanced

    def _tone_mapping_durand(self, image: np.ndarray) -> np.ndarray:
        """Durand色调映射（快速双边滤波）"""
        # 转换为浮点型
        img_float = image.astype(np.float32) / 255.0

        if len(img_float.shape) == 3:
            # 转换为灰度计算亮度
            gray = 0.299 * img_float[:, :, 2] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 0]
        else:
            gray = img_float

        # 计算对数亮度
        log_luminance = np.log10(gray + 1e-6)

        # 使用双边滤波分解为基础层和细节层
        base = cv2.bilateralFilter(log_luminance.astype(np.float32), 9, 0.4, 5)
        detail = log_luminance - base

        # 压缩动态范围
        max_base = np.max(base)
        min_base = np.min(base)
        compressed_base = (base - min_base) / (max_base - min_base + 1e-6)

        # 重新组合
        result_log = compressed_base + detail

        # 指数变换
        result = np.power(10, result_log)

        if len(img_float.shape) == 3:
            # 保持颜色
            enhanced = img_float.copy()
            for c in range(3):
                ratio = img_float[:, :, c] / (gray + 1e-6)
                enhanced[:, :, c] = result * ratio
        else:
            enhanced = result

        # 归一化
        enhanced = cv2.normalize(enhanced, None, 0, 1, cv2.NORM_MINMAX)
        enhanced = (enhanced * 255).astype(np.uint8)

        return enhanced

    def _tone_mapping_drago(self, image: np.ndarray) -> np.ndarray:
        """Drago色调映射"""
        # 简化的Drago方法
        img_float = image.astype(np.float32) / 255.0

        # Drago参数
        bias = 0.85
        exposure = 1.0

        if len(img_float.shape) == 3:
            # 计算亮度
            luminance = 0.299 * img_float[:, :, 2] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 0]
        else:
            luminance = img_float

        # 归一化亮度
        luminance = luminance / (luminance.max() + 1e-6)

        # Drago映射函数
        mapped = luminance / (luminance + (1 + bias * luminance / exposure) ** (-bias))

        if len(img_float.shape) == 3:
            # 保持颜色
            enhanced = img_float.copy()
            for c in range(3):
                ratio = img_float[:, :, c] / (luminance + 1e-6)
                enhanced[:, :, c] = mapped * ratio
        else:
            enhanced = mapped

        # 转换回8位
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)

        return enhanced