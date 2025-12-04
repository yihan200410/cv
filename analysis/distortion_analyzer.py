from dataclasses import dataclass
from typing import Dict, Tuple, List

import cv2
import numpy as np

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class DistortionMetrics:
    """失真度量结果"""
    blur_score: float
    noise_score: float
    contrast_score: float
    brightness_mean: float
    brightness_std: float
    shadow_coverage: float
    gradient_magnitude: float
    entropy: float


class DistortionAnalyzer:
    """图像失真分析器"""

    def __init__(self, config):
        self.config = config
        self.thresholds = config.DISTORTION_THRESHOLDS

    def analyze(self, image: np.ndarray) -> Dict:
        """
        综合分析图像失真类型

        Args:
            image: 输入图像 (BGR或灰度)

        Returns:
            dict: 包含失真类型和程度的分析结果
        """
        # 转换为灰度图（如果是彩色图）
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 计算各项失真指标
        metrics = self._compute_metrics(gray)

        # 判断失真类型
        distortion_types = self._classify_distortion(metrics)

        return {
            "metrics": metrics,
            "distortion_types": distortion_types,
            "recommendations": self._generate_recommendations(distortion_types)
        }

    def _compute_metrics(self, gray_image: np.ndarray) -> DistortionMetrics:
        """计算所有失真指标"""
        # 1. 模糊程度（基于拉普拉斯方差）
        blur_score = self._compute_blur_score(gray_image)

        # 2. 噪声水平（基于局部方差）
        noise_score = self._compute_noise_score(gray_image)

        # 3. 对比度
        contrast_score = self._compute_contrast_score(gray_image)

        # 4. 亮度统计
        brightness_mean = np.mean(gray_image)
        brightness_std = np.std(gray_image)

        # 5. 阴影覆盖率
        shadow_coverage = self._compute_shadow_coverage(gray_image)

        # 6. 梯度幅度（边缘强度）
        gradient_magnitude = self._compute_gradient_magnitude(gray_image)

        # 7. 图像熵（信息量）
        entropy = self._compute_entropy(gray_image)

        return DistortionMetrics(
            blur_score=blur_score,
            noise_score=noise_score,
            contrast_score=contrast_score,
            brightness_mean=brightness_mean,
            brightness_std=brightness_std,
            shadow_coverage=shadow_coverage,
            gradient_magnitude=gradient_magnitude,
            entropy=entropy
        )

    def _compute_blur_score(self, image: np.ndarray) -> float:
        """计算模糊分数（拉普拉斯方差）"""
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        # 归一化到0-1范围
        return 1 / (1 + laplacian_var * 0.01)

    def _compute_noise_score(self, image: np.ndarray) -> float:
        """计算噪声水平"""
        try:
            from scipy.ndimage import uniform_filter
            mean_filtered = uniform_filter(image, size=3)

            # 避免空数组或全零数组
            if image.size == 0 or np.all(image == 0):
                return 0.0

            variance = np.mean((image - mean_filtered) ** 2)

            # 避免负值或极小值
            if variance <= 0:
                return 0.0

            return min(np.sqrt(variance) / 255.0, 1.0)
        except:
            return 0.0

    def _compute_contrast_score(self, image: np.ndarray) -> float:
        """计算对比度（RMS对比度）"""
        mean_intensity = np.mean(image)
        contrast = np.sqrt(np.mean((image - mean_intensity) ** 2))
        return contrast

    def _compute_shadow_coverage(self, image: np.ndarray) -> float:
        """计算阴影覆盖率"""
        # 使用OTSU阈值分割阴影区域
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 低于阈值的区域视为阴影
        shadow_threshold = np.percentile(image, 30)
        shadow_pixels = np.sum(image < shadow_threshold)
        return shadow_pixels / image.size

    def _compute_gradient_magnitude(self, image: np.ndarray) -> float:
        """计算平均梯度幅度"""
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        return np.mean(gradient_magnitude)

    def _compute_entropy(self, image: np.ndarray) -> float:
        """计算图像熵"""
        histogram, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
        histogram = histogram / histogram.sum()
        entropy = -np.sum(histogram * np.log2(histogram + 1e-10))
        return entropy

    def _classify_distortion(self, metrics: DistortionMetrics) -> Dict[str, Tuple[str, float]]:
        """根据指标分类失真类型"""
        distortions = {}

        # 1. 模糊判断
        if metrics.blur_score > self.thresholds["blur_threshold"]:
            distortions["blur"] = ("severe" if metrics.blur_score > 0.3 else "moderate",
                                   metrics.blur_score)

        # 2. 噪声判断
        if metrics.noise_score > self.thresholds["noise_threshold"]:
            distortions["noise"] = ("severe" if metrics.noise_score > 0.1 else "moderate",
                                    metrics.noise_score)

        # 3. 对比度过低
        if metrics.contrast_score < self.thresholds["contrast_threshold"]:
            distortions["low_contrast"] = ("severe" if metrics.contrast_score < 15 else "moderate",
                                           metrics.contrast_score)

        # 4. 光照问题
        if metrics.brightness_mean < self.thresholds["brightness_low"]:
            distortions["low_light"] = ("severe" if metrics.brightness_mean < 30 else "moderate",
                                        metrics.brightness_mean)
        elif metrics.brightness_mean > self.thresholds["brightness_high"]:
            distortions["over_exposure"] = ("severe" if metrics.brightness_mean > 220 else "moderate",
                                            metrics.brightness_mean)

        # 5. 阴影问题
        if metrics.shadow_coverage > self.thresholds["shadow_threshold"]:
            distortions["shadow"] = ("severe" if metrics.shadow_coverage > 0.5 else "moderate",
                                     metrics.shadow_coverage)

        # 6. 细节缺失（低熵）
        if metrics.entropy < 5.0:
            distortions["low_detail"] = ("severe" if metrics.entropy < 3.0 else "moderate",
                                         metrics.entropy)

        return distortions

    def _generate_recommendations(self, distortions: Dict) -> List[str]:
        """根据失真类型生成增强建议"""
        recommendations = []

        if "blur" in distortions:
            severity, score = distortions["blur"]
            recommendations.append(f"应用锐化算法（模糊程度：{severity}）")

        if "noise" in distortions:
            severity, score = distortions["noise"]
            recommendations.append(f"应用降噪算法（噪声水平：{severity}）")

        if "low_contrast" in distortions:
            severity, score = distortions["low_contrast"]
            recommendations.append(f"应用对比度增强（对比度过低：{severity}）")

        if "low_light" in distortions:
            severity, score = distortions["low_light"]
            recommendations.append(f"应用亮度调整（光照不足：{severity}）")

        if "shadow" in distortions:
            severity, score = distortions["shadow"]
            recommendations.append(f"应用阴影校正（阴影覆盖：{score:.2%}）")

        if "over_exposure" in distortions:
            severity, score = distortions["over_exposure"]
            recommendations.append(f"应用曝光校正（过曝：{severity}）")

        if "low_detail" in distortions:
            severity, score = distortions["low_detail"]
            recommendations.append(f"应用细节增强（信息熵低：{score:.2f}）")

        if not distortions:
            recommendations.append("图像质量良好，建议保持原状或轻微增强")

        return recommendations