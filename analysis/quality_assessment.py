from typing import Dict

import cv2
import numpy as np


class BRISQUE:
    """BRISQUE无参考图像质量评估"""

    def __init__(self):
        # BRISQUE预训练参数（简化版，实际使用时需加载完整模型）
        self.scale_range = 2
        self.estimated_mean = [
            0.248685, 0.252099, 0.231983, 0.211215,
            0.237374, 0.229680, 0.232379, 0.195558,
            0.234069, 0.218349, 0.211020, 0.209599,
            0.201074, 0.205156, 0.201392, 0.191373,
            0.200142, 0.192587, 0.193441, 0.183444,
            0.186867, 0.182603, 0.178177, 0.173789,
            0.174613, 0.176051, 0.171697, 0.167344,
            0.168868, 0.166107, 0.164854, 0.162279,
            0.161358, 0.159318, 0.158655, 0.156838
        ]

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """提取BRISQUE特征"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        features = []

        # 多尺度特征提取
        for scale in range(self.scale_range):
            if scale > 0:
                h, w = image.shape
                image = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)

            # MSCN系数
            mscn_coefficients = self._calculate_mscn(image)

            # 提取特征
            features.extend(self._extract_ggd_features(mscn_coefficients))
            features.extend(self._extract_pair_product_features(mscn_coefficients))

        return np.array(features)

    def _calculate_mscn(self, image: np.ndarray) -> np.ndarray:
        """计算MSCN系数"""
        image = image.astype(np.float64)

        # 局部均值和方差
        mu = cv2.GaussianBlur(image, (7, 7), 7 / 6)
        mu_sq = mu * mu
        sigma = cv2.GaussianBlur(image * image, (7, 7), 7 / 6)
        sigma = np.sqrt(np.abs(sigma - mu_sq))

        # MSCN系数
        mscn = (image - mu) / (sigma + 1)
        return mscn

    def _extract_ggd_features(self, coefficients: np.ndarray) -> list:
        """提取广义高斯分布特征"""
        # 简化实现，实际需要拟合GGD参数
        return [np.mean(coefficients), np.std(coefficients)]

    def _extract_pair_product_features(self, coefficients: np.ndarray) -> list:
        """提取配对乘积特征"""
        # 四个方向
        shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
        features = []

        for shift in shifts:
            shifted = np.roll(coefficients, shift, axis=(0, 1))
            product = coefficients * shifted
            features.extend([np.mean(product), np.std(product)])

        return features

    def score(self, image: np.ndarray) -> float:
        """计算BRISQUE分数（越低越好）"""
        features = self.extract_features(image)

        # 简化的分数计算（实际需要SVR模型）
        # 这里使用特征与预训练均值的距离作为分数
        score = np.sqrt(np.sum((features - self.estimated_mean[:len(features)]) ** 2))

        # 归一化到0-100范围
        score = min(max(score * 10, 0), 100)
        return score


class ImageQualityAssessment:
    """图像质量综合评估"""

    def __init__(self):
        self.brisque = BRISQUE()

    def assess(self, image: np.ndarray) -> Dict:
        """综合质量评估"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        results = {}

        # 1. BRISQUE分数
        results['brisque'] = self.brisque.score(image)

        # 2. 结构清晰度（基于梯度）
        results['sharpness'] = self._calculate_sharpness(gray)

        # 3. 对比度
        results['contrast'] = self._calculate_contrast(gray)

        # 4. 噪声水平
        results['noise_level'] = self._estimate_noise(gray)

        # 5. 信息熵
        results['entropy'] = self._calculate_entropy(gray)

        # 6. 综合质量分数
        results['overall_quality'] = self._calculate_overall_quality(results)

        return results

    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """计算清晰度（基于拉普拉斯方差）"""
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return laplacian.var()

    def _calculate_contrast(self, image: np.ndarray) -> float:
        """计算对比度（Michelson对比度）"""
        # 确保使用float64避免溢出
        image_float = image.astype(np.float64)
        min_val = np.min(image_float)
        max_val = np.max(image_float)

        # 避免除零和溢出
        denominator = max_val + min_val
        if abs(denominator) < 1e-10:
            return 0.0

        # 计算对比度
        contrast = (max_val - min_val) / denominator

        # 确保结果在合理范围内
        return float(np.clip(contrast, 0.0, 1.0))

    def _estimate_noise(self, image: np.ndarray) -> float:
        """估计噪声水平"""
        # 使用小波方法
        coeffs = cv2.dct(np.float32(image))
        coeffs = np.abs(coeffs)
        coeffs = np.sort(coeffs.flatten())[::-1]
        # 取高频部分估计噪声
        noise_estimate = np.median(coeffs[int(len(coeffs) * 0.95):])
        return noise_estimate

    def _calculate_entropy(self, image: np.ndarray) -> float:
        """计算信息熵"""
        hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy

    def _calculate_overall_quality(self, metrics: Dict) -> float:
        """计算综合质量分数（0-100，越高越好）"""
        # 权重分配
        weights = {
            'brisque': 0.4,  # 逆指标，越低越好
            'sharpness': 0.3,  # 正指标，越高越好
            'contrast': 0.2,  # 正指标
            'entropy': 0.1  # 正指标
        }

        # 归一化各项指标
        normalized = {}

        # BRISQUE: 0-100 → 100-0（反转）
        normalized['brisque'] = max(0, 100 - metrics['brisque'])

        # 清晰度：0-1000 → 0-100
        normalized['sharpness'] = min(metrics['sharpness'] / 10, 100)

        # 对比度：0-1 → 0-100
        normalized['contrast'] = metrics['contrast'] * 100

        # 信息熵：0-8 → 0-100
        normalized['entropy'] = min(metrics['entropy'] / 8 * 100, 100)

        # 加权平均
        overall = sum(normalized[key] * weights[key] for key in weights)

        return overall