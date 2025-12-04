"""
边缘增强模块
处理模糊、边缘不清晰等问题
"""
import cv2
import numpy as np
import pywt  # 小波变换

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class EdgeEnhancer:
    """边缘增强器"""

    def __init__(self, config):
        self.config = config

    def apply_unsharp_mask(self, image: np.ndarray,
                           amount: float = 1.5,
                           sigma: float = 1.0,
                           threshold: float = 0) -> np.ndarray:
        """
        应用非锐化掩模

        Args:
            image: 输入图像
            amount: 锐化强度
            sigma: 高斯模糊标准差
            threshold: 锐化阈值

        Returns:
            锐化后的图像
        """
        # 高斯模糊
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)

        # 计算细节层
        if threshold > 0:
            # 带阈值的锐化
            detail = image.astype(float) - blurred.astype(float)
            mask = np.abs(detail) > threshold
            sharpened = image.astype(float) + amount * detail * mask
        else:
            # 标准非锐化掩模
            sharpened = image.astype(float) + amount * (image.astype(float) - blurred.astype(float))

        # 确保值在有效范围内
        sharpened = np.clip(sharpened, 0, 255).astype(image.dtype)

        return sharpened

    def apply_laplacian_sharpen(self, image: np.ndarray,
                                kernel_size: int = 3,
                                scale: float = 0.5) -> np.ndarray:
        """
        应用拉普拉斯锐化

        Args:
            image: 输入图像
            kernel_size: 拉普拉斯核大小
            scale: 锐化强度

        Returns:
            锐化后的图像
        """
        # 应用拉普拉斯算子
        if len(image.shape) == 3:
            # 分别处理每个通道
            channels = cv2.split(image)
            enhanced_channels = []

            for channel in channels:
                laplacian = cv2.Laplacian(channel, cv2.CV_64F, ksize=kernel_size)
                enhanced = channel.astype(float) - scale * laplacian
                enhanced_channels.append(enhanced)

            enhanced = cv2.merge(enhanced_channels)
        else:
            laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)
            enhanced = image.astype(float) - scale * laplacian

        # 确保值在有效范围内
        enhanced = np.clip(enhanced, 0, 255).astype(image.dtype)

        return enhanced

    def apply_bilateral_filter(self, image: np.ndarray,
                               d: int = 9,
                               sigma_color: float = 75,
                               sigma_space: float = 75) -> np.ndarray:
        """
        应用双边滤波（保边降噪）

        Args:
            image: 输入图像
            d: 滤波直径
            sigma_color: 颜色空间标准差
            sigma_space: 坐标空间标准差

        Returns:
            滤波后的图像
        """
        # 应用双边滤波
        filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

        return filtered

    def apply_non_local_means(self, image: np.ndarray,
                              h: float = 10,
                              template_window_size: int = 7,
                              search_window_size: int = 21) -> np.ndarray:
        """
        应用非局部均值降噪

        Args:
            image: 输入图像
            h: 滤波强度参数
            template_window_size: 模板窗口大小
            search_window_size: 搜索窗口大小

        Returns:
            降噪后的图像
        """
        # 应用非局部均值降噪
        denoised = cv2.fastNlMeansDenoising(
            image,
            h=h,
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size
        )

        return denoised

    def apply_guided_filter(self, image: np.ndarray,
                            radius: int = 8,
                            eps: float = 0.01) -> np.ndarray:
        """
        应用引导滤波（边缘保持平滑）

        Args:
            image: 输入图像
            radius: 滤波半径
            eps: 正则化参数

        Returns:
            滤波后的图像
        """

        def guided_filter_single(I, p, radius, eps):
            """单通道引导滤波"""
            # 获取图像尺寸
            h, w = I.shape

            # 均值滤波
            mean_I = cv2.boxFilter(I, -1, (radius, radius))
            mean_p = cv2.boxFilter(p, -1, (radius, radius))
            mean_Ip = cv2.boxFilter(I * p, -1, (radius, radius))

            # 计算协方差
            cov_Ip = mean_Ip - mean_I * mean_p

            # 计算方差
            mean_II = cv2.boxFilter(I * I, -1, (radius, radius))
            var_I = mean_II - mean_I * mean_I

            # 计算a和b
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I

            # 均值滤波a和b
            mean_a = cv2.boxFilter(a, -1, (radius, radius))
            mean_b = cv2.boxFilter(b, -1, (radius, radius))

            # 输出
            q = mean_a * I + mean_b

            return q

        if len(image.shape) == 3:
            # 彩色图像：分别处理每个通道
            channels = cv2.split(image)
            enhanced_channels = []

            for i, channel in enumerate(channels):
                # 使用自身作为引导图像
                enhanced = guided_filter_single(
                    channel.astype(np.float32),
                    channel.astype(np.float32),
                    radius, eps
                )
                enhanced_channels.append(enhanced)

            enhanced = cv2.merge(enhanced_channels)
        else:
            # 灰度图像
            enhanced = guided_filter_single(
                image.astype(np.float32),
                image.astype(np.float32),
                radius, eps
            )

        # 确保值在有效范围内
        enhanced = np.clip(enhanced, 0, 255).astype(image.dtype)

        return enhanced

    def apply_wavelet_denoise(self, image: np.ndarray,
                              wavelet: str = 'db4',
                              level: int = 3,
                              method: str = 'soft') -> np.ndarray:
        """
        应用小波降噪

        Args:
            image: 输入图像
            wavelet: 小波类型
            level: 分解层数
            method: 阈值方法 ('soft', 'hard')

        Returns:
            降噪后的图像
        """
        if len(image.shape) == 3:
            # 分别处理每个通道
            channels = cv2.split(image)
            enhanced_channels = []

            for channel in channels:
                enhanced = self._wavelet_denoise_single(channel, wavelet, level, method)
                enhanced_channels.append(enhanced)

            enhanced = cv2.merge(enhanced_channels)
        else:
            enhanced = self._wavelet_denoise_single(image, wavelet, level, method)

        return enhanced

    def _wavelet_denoise_single(self, image: np.ndarray,
                                wavelet: str, level: int,
                                method: str) -> np.ndarray:
        """单通道小波降噪"""
        # 小波分解
        coeffs = pywt.wavedec2(image, wavelet, level=level)

        # 估计噪声标准差（使用最高频子带）
        detail_coeffs = coeffs[1:]
        sigma = np.median(np.abs([c for sublist in detail_coeffs for row in sublist for c in row.flatten()])) / 0.6745

        # 计算阈值
        threshold = sigma * np.sqrt(2 * np.log(image.size))

        # 应用阈值
        new_coeffs = [coeffs[0]]  # 保留近似系数

        for coeff in coeffs[1:]:
            if isinstance(coeff, tuple):
                # 细节系数（三个方向）
                denoised_detail = []
                for c in coeff:
                    if method == 'soft':
                        denoised = pywt.threshold(c, threshold, mode='soft')
                    else:  # hard
                        denoised = pywt.threshold(c, threshold, mode='hard')
                    denoised_detail.append(denoised)
                new_coeffs.append(tuple(denoised_detail))
            else:
                # 处理特殊情况
                if method == 'soft':
                    denoised = pywt.threshold(coeff, threshold, mode='soft')
                else:
                    denoised = pywt.threshold(coeff, threshold, mode='hard')
                new_coeffs.append(denoised)

        # 小波重构
        denoised = pywt.waverec2(new_coeffs, wavelet)

        # 确保尺寸匹配
        if denoised.shape != image.shape:
            denoised = cv2.resize(denoised, (image.shape[1], image.shape[0]))

        # 确保值在有效范围内
        denoised = np.clip(denoised, 0, 255).astype(image.dtype)

        return denoised

    def apply_edge_preserving_smoothing(self, image: np.ndarray,
                                        lambda_: float = 0.1,
                                        iterations: int = 3) -> np.ndarray:
        """
        应用边缘保持平滑（基于各向异性扩散）

        Args:
            image: 输入图像
            lambda_: 扩散系数
            iterations: 迭代次数

        Returns:
            平滑后的图像
        """

        def anisotropic_diffusion(image, lambda_, iterations):
            """各向异性扩散实现"""
            img = image.astype(np.float32)

            for _ in range(iterations):
                # 计算梯度
                grad_n = np.roll(img, -1, axis=0) - img
                grad_s = np.roll(img, 1, axis=0) - img
                grad_e = np.roll(img, -1, axis=1) - img
                grad_w = np.roll(img, 1, axis=1) - img

                # 计算扩散系数（Perona-Malik）
                k = 10.0
                cn = 1.0 / (1.0 + (grad_n / k) ** 2)
                cs = 1.0 / (1.0 + (grad_s / k) ** 2)
                ce = 1.0 / (1.0 + (grad_e / k) ** 2)
                cw = 1.0 / (1.0 + (grad_w / k) ** 2)

                # 更新图像
                img = img + lambda_ * (cn * grad_n + cs * grad_s + ce * grad_e + cw * grad_w)

            return np.clip(img, 0, 255).astype(image.dtype)

        if len(image.shape) == 3:
            # 分别处理每个通道
            channels = cv2.split(image)
            enhanced_channels = []

            for channel in channels:
                enhanced = anisotropic_diffusion(channel, lambda_, iterations)
                enhanced_channels.append(enhanced)

            enhanced = cv2.merge(enhanced_channels)
        else:
            enhanced = anisotropic_diffusion(image, lambda_, iterations)

        return enhanced

    def apply_multi_scale_enhancement(self, image: np.ndarray,
                                      scales: list = [1, 2, 4],
                                      method: str = "laplacian") -> np.ndarray:
        """
        应用多尺度增强

        Args:
            image: 输入图像
            scales: 尺度列表
            method: 增强方法

        Returns:
            增强后的图像
        """
        if method == "laplacian":
            return self._multi_scale_laplacian(image, scales)
        elif method == "gradient":
            return self._multi_scale_gradient(image, scales)
        else:
            raise ValueError(f"未知的多尺度增强方法: {method}")

    def _multi_scale_laplacian(self, image: np.ndarray, scales: list) -> np.ndarray:
        """多尺度拉普拉斯金字塔增强"""
        # 创建高斯金字塔
        pyramid = [image.astype(np.float32)]
        for scale in scales[1:]:
            reduced = cv2.pyrDown(pyramid[-1])
            pyramid.append(reduced)

        # 创建拉普拉斯金字塔
        laplacian_pyramid = []
        for i in range(len(pyramid) - 1):
            size = (pyramid[i].shape[1], pyramid[i].shape[0])
            expanded = cv2.pyrUp(pyramid[i + 1], dstsize=size)
            laplacian = pyramid[i] - expanded
            laplacian_pyramid.append(laplacian)

        # 增强拉普拉斯系数
        enhanced_laplacian = []
        for i, lap in enumerate(laplacian_pyramid):
            # 根据尺度调整增强强度
            enhancement_factor = 1.0 + 0.5 * i / len(laplacian_pyramid)
            enhanced = lap * enhancement_factor
            enhanced_laplacian.append(enhanced)

        # 重建图像
        reconstructed = pyramid[-1]
        for i in range(len(enhanced_laplacian) - 1, -1, -1):
            size = (enhanced_laplacian[i].shape[1], enhanced_laplacian[i].shape[0])
            reconstructed = cv2.pyrUp(reconstructed, dstsize=size)
            reconstructed = reconstructed + enhanced_laplacian[i]

        # 确保值在有效范围内
        reconstructed = np.clip(reconstructed, 0, 255).astype(image.dtype)

        return reconstructed

    def _multi_scale_gradient(self, image: np.ndarray, scales: list) -> np.ndarray:
        """多尺度梯度增强"""
        enhanced = image.astype(np.float32)

        for scale in scales:
            # 不同尺度的高斯模糊
            sigma = scale * 2
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)

            # 计算梯度
            grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

            # 归一化梯度
            grad_magnitude = cv2.normalize(grad_magnitude, None, 0, 1, cv2.NORM_MINMAX)

            # 增强梯度区域
            enhancement = 1.0 + 0.3 * grad_magnitude
            enhanced = enhanced * enhancement

        # 确保值在有效范围内
        enhanced = np.clip(enhanced, 0, 255).astype(image.dtype)

        return enhanced

    def enhance_edges_selective(self, image: np.ndarray,
                                edge_threshold: float = 50,
                                enhancement_factor: float = 2.0) -> np.ndarray:
        """
        选择性边缘增强

        Args:
            image: 输入图像
            edge_threshold: 边缘检测阈值
            enhancement_factor: 增强因子

        Returns:
            边缘增强后的图像
        """
        # 边缘检测
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 使用Canny边缘检测
        edges = cv2.Canny(gray, edge_threshold, edge_threshold * 2)

        # 膨胀边缘掩码
        kernel = np.ones((3, 3), np.uint8)
        edge_mask = cv2.dilate(edges, kernel, iterations=1)

        # 将掩码转换为浮点型
        edge_mask_float = edge_mask.astype(np.float32) / 255.0

        # 应用非锐化掩模
        sharpened = self.apply_unsharp_mask(image, amount=enhancement_factor)

        # 混合原始图像和锐化图像（只在边缘区域）
        enhanced = image.astype(np.float32) * (1 - edge_mask_float) + \
                   sharpened.astype(np.float32) * edge_mask_float

        # 确保值在有效范围内
        enhanced = np.clip(enhanced, 0, 255).astype(image.dtype)

        return enhanced