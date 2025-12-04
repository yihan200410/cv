from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .contrast_enhancer import ContrastEnhancer
from .edge_enhancer import EdgeEnhancer
from .illumination_corrector import IlluminationCorrector
from ..analysis.distortion_analyzer import DistortionAnalyzer
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class EnhancementResult:
    """增强结果"""
    enhanced_image: np.ndarray
    algorithm_name: str
    parameters: Dict
    quality_improvement: float
    processing_time: float


class EnhancementSelector:
    """增强算法选择器"""

    def __init__(self, config):
        self.config = config
        self.distortion_analyzer = DistortionAnalyzer(config)
        self.contrast_enhancer = ContrastEnhancer(config)
        self.edge_enhancer = EdgeEnhancer(config)
        self.illumination_corrector = IlluminationCorrector(config)

        # 算法映射表
        self.algorithm_registry = {
            "low_contrast": [
                ("clahe", self.contrast_enhancer.apply_clahe),
                ("histogram_equalization", self.contrast_enhancer.apply_histogram_equalization),
                ("retinex", self.contrast_enhancer.apply_retinex),
                ("contrast_stretch", self.contrast_enhancer.apply_contrast_stretch)
            ],
            "blur": [
                ("unsharp_mask", self.edge_enhancer.apply_unsharp_mask),
                ("laplacian_sharpen", self.edge_enhancer.apply_laplacian_sharpen),
                ("guided_filter", self.edge_enhancer.apply_guided_filter)
            ],
            "noise": [
                ("bilateral_filter", self.edge_enhancer.apply_bilateral_filter),
                ("non_local_means", self.edge_enhancer.apply_non_local_means),
                ("wavelet_denoise", self.edge_enhancer.apply_wavelet_denoise)
            ],
            "low_light": [
                ("gamma_correction", self.illumination_corrector.apply_gamma_correction),
                ("adaptive_histogram", self.contrast_enhancer.apply_adaptive_histogram),
                ("brightness_adjustment", self.illumination_corrector.adjust_brightness)
            ],
            "shadow": [
                ("homomorphic_filtering", self.illumination_corrector.apply_homomorphic_filtering),
                ("multi_scale_retinex", self.contrast_enhancer.apply_multi_scale_retinex),
                ("shadow_removal", self.illumination_corrector.remove_shadows)
            ],
            "over_exposure": [
                ("exposure_correction", self.illumination_corrector.correct_exposure),
                ("tone_mapping", self.illumination_corrector.apply_tone_mapping)
            ]
        }

    def select_and_apply(self, image: np.ndarray,
                         distortion_types: Dict = None,
                         strategy: str = "adaptive") -> EnhancementResult:
        """
        根据失真类型选择并应用增强算法

        Args:
            image: 输入图像
            distortion_types: 失真类型字典（如果为None则自动分析）
            strategy: 选择策略 ("adaptive", "conservative", "aggressive")

        Returns:
            EnhancementResult: 增强结果
        """
        import time

        # 1. 分析失真类型
        if distortion_types is None:
            analysis_result = self.distortion_analyzer.analyze(image)
            distortion_types = analysis_result["distortion_types"]
            logger.info(f"检测到的失真类型: {distortion_types.keys()}")

        # 2. 根据策略选择算法
        selected_algorithms = self._select_algorithms(distortion_types, strategy)

        # 3. 按顺序应用算法
        enhanced = image.copy()
        applied_algorithms = []
        total_time = 0

        for algo_name, algo_func, params in selected_algorithms:
            start_time = time.time()

            try:
                enhanced = algo_func(enhanced, **params)
                processing_time = time.time() - start_time
                total_time += processing_time

                applied_algorithms.append({
                    "name": algo_name,
                    "time": processing_time,
                    "params": params
                })

                logger.info(f"应用算法: {algo_name}, 耗时: {processing_time:.3f}s")

            except Exception as e:
                logger.warning(f"算法 {algo_name} 应用失败: {e}")
                continue

        # 4. 包装结果
        return EnhancementResult(
            enhanced_image=enhanced,
            algorithm_name="->".join([a["name"] for a in applied_algorithms]),
            parameters={"applied_algorithms": applied_algorithms},
            quality_improvement=0,  # 由外部评估
            processing_time=total_time
        )

    def _select_algorithms(self, distortion_types: Dict,
                           strategy: str) -> List[tuple]:
        """根据失真类型和策略选择算法"""
        selected = []

        # 策略权重
        strategy_params = {
            "conservative": {"max_algorithms": 2, "prefer_mild": True},
            "adaptive": {"max_algorithms": 3, "prefer_mild": False},
            "aggressive": {"max_algorithms": 4, "prefer_mild": False}
        }

        params = strategy_params.get(strategy, strategy_params["adaptive"])
        max_algorithms = params["max_algorithms"]

        # 根据失真严重度排序
        sorted_distortions = sorted(
            distortion_types.items(),
            key=lambda x: (1 if x[1][0] == "severe" else 0, x[1][1]),
            reverse=True
        )

        # 为每种失真类型选择算法
        for dist_type, (severity, score) in sorted_distortions:
            if dist_type in self.algorithm_registry:
                algorithms = self.algorithm_registry[dist_type]

                # 根据严重度选择算法
                if severity == "severe":
                    # 严重失真：选择更强的算法
                    selected_algo = algorithms[0]  # 第一个通常是最强的
                elif params["prefer_mild"]:
                    # 保守策略：选择较温和的算法
                    selected_algo = algorithms[-1] if len(algorithms) > 1 else algorithms[0]
                else:
                    # 自适应：选择中等强度的算法
                    selected_algo = algorithms[len(algorithms) // 2]

                # 转换为标准格式 (算法名, 函数, 参数)
                algo_name, algo_func = selected_algo
                algo_params = self.config.ENHANCEMENT_PARAMS.get(algo_name, {})

                # 根据严重度调整参数
                if severity == "severe":
                    algo_params = self._adjust_parameters(algo_name, algo_params, "strengthen")
                elif severity == "mild":
                    algo_params = self._adjust_parameters(algo_name, algo_params, "weaken")

                selected.append((algo_name, algo_func, algo_params))

                if len(selected) >= max_algorithms:
                    break

        return selected

    def _adjust_parameters(self, algorithm: str, params: Dict,
                           adjustment: str) -> Dict:
        """根据调整类型调整算法参数"""
        adjusted = params.copy()

        if adjustment == "strengthen":
            if algorithm == "clahe":
                adjusted["clip_limit"] = min(params.get("clip_limit", 2.0) * 1.5, 4.0)
            elif algorithm == "unsharp_mask":
                adjusted["amount"] = params.get("amount", 1.5) * 1.3
            elif algorithm == "gamma_correction":
                adjusted["gamma"] = max(params.get("gamma", 1.2) * 0.8, 0.5)

        elif adjustment == "weaken":
            if algorithm == "clahe":
                adjusted["clip_limit"] = max(params.get("clip_limit", 2.0) * 0.7, 0.5)
            elif algorithm == "unsharp_mask":
                adjusted["amount"] = params.get("amount", 1.5) * 0.7
            elif algorithm == "gamma_correction":
                adjusted["gamma"] = min(params.get("gamma", 1.2) * 1.2, 2.5)

        return adjusted

    def evaluate_enhancement(self, original: np.ndarray,
                             enhanced: np.ndarray,
                             quality_assessor) -> Dict:
        """评估增强效果"""
        # 计算质量改进
        orig_quality = quality_assessor.assess(original)
        enh_quality = quality_assessor.assess(enhanced)

        improvement = {
            "original_quality": orig_quality,
            "enhanced_quality": enh_quality,
            "improvements": {}
        }

        # 计算各项指标的改进
        for metric in ["brisque", "sharpness", "contrast", "entropy", "overall_quality"]:
            if metric in orig_quality and metric in enh_quality:
                orig_val = orig_quality[metric]
                enh_val = enh_quality[metric]

                # 对于BRISQUE，分数越低越好
                if metric == "brisque":
                    change = orig_val - enh_val  # 负值表示改善
                else:
                    change = enh_val - orig_val  # 正值表示改善

                improvement["improvements"][metric] = {
                    "original": orig_val,
                    "enhanced": enh_val,
                    "change": change,
                    "percent_change": (change / (orig_val + 1e-10)) * 100
                }

        return improvement