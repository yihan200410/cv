#!/usr/bin/env python
"""
汽车漆面缺陷图像增强系统主程序
"""
import cv2
import numpy as np
import argparse
import time
from pathlib import Path
import json

# 添加项目根目录到Python路径
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import config
from src.analysis.distortion_analyzer import DistortionAnalyzer
from src.analysis.quality_assessment import ImageQualityAssessment
from src.enhancement.algorithm_selector import EnhancementSelector
from src.utils.image_io import ImageIO
from src.utils.visualization import Visualization
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class CarPaintDefectEnhancer:
    """汽车漆面缺陷增强主类"""

    def __init__(self):
        self.config = config
        self.distortion_analyzer = DistortionAnalyzer(config)
        self.quality_assessor = ImageQualityAssessment()
        self.enhancement_selector = EnhancementSelector(config)
        self.image_io = ImageIO(config)
        self.visualizer = Visualization()

    def process_single_image(self, image_path: Path,
                             output_dir: Path = None,
                             strategy: str = "adaptive") -> dict:
        """处理单张图像"""
        logger.info(f"处理图像: {image_path}")

        # 1. 读取图像
        image = self.image_io.read_image(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 2. 分析失真
        logger.info("分析图像失真...")
        analysis_result = self.distortion_analyzer.analyze(image)

        print("\n" + "=" * 50)
        print("图像失真分析结果:")
        print("=" * 50)
        for dist_type, (severity, score) in analysis_result["distortion_types"].items():
            print(f"  {dist_type}: {severity} (得分: {score:.3f})")

        print("\n增强建议:")
        for rec in analysis_result["recommendations"]:
            print(f"  • {rec}")

        # 3. 原始图像质量评估
        logger.info("评估原始图像质量...")
        orig_quality = self.quality_assessor.assess(image)

        print(f"\n原始图像质量:")
        print(f"  BRISQUE分数: {orig_quality['brisque']:.2f} (越低越好)")
        print(f"  清晰度: {orig_quality['sharpness']:.2f}")
        print(f"  对比度: {orig_quality['contrast']:.3f}")
        print(f"  综合质量: {orig_quality['overall_quality']:.2f}/100")

        # 4. 选择并应用增强算法
        logger.info(f"应用增强算法 (策略: {strategy})...")
        start_time = time.time()

        enhancement_result = self.enhancement_selector.select_and_apply(
            image,
            analysis_result["distortion_types"],
            strategy
        )

        processing_time = time.time() - start_time

        # 5. 增强后质量评估
        logger.info("评估增强后图像质量...")
        enh_quality = self.quality_assessor.assess(enhancement_result.enhanced_image)

        # 6. 计算改进
        improvement = self.enhancement_selector.evaluate_enhancement(
            image,
            enhancement_result.enhanced_image,
            self.quality_assessor
        )

        print(f"\n增强结果:")
        print(f"  应用算法: {enhancement_result.algorithm_name}")
        print(f"  处理时间: {processing_time:.3f}s")
        print(f"  增强后BRISQUE: {enh_quality['brisque']:.2f}")
        print(f"  质量提升: {improvement['improvements']['overall_quality']['change']:.2f}分")

        # 7. 保存结果
        if output_dir:
            self._save_results(image_path, image, enhancement_result,
                               analysis_result, improvement, output_dir)

        # 8. 可视化
        self._visualize_results(image, enhancement_result.enhanced_image,
                                analysis_result, improvement)

        return {
            "original_image": image,
            "enhanced_image": enhancement_result.enhanced_image,
            "analysis_result": analysis_result,
            "quality_improvement": improvement,
            "processing_time": processing_time
        }

    def process_batch(self, input_dir: Path, output_dir: Path,
                      strategy: str = "adaptive") -> dict:
        """批量处理图像"""
        results = {}

        # 查找所有图像文件
        image_paths = self.image_io.find_images(input_dir)
        logger.info(f"找到 {len(image_paths)} 张图像")

        for i, img_path in enumerate(image_paths):
            try:
                logger.info(f"处理图像 {i + 1}/{len(image_paths)}: {img_path.name}")

                result = self.process_single_image(
                    img_path,
                    output_dir / img_path.stem,
                    strategy
                )

                results[img_path.name] = {
                    "quality_improvement": result["quality_improvement"]["improvements"]["overall_quality"]["change"],
                    "processing_time": result["processing_time"]
                }

            except Exception as e:
                logger.error(f"处理图像 {img_path} 时出错: {e}")
                continue

        # 生成批量处理报告
        self._generate_batch_report(results, output_dir)

        return results

    def _save_results(self, image_path: Path,
                      original: np.ndarray,
                      enhancement_result,
                      analysis_result: dict,
                      improvement: dict,
                      output_dir: Path):
        """保存处理结果"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建增强图片专门目录（在输出目录的同级）
        enhanced_only_dir = output_dir.parent / "enhanced_only"
        enhanced_only_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存增强后的图像到结果目录
        enhanced_path = output_dir / f"{image_path.stem}_enhanced{image_path.suffix}"
        cv2.imwrite(str(enhanced_path), enhancement_result.enhanced_image)

        # 2. 保存增强后的图像到专门目录（保持原文件名）
        enhanced_only_path = enhanced_only_dir / f"{image_path.name}"
        cv2.imwrite(str(enhanced_only_path), enhancement_result.enhanced_image)

        # 3. 保存分析结果
        analysis_path = output_dir / f"{image_path.stem}_analysis.json"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump({
                "distortion_types": analysis_result["distortion_types"],
                "metrics": {k: float(v) for k, v in analysis_result["metrics"].__dict__.items()},
                "applied_algorithms": enhancement_result.parameters.get("applied_algorithms", []),
                "quality_improvement": improvement["improvements"]
            }, f, indent=4, ensure_ascii=False)

        # 4. 保存对比图像
        comparison = self.visualizer.create_comparison(
            original,
            enhancement_result.enhanced_image,
            f"Original vs Enhanced\nQuality: {improvement['improvements']['overall_quality']['change']:.1f}+"
        )
        comparison_path = output_dir / f"{image_path.stem}_comparison.jpg"
        cv2.imwrite(str(comparison_path), comparison)

        logger.info(f"结果已保存到: {output_dir}")
        logger.info(f"增强图片已单独保存到: {enhanced_only_dir}")

    def _visualize_results(self, original: np.ndarray,
                           enhanced: np.ndarray,
                           analysis_result: dict,
                           improvement: dict):
        """静默处理，不显示可视化结果"""
        # 只保存结果，不显示
        try:
            # 1. 创建对比图并保存
            comparison = self.visualizer.create_comparison(
                original, enhanced,
                f"Quality Improvement: {improvement['improvements']['overall_quality']['change']:+.1f}"
            )

            # 2. 保存失真分析图（不显示）
            self.visualizer.plot_distortion_analysis(
                analysis_result,
                save_path=None,  # 不保存图表文件
                show=False  # 不显示
            )

            # 3. 保存质量改进图（不显示）
            self.visualizer.plot_quality_improvement(
                improvement,
                save_path=None,  # 不保存图表文件
                show=False  # 不显示
            )

            # 4. 保存直方图对比（不显示）
            self.visualizer.plot_histograms(
                original, enhanced,
                save_path=None,  # 不保存图表文件
                show=False  # 不显示
            )

            # 注意：display_image 方法会弹出窗口，我们不调用它
            # self.visualizer.display_image(comparison, "Original vs Enhanced")

        except Exception as e:
            logger.warning(f"可视化处理失败: {e}")

    def _generate_batch_report(self, results: dict, output_dir: Path):
        """生成批量处理报告"""
        report_path = output_dir / "batch_report.json"

        report = {
            "total_images": len(results),
            "successful": len([r for r in results.values() if r["quality_improvement"] > 0]),
            "failed": len([r for r in results.values() if r["quality_improvement"] <= 0]),
            "average_improvement": np.mean([r["quality_improvement"] for r in results.values()]),
            "average_processing_time": np.mean([r["processing_time"] for r in results.values()]),
            "detailed_results": results
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        logger.info(f"批量处理报告已保存到: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='汽车漆面缺陷图像增强系统')
    parser.add_argument('--input', type=str, required=True,
                        help='输入图像路径或目录')
    parser.add_argument('--output', type=str, default='./results',
                        help='输出目录')
    parser.add_argument('--strategy', type=str, default='adaptive',
                        choices=['conservative', 'adaptive', 'aggressive'],
                        help='增强策略')
    parser.add_argument('--batch', action='store_true',
                        help='批量处理模式')

    args = parser.parse_args()

    # 创建增强器实例
    enhancer = CarPaintDefectEnhancer()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.batch and input_path.is_dir():
            # 批量处理模式
            logger.info(f"开始批量处理: {input_path}")
            results = enhancer.process_batch(input_path, output_dir, args.strategy)
            logger.info(f"批量处理完成，处理了 {len(results)} 张图像")
        else:
            # 单图像处理模式
            if not input_path.is_file():
                raise FileNotFoundError(f"文件不存在: {input_path}")

            result = enhancer.process_single_image(input_path, output_dir, args.strategy)
            logger.info(f"单图像处理完成")

    except Exception as e:
        logger.error(f"处理失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()