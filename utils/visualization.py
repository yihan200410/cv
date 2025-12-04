"""
可视化工具模块
用于显示和分析结果
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .logger import setup_logger

logger = setup_logger(__name__)


class Visualization:
    """可视化类"""

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        初始化可视化工具

        Args:
            figsize: 图形大小
            dpi: 分辨率
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style = 'seaborn-v0_8-darkgrid'
        plt.style.use(self.style)

        # 完全禁用中文字体，确保只使用英文字体
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
        matplotlib.rcParams['axes.unicode_minus'] = False
        # 确保不使用任何中文字体
        matplotlib.rcParams['font.family'] = 'sans-serif'

        # 设置图表标题为英文
        self.chart_titles = {
            'distortion_analysis': 'Image Distortion Analysis',
            'distortion_types': 'Distortion Types Distribution',
            'distortion_metrics': 'Distortion Metrics',
            'distortion_values': 'Distortion Metrics Values',
            'recommendations': 'Enhancement Recommendations',
            'quality_improvement': 'Quality Improvement Analysis',
            'metrics_comparison': 'Key Metrics Comparison',
            'improvement_percentage': 'Improvement Percentage',
            'overall_quality': 'Overall Quality Improvement',
            'correlation_analysis': 'Correlation Analysis',
            'histogram_comparison': 'Histogram Comparison',
            'original_histogram': 'Original Image Histogram',
            'enhanced_histogram': 'Enhanced Image Histogram',
            'cdf': 'Cumulative Distribution Function',
            'histogram_overlay': 'Histogram Overlay',
            'image_display': 'Image Display'
        }

    def create_comparison(self, original: np.ndarray,
                          enhanced: np.ndarray,
                          title: str = "Original vs Enhanced",
                          save_path: Optional[Path] = None) -> np.ndarray:
        """
        创建原始图像与增强图像的对比图

        Args:
            original: 原始图像
            enhanced: 增强后的图像
            title: 对比图标题
            save_path: 保存路径

        Returns:
            对比图图像
        """
        # 确保图像大小一致
        if original.shape != enhanced.shape:
            # 调整增强图像大小
            enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))

        # 创建对比图
        h, w = original.shape[:2]
        comparison = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)

        # 放置原始图像
        if len(original.shape) == 2:
            original_color = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            original_color = original.copy()

        # 放置增强图像
        if len(enhanced.shape) == 2:
            enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            enhanced_color = enhanced.copy()

        # 拼接图像
        comparison[:h, :w] = original_color
        comparison[:h, w + 20: w * 2 + 20] = enhanced_color

        # 添加分隔线
        comparison[:, w:w + 20] = 100

        # 添加标签（英文）
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Enhanced", (w + 30, 30), font, 1, (255, 255, 255), 2)

        # 添加标题
        title_y = h - 10 if h > 100 else h - 5
        cv2.putText(comparison, title, (w // 2 - 100, title_y), font, 0.8, (255, 255, 255), 2)

        # 保存图像
        if save_path:
            cv2.imwrite(str(save_path), comparison)
            logger.info(f"Comparison image saved: {save_path}")

        return comparison

    def plot_distortion_analysis(self, analysis_result: Dict,
                                 save_path: Optional[Path] = None,
                                 show: bool = False):
        """
        绘制失真分析结果
        """
        metrics = analysis_result.get("metrics", {})
        distortion_types = analysis_result.get("distortion_types", {})

        # 创建图形，只用3个子图，去掉雷达图
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle('Image Distortion Analysis', fontsize=16, fontweight='bold')

        # 1. 失真类型饼图
        if distortion_types:
            ax1 = axes[0, 0]
            labels = list(distortion_types.keys())
            scores = [dist[1] for dist in distortion_types.values()]

            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            wedges, texts, autotexts = ax1.pie(
                scores, labels=labels, autopct='%1.1f%%',
                startangle=90, colors=colors
            )
            for text in texts:
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            ax1.set_title('Distortion Types Distribution')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Distortions Detected',
                            horizontalalignment='center',
                            verticalalignment='center')
            axes[0, 0].set_title('Distortion Types')

        # 2. 指标数值表格
        ax2 = axes[0, 1]
        ax2.axis('off')
        ax2.set_title('Distortion Metrics')

        if hasattr(metrics, '__dict__'):
            metric_dict = metrics.__dict__
        else:
            metric_dict = metrics

        if metric_dict:
            # 创建文本表格
            table_data = []
            for key, value in metric_dict.items():
                if isinstance(value, (int, float)):
                    table_data.append([key, f'{value:.3f}'])

            if table_data:
                table = ax2.table(cellText=table_data,
                                  colLabels=['Metric', 'Value'],
                                  cellLoc='center',
                                  loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)
        else:
            ax2.text(0.5, 0.5, 'No metrics data',
                     horizontalalignment='center',
                     verticalalignment='center')

        # 3. 指标柱状图
        ax3 = axes[1, 0]
        if hasattr(metrics, '__dict__'):
            metric_dict = metrics.__dict__
        else:
            metric_dict = metrics

        if metric_dict:
            numeric_metrics = {}
            for key, value in metric_dict.items():
                if isinstance(value, (int, float)):
                    numeric_metrics[key] = float(value)

            if numeric_metrics:
                keys = list(numeric_metrics.keys())
                values = list(numeric_metrics.values())
                bars = ax3.bar(range(len(keys)), values,
                               color=plt.cm.viridis(np.linspace(0, 1, len(keys))))
                ax3.set_xticks(range(len(keys)))
                ax3.set_xticklabels(keys, rotation=45, ha='right')
                ax3.set_title('Distortion Metrics Values')
                ax3.set_ylabel('Value')
                ax3.grid(axis='y', alpha=0.3)

                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width() / 2., height,
                             f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            else:
                ax3.text(0.5, 0.5, 'No numeric metrics',
                         horizontalalignment='center',
                         verticalalignment='center')
        else:
            ax3.text(0.5, 0.5, 'No metrics data',
                     horizontalalignment='center',
                     verticalalignment='center')

        # 4. 增强建议 - 修复：确保所有文本都是英文
        ax4 = axes[1, 1]
        recommendations = analysis_result.get("recommendations", [])

        if recommendations:
            ax4.axis('off')
            ax4.set_title('Enhancement Recommendations')
            text_y = 0.95
            for i, rec in enumerate(recommendations):
                # 确保推荐文本是英文，如果没有英文，使用占位符
                if isinstance(rec, str):
                    # 如果有中文，替换为英文或移除
                    rec_text = rec
                else:
                    rec_text = str(rec)
                ax4.text(0.05, text_y, f'• {rec_text}',
                         transform=ax4.transAxes,
                         fontsize=10,
                         verticalalignment='top')
                text_y -= 0.08
        else:
            ax4.text(0.5, 0.5, 'No Recommendations',
                     horizontalalignment='center',
                     verticalalignment='center')
            ax4.set_title('Recommendations')

        plt.tight_layout()

        if save_path:
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
            logger.info(f"Distortion analysis plot saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_quality_improvement(self, improvement: Dict,
                                 save_path: Optional[Path] = None,
                                 show: bool = True):
        """
        绘制质量改进对比图

        Args:
            improvement: 质量改进数据
            save_path: 保存路径
            show: 是否显示图像
        """
        improvements = improvement.get("improvements", {})

        if not improvements:
            logger.warning("No quality improvement data to display")
            return

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle('Quality Improvement Analysis', fontsize=16, fontweight='bold')

        # 1. 关键指标对比图
        ax1 = axes[0, 0]

        key_metrics = ['brisque', 'sharpness', 'contrast', 'overall_quality']
        metric_names = ['BRISQUE', 'Sharpness', 'Contrast', 'Overall Quality']

        original_vals = []
        enhanced_vals = []

        for metric in key_metrics:
            if metric in improvements:
                metric_data = improvements[metric]
                original_vals.append(metric_data.get("original", 0))
                enhanced_vals.append(metric_data.get("enhanced", 0))
            else:
                original_vals.append(0)
                enhanced_vals.append(0)

        x = np.arange(len(metric_names))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, original_vals, width, label='Original', color='lightblue')
        bars2 = ax1.bar(x + width / 2, enhanced_vals, width, label='Enhanced', color='lightgreen')

        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Key Metrics Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metric_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        # 2. 改进百分比图
        ax2 = axes[0, 1]

        improvements_pct = []
        for metric in key_metrics:
            if metric in improvements:
                pct_change = improvements[metric].get("percent_change", 0)
                # 对于BRISQUE，负的百分比变化表示改进
                if metric == 'brisque':
                    pct_change = -pct_change
                improvements_pct.append(pct_change)
            else:
                improvements_pct.append(0)

        colors = ['green' if val > 0 else 'red' for val in improvements_pct]

        # 修复：先设置刻度位置，再设置标签
        x_positions = np.arange(len(metric_names))
        bars = ax2.bar(x_positions, improvements_pct, color=colors)
        ax2.set_xticks(x_positions)  # 先设置刻度位置
        ax2.set_xticklabels(metric_names, rotation=45, ha='right')  # 再设置标签
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Improvement Percentage')
        ax2.grid(axis='y', alpha=0.3)

        # 3. 综合质量趋势图
        ax3 = axes[1, 0]

        if 'overall_quality' in improvements:
            orig_quality = improvements['overall_quality'].get("original", 0)
            enh_quality = improvements['overall_quality'].get("enhanced", 0)
            change = improvements['overall_quality'].get("change", 0)

            stages = ['Original', 'Enhanced']
            values = [orig_quality, enh_quality]

            ax3.plot(stages, values, 'o-', linewidth=2, markersize=8, color='orange')
            ax3.fill_between(stages, values, alpha=0.2, color='orange')

            # 添加箭头显示改进
            ax3.annotate('', xy=(1, enh_quality), xytext=(0, orig_quality),
                         arrowprops=dict(arrowstyle='->', color='red', lw=2))

            ax3.text(0.5, (orig_quality + enh_quality) / 2,
                     f'Improvement: {change:+.1f}',
                     ha='center', va='center', backgroundcolor='white')

            ax3.set_ylim(0, 100)
            ax3.set_xlabel('Processing Stage')
            ax3.set_ylabel('Quality Score')
            ax3.set_title('Overall Quality Improvement')
            ax3.grid(True, alpha=0.3)

            # 添加数值标签
            for i, (stage, value) in enumerate(zip(stages, values)):
                ax3.text(i, value + 2, f'{value:.1f}', ha='center', fontsize=10, fontweight='bold')

        # 4. 指标相关性热图
        ax4 = axes[1, 1]

        # 收集所有指标数据
        all_metrics = {}
        for metric_name, metric_data in improvements.items():
            if isinstance(metric_data, dict):
                orig_val = metric_data.get("original", 0)
                enh_val = metric_data.get("enhanced", 0)
                # 只添加有效数值
                if not (np.isnan(orig_val) or np.isnan(enh_val)):
                    all_metrics[f"{metric_name}_orig"] = orig_val
                    all_metrics[f"{metric_name}_enh"] = enh_val

        if len(all_metrics) > 1:
            try:
                import pandas as pd
                df = pd.DataFrame([all_metrics])

                # 检查是否有足够的有效数据
                if df.notna().sum().sum() >= 4:  # 至少4个有效值
                    corr_matrix = df.corr()

                    # 检查相关性矩阵是否有效
                    if not corr_matrix.isna().all().all():
                        # 绘制热图
                        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                                    center=0, square=True, linewidths=1,
                                    cbar_kws={"shrink": 0.8}, ax=ax4)
                        ax4.set_title('Metrics Correlation Heatmap')
                    else:
                        ax4.text(0.5, 0.5, 'No valid correlation data',
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 transform=ax4.transAxes)
                        ax4.set_title('Correlation Analysis')
                else:
                    ax4.text(0.5, 0.5, 'Insufficient data for correlation',
                             horizontalalignment='center',
                             verticalalignment='center',
                             transform=ax4.transAxes)
                    ax4.set_title('Correlation Analysis')
            except Exception as e:
                ax4.text(0.5, 0.5, f'Error in correlation analysis',
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=ax4.transAxes)
                ax4.set_title('Correlation Analysis')
        else:
            ax4.text(0.5, 0.5, 'Insufficient data for correlation analysis',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax4.transAxes)
            ax4.set_title('Correlation Analysis')

        plt.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
            logger.info(f"Quality improvement plot saved: {save_path}")

        # 显示图形
        if show:
            plt.show()
        else:
            plt.close()

    def display_image(self, image: np.ndarray,
                      title: str = "Image Display",
                      cmap: str = None,
                      save_path: Optional[Path] = None,
                      show: bool = True):
        """
        显示图像

        Args:
            image: 要显示的图像
            title: 图像标题
            cmap: 颜色映射（用于灰度图像）
            save_path: 保存路径
            show: 是否显示图像
        """
        plt.figure(figsize=(10, 8))

        if len(image.shape) == 2:
            # 灰度图像
            display_cmap = cmap or 'gray'
            plt.imshow(image, cmap=display_cmap)
        elif len(image.shape) == 3:
            # 彩色图像
            if image.shape[2] == 3:
                # BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.imshow(image_rgb)
            else:
                plt.imshow(image)
        else:
            logger.error(f"Unsupported image dimension: {image.shape}")
            return

        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)

        # 保存图像
        if save_path:
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
            logger.info(f"Image saved: {save_path}")

        # 显示图像
        if show:
            plt.show()
        else:
            plt.close()

    def plot_histograms(self, original: np.ndarray,
                        enhanced: np.ndarray,
                        save_path: Optional[Path] = None,
                        show: bool = True):
        """
        绘制原始图像和增强图像的直方图对比

        Args:
            original: 原始图像
            enhanced: 增强后的图像
            save_path: 保存路径
            show: 是否显示图像
        """
        # 转换为灰度图像用于直方图
        if len(original.shape) == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original

        if len(enhanced.shape) == 3:
            enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        else:
            enhanced_gray = enhanced

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle('Histogram Comparison', fontsize=16, fontweight='bold')

        # 1. 原始图像直方图
        ax1 = axes[0, 0]
        ax1.hist(original_gray.ravel(), bins=256, range=(0, 256),
                 color='blue', alpha=0.7, edgecolor='black')
        ax1.set_title('Original Image Histogram')
        ax1.set_xlabel('Pixel Intensity')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)

        # 添加统计信息
        orig_mean = np.mean(original_gray)
        orig_std = np.std(original_gray)
        ax1.axvline(orig_mean, color='red', linestyle='--', label=f'Mean: {orig_mean:.1f}')
        ax1.legend()

        # 2. 增强图像直方图
        ax2 = axes[0, 1]
        ax2.hist(enhanced_gray.ravel(), bins=256, range=(0, 256),
                 color='green', alpha=0.7, edgecolor='black')
        ax2.set_title('Enhanced Image Histogram')
        ax2.set_xlabel('Pixel Intensity')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)

        # 添加统计信息
        enh_mean = np.mean(enhanced_gray)
        enh_std = np.std(enhanced_gray)
        ax2.axvline(enh_mean, color='red', linestyle='--', label=f'Mean: {enh_mean:.1f}')
        ax2.legend()

        # 3. 累积分布函数对比
        ax3 = axes[1, 0]

        # 计算CDF
        hist_orig, bins = np.histogram(original_gray.ravel(), 256, [0, 256])
        hist_enh, _ = np.histogram(enhanced_gray.ravel(), 256, [0, 256])

        cdf_orig = hist_orig.cumsum()
        cdf_enh = hist_enh.cumsum()

        cdf_orig_normalized = cdf_orig / cdf_orig.max()
        cdf_enh_normalized = cdf_enh / cdf_enh.max()

        ax3.plot(bins[:-1], cdf_orig_normalized, 'b-', label='Original', linewidth=2)
        ax3.plot(bins[:-1], cdf_enh_normalized, 'g-', label='Enhanced', linewidth=2)
        ax3.set_title('Cumulative Distribution Function')
        ax3.set_xlabel('Pixel Intensity')
        ax3.set_ylabel('CDF')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 直方图重叠对比
        ax4 = axes[1, 1]

        ax4.hist(original_gray.ravel(), bins=256, range=(0, 256),
                 color='blue', alpha=0.5, edgecolor='black', label='Original')
        ax4.hist(enhanced_gray.ravel(), bins=256, range=(0, 256),
                 color='green', alpha=0.5, edgecolor='black', label='Enhanced')
        ax4.set_title('Histogram Overlay')
        ax4.set_xlabel('Pixel Intensity')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
            logger.info(f"Histogram comparison plot saved: {save_path}")

        # 显示图形
        if show:
            plt.show()
        else:
            plt.close()

    def create_enhancement_report(self, original: np.ndarray,
                                  enhanced: np.ndarray,
                                  analysis_result: Dict,
                                  quality_improvement: Dict,
                                  save_dir: Path,
                                  image_name: str = "enhancement_report"):
        """
        创建完整的增强报告

        Args:
            original: 原始图像
            enhanced: 增强后的图像
            analysis_result: 失真分析结果
            quality_improvement: 质量改进数据
            save_dir: 保存目录
            image_name: 图像名称

        Returns:
            报告文件路径列表
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report_files = []

        # 1. 保存对比图像
        comparison_path = save_dir / f"{image_name}_comparison_{timestamp}.jpg"
        comparison = self.create_comparison(
            original, enhanced,
            title=f"Enhancement Report - {timestamp}",
            save_path=comparison_path
        )
        report_files.append(comparison_path)

        # 2. 保存失真分析图
        distortion_path = save_dir / f"{image_name}_distortion_analysis_{timestamp}.png"
        self.plot_distortion_analysis(
            analysis_result,
            save_path=distortion_path,
            show=False
        )
        report_files.append(distortion_path)

        # 3. 保存质量改进图
        quality_path = save_dir / f"{image_name}_quality_improvement_{timestamp}.png"
        self.plot_quality_improvement(
            quality_improvement,
            save_path=quality_path,
            show=False
        )
        report_files.append(quality_path)

        # 4. 保存直方图对比
        histogram_path = save_dir / f"{image_name}_histograms_{timestamp}.png"
        self.plot_histograms(
            original, enhanced,
            save_path=histogram_path,
            show=False
        )
        report_files.append(histogram_path)

        # 5. 保存增强后的图像
        enhanced_path = save_dir / f"{image_name}_enhanced_{timestamp}.jpg"
        cv2.imwrite(str(enhanced_path), enhanced)
        report_files.append(enhanced_path)

        logger.info(f"Enhancement report generated, containing {len(report_files)} files")

        return report_files

    @staticmethod
    def create_video_from_images(images: List[np.ndarray],
                                 output_path: Path,
                                 fps: int = 10):
        """
        从图像列表创建视频

        Args:
            images: 图像列表
            output_path: 输出视频路径
            fps: 帧率
        """
        if not images:
            logger.error("Image list is empty")
            return

        # 获取图像尺寸
        height, width = images[0].shape[:2]

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (width, height)
        )

        # 写入图像帧
        for i, image in enumerate(images):
            if len(image.shape) == 2:
                # 灰度图像转换为彩色
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            video_writer.write(image)

            # 添加进度显示
            if i % 10 == 0:
                logger.info(f"Processing frame {i + 1}/{len(images)}")

        # 释放视频写入器
        video_writer.release()
        logger.info(f"Video saved: {output_path}")


# 使用示例
if __name__ == "__main__":
    # 创建测试图像
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_enhanced = np.clip(test_image.astype(float) * 1.5, 0, 255).astype(np.uint8)

    # 创建可视化实例
    visualizer = Visualization()

    # 显示图像
    visualizer.display_image(test_image, title="Test Image")

    # 创建对比图
    comparison = visualizer.create_comparison(test_image, test_enhanced)

    # 创建测试分析结果
    test_analysis = {
        "metrics": {
            "blur_score": 0.2,
            "noise_score": 0.1,
            "contrast_score": 45.6,
            "brightness_mean": 128.3,
            "shadow_coverage": 0.25
        },
        "distortion_types": {
            "blur": ("moderate", 0.2),
            "low_contrast": ("mild", 45.6)
        },
        "recommendations": [
            "Apply sharpening algorithm",
            "Apply contrast enhancement"
        ]
    }

    # 绘制失真分析
    visualizer.plot_distortion_analysis(test_analysis)

    # 测试质量改进图
    test_improvement = {
        "improvements": {
            "brisque": {"original": 65.3, "enhanced": 42.1, "change": -23.2, "percent_change": -35.5},
            "sharpness": {"original": 15.7, "enhanced": 28.4, "change": 12.7, "percent_change": 80.9},
            "overall_quality": {"original": 62.5, "enhanced": 78.3, "change": 15.8, "percent_change": 25.3}
        }
    }

    visualizer.plot_quality_improvement(test_improvement)

    # 绘制直方图对比
    visualizer.plot_histograms(test_image, test_enhanced)