import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Union


class ImageIO:
    """图像读写工具类"""

    def __init__(self, config):
        self.config = config

    def read_image(self, image_path: Union[str, Path],
                   mode: str = 'color') -> Optional[np.ndarray]:
        """读取图像"""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"图像文件不存在: {image_path}")

            if mode == 'color':
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            elif mode == 'grayscale':
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            elif mode == 'unchanged':
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            else:
                raise ValueError(f"不支持的读取模式: {mode}")

            if image is None:
                raise ValueError(f"无法读取图像，可能格式不支持: {image_path}")

            return image

        except Exception as e:
            print(f"读取图像时出错 {image_path}: {e}")
            return None

    def save_image(self, image: np.ndarray, save_path: Union[str, Path]):
        """保存图像"""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            success = cv2.imwrite(str(save_path), image)
            if not success:
                raise ValueError(f"保存图像失败: {save_path}")

            return True

        except Exception as e:
            print(f"保存图像时出错 {save_path}: {e}")
            return False

    def find_images(self, directory: Union[str, Path],
                    extensions: List[str] = None) -> List[Path]:
        """查找目录中的所有图像文件"""
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"目录不存在: {directory}")

        image_files = []
        for ext in extensions:
            image_files.extend(directory.glob(f'**/*{ext}'))
            image_files.extend(directory.glob(f'**/*{ext.upper()}'))

        # 去重和排序
        image_files = list(set(image_files))
        image_files.sort()

        return image_files