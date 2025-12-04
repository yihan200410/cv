# src/segmentation/utils/yolo_to_mask.py
import cv2
import numpy as np
from pathlib import Path


class YOLOToMaskConverter:
    """将YOLO多边形标注转换为分割掩码"""

    @staticmethod
    def yolo_to_polygon(yolo_line, img_width, img_height):
        """
        将YOLO格式转换为像素坐标多边形
        """
        try:
            data = list(map(float, yolo_line.strip().split()))

            if len(data) < 5:
                return None

            polygon_norm = data[5:]

            if len(polygon_norm) < 6:
                return None

            polygon_points = []
            for i in range(0, len(polygon_norm), 2):
                if i + 1 >= len(polygon_norm):
                    break

                x_norm = polygon_norm[i]
                y_norm = polygon_norm[i + 1]
                x_pixel = int(x_norm * img_width)
                y_pixel = int(y_norm * img_height)
                polygon_points.append((x_pixel, y_pixel))

            if len(polygon_points) < 3:
                return None

            return np.array(polygon_points, dtype=np.int32)

        except Exception as e:
            print(f"解析YOLO标注时出错: {e}")
            return None

    @staticmethod
    def create_mask_from_yolo(label_path, image_shape):
        """
        从YOLO标签文件创建掩码
        """
        if isinstance(image_shape, tuple) and len(image_shape) >= 2:
            if len(image_shape) == 3:
                height, width = image_shape[:2]
            else:
                height, width = image_shape
        else:
            raise ValueError(f"无效的图像形状: {image_shape}")

        mask = np.zeros((height, width), dtype=np.uint8)

        try:
            label_path = Path(label_path)
            if not label_path.exists():
                return mask

            with open(label_path, 'r') as f:
                lines = f.readlines()

            if not lines:
                return mask

            for line in lines:
                polygon = YOLOToMaskConverter.yolo_to_polygon(line, width, height)
                if polygon is not None and len(polygon) >= 3:
                    cv2.fillPoly(mask, [polygon], 255)

        except Exception as e:
            print(f"创建掩码时出错 {label_path}: {e}")

        return mask