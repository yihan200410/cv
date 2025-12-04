"""
日志配置模块
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# 日志级别颜色映射
LOG_COLORS = {
    'DEBUG': '\033[36m',  # 青色
    'INFO': '\033[32m',  # 绿色
    'WARNING': '\033[33m',  # 黄色
    'ERROR': '\033[31m',  # 红色
    'CRITICAL': '\033[35m',  # 紫色
}

RESET_COLOR = '\033[0m'


class ColorFormatter(logging.Formatter):
    """彩色日志格式化器"""

    def format(self, record):
        # 添加颜色
        levelname = record.levelname
        if levelname in LOG_COLORS:
            record.levelname = f"{LOG_COLORS[levelname]}{levelname}{RESET_COLOR}"
            record.msg = f"{LOG_COLORS[levelname]}{record.msg}{RESET_COLOR}"

        return super().format(record)


class FileFormatter(logging.Formatter):
    """文件日志格式化器"""

    def format(self, record):
        # 移除颜色代码
        record.msg = self._remove_color_codes(str(record.msg))
        return super().format(record)

    def _remove_color_codes(self, text: str) -> str:
        """移除ANSI颜色代码"""
        import re
        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        return ansi_escape.sub('', text)


def setup_logger(name: str,
                 log_level: str = "INFO",
                 log_file: Optional[Path] = None,
                 console_output: bool = True) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志器名称
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径
        console_output: 是否输出到控制台

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)

    # 避免重复设置
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper()))

    # 清除现有的处理器
    logger.handlers.clear()

    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColorFormatter(
            '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        # 确保日志目录存在
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = FileFormatter(
            '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # 避免传播到根日志器
    logger.propagate = False

    return logger


def get_project_logger(name: str = None) -> logging.Logger:
    """
    获取项目标准日志记录器

    Args:
        name: 模块名称，如果为None则使用调用者模块名

    Returns:
        logging.Logger: 项目日志记录器
    """
    import inspect

    if name is None:
        # 获取调用者模块名
        frame = inspect.currentframe().f_back
        module = inspect.getmodule(frame)
        name = module.__name__ if module else "__main__"

    # 项目根目录
    project_root = Path(__file__).parent.parent.parent

    # 日志文件路径
    log_dir = project_root / "logs"
    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d')}.log"

    return setup_logger(
        name=name,
        log_level="INFO",
        log_file=log_file,
        console_output=True
    )


# 全局日志器
logger = get_project_logger(__name__)

if __name__ == "__main__":
    # 测试日志功能
    test_logger = get_project_logger("test_module")

    test_logger.debug("这是一个调试信息")
    test_logger.info("这是一个普通信息")
    test_logger.warning("这是一个警告信息")
    test_logger.error("这是一个错误信息")
    test_logger.critical("这是一个严重错误信息")