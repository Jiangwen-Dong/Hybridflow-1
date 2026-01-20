import logging
import os

# 全局变量，用于保存日志记录器实例
logger = None

def setup_logger(log_dir, log_filename="run_details.log"):
    """
    配置一个基于文件的日志记录器。

    参数:
        log_dir: 日志文件存放的目录。
        log_filename: 日志文件的名称。
    """
    global logger
    log_path = os.path.join(log_dir, log_filename)

    # 获取名为 "LLM_Logger" 的日志记录器实例
    logger = logging.getLogger("LLM_Logger")
    logger.setLevel(logging.INFO)

    # 如果已有处理器，先清空，避免重复记录
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建文件处理器
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # === 修改部分：更新格式化器以匹配您的要求 ===
    # 新格式: [级别] 月-日 时:分:秒 文件名:行号] 日志信息
    log_format = '[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d] %(message)s'
    date_format = '%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, datefmt=date_format)
    # ============================================
    
    file_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)

    return logger

def get_logger():
    """
    获取已配置的日志记录器实例。
    """
    global logger
    if logger is None:
        # 如果 setup_logger 未被调用，则提供一个备用基础配置
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("LLM_Logger")
    return logger

def log_separator():
    """
    记录一个分隔符行，用于区分不同的日志内容。
    注意：此函数现在会记录一个没有附加格式的简单分隔符。
    """
    # 创建一个临时的、没有复杂格式的处理器来打印分隔符
    temp_logger = logging.getLogger("LLM_Logger")
    
    # 检查是否有处理器，避免无处理器时报错
    if not temp_logger.handlers:
        return

    # 暂时移除格式，打印纯文本，然后再加回去
    original_formatter = temp_logger.handlers[0].formatter
    temp_logger.handlers[0].setFormatter(logging.Formatter('%(message)s'))
    temp_logger.handlers[0].setFormatter(original_formatter)