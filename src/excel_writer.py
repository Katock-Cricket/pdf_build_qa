#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import logging
from datetime import datetime
import json
import re
import threading
import time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExcelWriter:
    """增强版Excel文件写入类，支持多层次问答对和元数据"""

    def __init__(self, output_dir="output"):
        """
        初始化Excel写入器

        Args:
            output_dir (str): 输出目录路径
        """
        # 转换为绝对路径，确保多线程环境下路径一致
        self.output_dir = os.path.abspath(output_dir)

        # 文件写入锁，防止并发写入冲突
        self._file_lock = threading.Lock()

        # 文件计数器，用于生成唯一文件名
        self._file_counter = 0
        self._counter_lock = threading.Lock()

        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"创建输出目录: {self.output_dir}")

        logger.info(f"增强版Excel写入器初始化，输出目录: {self.output_dir}")

    def _sanitize_filename(self, filename):
        """
        清理文件名，去除非法字符

        Args:
            filename (str): 原始文件名

        Returns:
            str: 合法的文件名
        """
        # 去除扩展名
        name_without_ext = os.path.splitext(filename)[0]

        # 替换Windows和Unix文件系统中的非法字符
        # Windows: \ / : * ? " < > |
        # 额外考虑空格和特殊字符
        illegal_chars = r'[<>:"/\\|?*\x00-\x1f]'
        sanitized = re.sub(illegal_chars, '_', name_without_ext)

        # 去除首尾空格和点号
        sanitized = sanitized.strip('. ')

        # 如果清理后为空，使用默认名称
        if not sanitized:
            sanitized = 'unnamed'

        # 限制文件名长度(避免路径过长问题)
        max_length = 200
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized

    def _get_unique_counter(self):
        """
        获取唯一的文件计数器（线程安全）

        Returns:
            int: 唯一的计数器值
        """
        with self._counter_lock:
            self._file_counter += 1
            return self._file_counter

    def save_single_pdf_qa(self, qa_pairs, source, metadata, mode):
        """
        保存单个PDF的问答对到JSON文件（线程安全）

        Args:
            qa_pairs (list): 问答对列表
            source (str): 源文件名
            metadata (dict): PDF元数据
            mode (str): 模式: normal-正常模式生成大量较短的且相对常见基础的知识问答对, pro-专业模式生成少量但长篇的深入研讨问答对
        Returns:
            str: 保存的JSON文件路径，失败返回空字符串
        """
        # 使用锁确保文件写入的线程安全
        with self._file_lock:
            try:
                # 清理文件名
                sanitized_name = self._sanitize_filename(source)

                # 生成唯一的文件名（时间戳 + 微秒 + 计数器，确保唯一性）
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                microsecond = datetime.now().microsecond
                counter = self._get_unique_counter()
                json_filename = f"{sanitized_name}_{timestamp}_{microsecond:06d}_{counter:04d}.json"
                
                # 使用绝对路径，确保多线程环境下路径一致
                output_dir = os.path.abspath(os.path.join(self.output_dir, mode))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                json_filepath = os.path.join(output_dir, json_filename)

                # 准备JSON数据
                json_data = {
                    "source": source,
                    "metadata": metadata or {},
                    "qa_pairs": qa_pairs,
                    "generated_at": timestamp,
                    "total_qa_pairs": len(qa_pairs),
                    "thread_id": threading.current_thread().name  # 添加线程信息用于调试
                }

                # 保存JSON文件
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)

                # 验证文件是否成功写入
                if not os.path.exists(json_filepath):
                    raise IOError(f"文件写入后不存在: {json_filepath}")

                file_size = os.path.getsize(json_filepath)
                logger.info(
                    f"成功保存 {source} 的 {len(qa_pairs)} 个问答对到: {json_filepath} "
                    f"(大小: {file_size} bytes, 线程: {threading.current_thread().name})")

                return json_filepath

            except Exception as e:
                logger.error(
                    f"保存单个PDF问答对时出错 ({source}): {str(e)}, "
                    f"线程: {threading.current_thread().name}",
                    exc_info=True)
                return ""
