#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import re
import io
import numpy as np
from PIL import Image
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
from pix2tex.cli import LatexOCR
from munch import Munch
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDFProcessor:
    """增强型PDF处理类，支持文本、结构和公式提取"""

    def __init__(self, pdf_dir, use_latex_ocr=True):
        """
        初始化PDF处理器

        Args:
            pdf_dir (str): PDF文件所在的目录路径
            use_latex_ocr (bool): 是否使用LatexOCR处理公式
        """
        self.pdf_dir = pdf_dir
        self.use_latex_ocr = use_latex_ocr
        self.latex_ocr = None

        # 如果启用了LaTeX OCR，就初始化模型
        if self.use_latex_ocr:
            try:
                logger.info("正在加载LatexOCR模型...")
                arguments = Munch({
                    'config': 'settings/config.yaml',
                    'checkpoint': 'checkpoints/weights.pth',
                    'no_cuda': False,  # 设置为False以启用CUDA
                    'no_resize': False
                })
                root_logger = logging.getLogger()
                original_level = root_logger.level
                self.latex_ocr = LatexOCR(arguments)
                root_logger.setLevel(original_level)
                logger.info("LatexOCR模型加载完成")
            except Exception as e:
                logger.error(f"加载LatexOCR模型失败: {str(e)}")
                self.use_latex_ocr = False
                logging.getLogger().setLevel(logging.INFO)

        logger.info(f"增强型PDF处理器初始化，目录: {pdf_dir}, 公式OCR: {self.use_latex_ocr}")

    def get_pdf_files(self):
        """
        获取目录中所有的PDF文件（返回绝对路径）

        Returns:
            list: PDF文件绝对路径列表
        """
        pdf_files = []
        try:
            normalized_dir = os.path.abspath(os.path.normpath(self.pdf_dir))
            for file in os.listdir(normalized_dir):
                if file.lower().endswith('.pdf'):
                    abs_path = os.path.abspath(os.path.join(normalized_dir, file))
                    pdf_files.append(abs_path)
            logger.info(f"找到 {len(pdf_files)} 个PDF文件")
            return pdf_files
        except Exception as e:
            logger.error(f"获取PDF文件列表时出错: {str(e)}")
            return []

    def _is_likely_formula(self, block):
        """
        判断文本块是否可能是公式

        Args:
            block: PyMuPDF文本块

        Returns:
            bool: 是否可能是公式
        """
        if "lines" not in block:
            return False

        # 提取块中的所有文本
        block_text = ""
        for line in block["lines"]:
            for span in line.get("spans", []):
                block_text += span.get("text", "")

        # 去除空白
        block_text = block_text.strip()

        # 基本过滤条件
        if not block_text:
            return False

        # 如果文本太长（超过200字符），很可能不是公式
        if len(block_text) > 200:
            return False

        # 如果是纯中文或纯英文段落，不是公式
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', block_text))
        english_words = len(re.findall(r'\b[a-zA-Z]{3,}\b', block_text))

        # 如果中文字符超过4个，很可能是正文
        if chinese_chars > 4:
            return False

        # 如果英文单词超过5个，很可能是正文
        if english_words > 5:
            return False

        # 检查是否包含数学符号或特殊字符
        math_symbols = r'[∫∑∏√±×÷≠≈≤≥∞∂∇∈∉⊂⊃∪∩∧∨¬⊕⊗αβγδεζηθικλμνξοπρστυφχψω]'
        has_math_symbols = bool(re.search(math_symbols, block_text))

        # 检查是否包含常见的数学运算符和数字组合
        has_math_pattern = bool(re.search(r'[+\-*/=^_(){}\[\]<>]', block_text)) and \
            bool(re.search(r'\d', block_text))

        # 检查字体大小差异（公式常有上下标）
        font_sizes = []
        for line in block["lines"]:
            for span in line.get("spans", []):
                font_sizes.append(span.get("size", 0))

        # 如果有明显的字体大小变化（上下标），可能是公式
        has_size_variation = len(set(font_sizes)) > 1 and max(
            font_sizes) - min(font_sizes) > 2

        # 检查宽高比（公式通常比较紧凑）
        bbox = block.get("bbox", (0, 0, 0, 0))
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        # 避免除零
        if height == 0:
            return False

        aspect_ratio = width / height

        # 公式通常宽度不会太大（排除表格标题等）
        if width > 400:
            return False

        # 综合判断
        # 必须满足以下条件之一：
        # 1. 包含数学符号
        # 2. 包含数学运算模式且有字体大小变化
        # 3. 行数很少（1-2行）且包含数学运算模式

        if has_math_symbols:
            return True

        if has_math_pattern and has_size_variation:
            return True

        if len(block["lines"]) <= 2 and has_math_pattern:
            # 额外检查：确保不是普通的编号或页码
            if not re.match(r'^[\d\.\s]+$', block_text):
                return True

        return False

    def _detect_and_extract_formulas(self, doc, page_num):
        """
        检测并提取页面中的公式

        Args:
            doc: PyMuPDF文档对象
            page_num: 页码

        Returns:
            list: 公式列表，每个元素是(bbox, latex)元组
        """
        if not self.use_latex_ocr or self.latex_ocr is None:
            return []

        formulas = []
        page = doc[page_num]

        # 使用PyMuPDF检测可能是公式的区域
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" not in block:
                continue

            # 使用更智能的公式检测
            if not self._is_likely_formula(block):
                continue

            # 获取区域坐标
            bbox = block["bbox"]
            x0, y0, x1, y1 = bbox

            # 为了更好地识别，稍微扩大区域
            margin = 5
            x0 = max(0, x0 - margin)
            y0 = max(0, y0 - margin)
            x1 = min(page.rect.width, x1 + margin)
            y1 = min(page.rect.height, y1 + margin)

            # 渲染区域为图像
            pix = page.get_pixmap(
                clip=(x0, y0, x1, y1), matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            # logger.info(f"准备OCR识别公式，图像大小: {img.size}")

            # 使用LatexOCR识别公式
            try:
                latex = self.latex_ocr(img)
                if latex and len(latex) > 5:  # 确保有意义的输出
                    formulas.append((bbox, latex))
                    # logger.info(f"✓ 成功识别公式: {latex[:50]}...")
            except Exception as e:
                logger.debug(f"公式识别失败: {str(e)}")
                continue

        return formulas

    def _process_with_pdfplumber(self, pdf_path):
        """
        使用pdfplumber提取文本

        Args:
            pdf_path: PDF文件路径

        Returns:
            str: 提取的文本
        """
        text = ""
        try:
            # 确保使用绝对路径
            abs_pdf_path = os.path.abspath(pdf_path)
            with pdfplumber.open(abs_pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(x_tolerance=1, y_tolerance=3)
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            logger.error(f"使用pdfplumber提取文本失败: {str(e)}")
            return ""

    def _process_with_pymupdf(self, pdf_path):
        """
        使用PyMuPDF提取结构化内容

        Args:
            pdf_path: PDF文件路径

        Returns:
            tuple: (结构化文本, 公式列表)
        """
        structured_text = ""
        all_formulas = []

        try:
            # 确保使用绝对路径（PyMuPDF对路径格式更敏感）
            abs_pdf_path = os.path.abspath(pdf_path)
            # 打开PDF文档
            doc = fitz.open(abs_pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]

                # 提取普通文本
                page_text = page.get_text("text")

                # 检测并提取公式
                formulas = self._detect_and_extract_formulas(doc, page_num)
                all_formulas.extend([(page_num, *formula)
                                    for formula in formulas])

                # 添加页面文本
                structured_text += page_text + "\n"

            return structured_text, all_formulas
        except Exception as e:
            logger.error(f"使用PyMuPDF提取结构化内容失败: {str(e)}")
            return "", []

    def _integrate_content(self, basic_text, structured_text, formulas):
        """
        整合不同来源的内容，确保文本连贯，公式位于正确位置

        Args:
            basic_text: 基本文本
            structured_text: 结构化文本
            formulas: 公式列表

        Returns:
            str: 整合后的最终文本
        """
        # 如果没有检测到公式，直接返回更好的文本版本
        if not formulas:
            return structured_text if structured_text else basic_text

        # 将公式按照页码和位置排序
        formulas.sort(key=lambda x: (x[0], x[1][1]))  # 按页码和y坐标排序

        # 将公式插入到文本中适当位置
        lines = structured_text.split('\n')
        result = []
        formula_idx = 0

        # 假设每个公式应该插入在最近的文本段落后面
        current_page = 0

        for i, line in enumerate(lines):
            # 检查是否需要插入公式
            result.append(line)

            if formula_idx < len(formulas):
                formula_page, formula_bbox, formula_latex = formulas[formula_idx]

                if i < len(lines) - 1:
                    next_line = lines[i + 1]
                    if not next_line.strip():  # 如果下一行是空行，可能是段落结束
                        if formula_page == current_page:
                            # 在段落结束处插入公式
                            result.append(f"[FORMULA: {formula_latex}]")
                            formula_idx += 1

            # 估计当前页码
            if not line.strip() and i > 0 and i < len(lines) - 1:
                # 检测可能的页面变化
                if lines[i - 1].strip() and lines[i + 1].strip():
                    current_page += 1

        # 将剩余的公式添加到文档末尾
        while formula_idx < len(formulas):
            _, _, formula_latex = formulas[formula_idx]
            result.append(f"[FORMULA: {formula_latex}]")
            formula_idx += 1

        return '\n'.join(result)

    def extract_text_from_pdf(self, pdf_path):
        """
        从PDF文件中提取增强的文本内容

        Args:
            pdf_path (str): PDF文件路径

        Returns:
            tuple: (增强文本内容, 文件名, 元数据)
        """
        try:
            # 转换为绝对路径，确保所有PDF库都能正确访问
            pdf_path = os.path.abspath(pdf_path)
            pdf_filename = os.path.basename(pdf_path)
            logger.info(f"开始增强处理PDF文件: {pdf_filename}")

            # 使用PyPDF2提取基本文本
            basic_text = ""
            metadata = {}

            try:
                reader = PyPDF2.PdfReader(pdf_path)
                # 提取元数据
                if reader.metadata:
                    metadata = {
                        "title": reader.metadata.get('/Title', ''),
                        "author": reader.metadata.get('/Author', ''),
                        "subject": reader.metadata.get('/Subject', ''),
                        "creator": reader.metadata.get('/Creator', ''),
                        "producer": reader.metadata.get('/Producer', '')
                    }

                # 提取文本
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        basic_text += page_text + "\n"
            except Exception as e:
                logger.error(f"使用PyPDF2提取文本失败: {str(e)}")

            # # 使用pdfplumber提取更精确的文本
            # plumber_text = self._process_with_pdfplumber(pdf_path)

            # 使用PyMuPDF提取结构化文本和公式
            logger.info("使用PyMuPDF提取结构化文本和公式")
            structured_text, formulas = self._process_with_pymupdf(pdf_path)

            # 整合内容
            final_text = self._integrate_content(
                basic_text, structured_text, formulas)

            # 后处理：清理文本中的特殊字符和多余空白
            final_text = re.sub(r'\s+', ' ', final_text)  # 替换多个空白为单个空格
            final_text = re.sub(
                r'(\n\s*){3,}', '\n\n', final_text)  # 最多保留两个连续换行

            content_length = len(final_text)
            formula_count = len(formulas)
            logger.info(
                f"成功从 {pdf_filename} 提取文本，共 {content_length} 个字符，检测到 {formula_count} 个公式")

            return final_text, pdf_filename, metadata
        except Exception as e:
            logger.error(f"从PDF文件 {pdf_path} 提取文本时出错: {str(e)}")
            return "", os.path.basename(pdf_path), {}

    def extract_pdfs_to_txt(self, txt_dir, max_workers=3):
        """
        提取PDF内容并保存为txt文件
        
        Args:
            txt_dir (str): txt文件保存目录
            max_workers (int): 最大并行处理的文件数
        
        Returns:
            tuple: (成功数, 跳过数, 失败数)
        """
        # 确保txt目录存在
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir, exist_ok=True)
            logger.info(f"创建txt目录: {txt_dir}")
        
        pdf_files = self.get_pdf_files()
        
        if not pdf_files:
            logger.warning("没有找到PDF文件")
            print("警告: 没有找到PDF文件")
            return 0, 0, 0

        txt_files = []
        # 生成对应的txt文件名（去除.pdf扩展名，添加.txt）
        for pdf_file in pdf_files:
            pdf_filename = os.path.basename(pdf_file)
            txt_filename = os.path.splitext(pdf_filename)[0] + '.txt'
            txt_files.append(os.path.abspath(os.path.join(txt_dir, txt_filename)))

        logger.info(f"找到 {len(pdf_files)} 个PDF文件，开始提取文本...")
        print(f"找到 {len(pdf_files)} 个PDF文件，开始提取文本...")
        
        success_count = 0
        skip_count = 0
        failed_count = 0
        failed_files = []
        
        def process_single_pdf(pdf_path, txt_filepath):
            """处理单个PDF文件"""
            pdf_filename = os.path.basename(pdf_path)
            
            # 检查txt文件是否已存在
            if os.path.exists(txt_filepath):
                logger.info(f"跳过已存在的文件: {txt_filepath}")
                return 'skip', pdf_filename
            
            try:
                # 提取文本
                content, _, metadata = self.extract_text_from_pdf(pdf_path)
                
                if not content:
                    logger.warning(f"PDF文件 {pdf_filename} 没有提取到内容")
                    return 'failed', pdf_filename
                
                # 保存到txt文件
                with open(txt_filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"成功提取并保存: {txt_filepath} (长度: {len(content)} 字符)")
                return 'success', pdf_filename
                
            except Exception as e:
                logger.error(f"处理PDF文件 {pdf_path} 时出错: {str(e)}", exc_info=True)
                return 'failed', pdf_filename
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_pdf = {
                executor.submit(process_single_pdf, pdf, txt): pdf for pdf, txt in zip(pdf_files, txt_files)
            }
            
            # 收集结果
            for future in as_completed(future_to_pdf):
                pdf = future_to_pdf[future]
                try:
                    status, filename = future.result()
                    if status == 'success':
                        success_count += 1
                    elif status == 'skip':
                        skip_count += 1
                    else:
                        failed_count += 1
                        failed_files.append(filename)
                except Exception as e:
                    logger.error(f"获取文件 {pdf} 的处理结果时出错: {str(e)}")
                    failed_count += 1
                    failed_files.append(os.path.basename(pdf))
        
        # 打印统计信息
        logger.info(f"提取完成: 成功 {success_count}, 跳过 {skip_count}, 失败 {failed_count}")
        print(f"\n提取统计:")
        print(f"- 成功: {success_count} 个文件")
        print(f"- 跳过: {skip_count} 个文件")
        print(f"- 失败: {failed_count} 个文件")
        
        if failed_files:
            print(f"\n失败的文件:")
            for file in failed_files:
                print(f"  - {file}")
        
        print(f"\n所有txt文件已保存到: {os.path.abspath(txt_dir)}")
        
        return success_count, skip_count, failed_count
