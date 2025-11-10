#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from .pdf_processor import PDFProcessor
from .deepseek_client import DeepSeekClient

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QAGenerator:
    """问答生成器类"""

    def __init__(self, pdf_dir="pdf_files", num_qa_pairs=20, max_workers=3,
                 api_max_retries=3, api_retry_delay=2,
                 use_latex_ocr=True, answer_max_workers=5):
        """
        初始化问答生成器

        Args:
            pdf_dir (str): PDF文件目录
            num_qa_pairs (int): 每个PDF生成的问答对数量
            max_workers (int): 最大并行处理的文件数
            api_max_retries (int): API调用最大重试次数
            api_retry_delay (int): API调用重试间隔(秒)
            use_latex_ocr (bool): 是否使用LaTeX OCR
            answer_max_workers (int): 答案生成的并行线程数（默认5）
        """
        self.pdf_processor = PDFProcessor(pdf_dir, use_latex_ocr)
        self.deepseek_client = DeepSeekClient(
            max_retries=api_max_retries, retry_delay=api_retry_delay)
        self.num_qa_pairs = num_qa_pairs
        self.max_workers = max_workers
        self.answer_max_workers = answer_max_workers
        self.failed_files = []  # 用于记录处理失败的文件

        logger.info(
            f"问答生成器初始化，目录: {pdf_dir}，每个PDF生成 {num_qa_pairs} 个问答对，答案生成并行数: {answer_max_workers}")

    def _get_question_generation_prompt(self, content, num_questions, metadata=None):
        """生成问题列表的prompt模板"""
        return f"""
你是一位卫星互联网领域的资深研究员和博士导师，需要根据以下学术文献内容，提炼出{num_questions}个高质量的研究问题。

【角色定位】：
你是卫星互联网领域的专家，精通卫星通信、信号处理、网络协议、轨道设计等相关技术。

【问题类型指引】：
建议但不限于以下类型的问题（请根据文档实际内容灵活选择）：
1. 理论原理类：核心概念、基础理论、数学模型、物理原理
2. 技术方法类：算法设计、实现方案、优化策略、关键技术
3. 系统架构类：系统设计、模块组成、协议栈、接口定义
4. 性能分析类：性能指标、复杂度分析、对比优势、适用场景
5. 应用场景类：实际应用、工程实践、问题解决、未来展望

【重要要求】：
1. 问题必须基于文档内容，提取文档中真实存在的知识点
2. 不要问"本文/该论文怎么样"这类针对论文本身的问题
3. 问题应该是领域知识问题，而非论文工作评价
4. 问题难度应达到博士或资深研究者水平
5. 问题应该具有提炼能力，抓住核心要点
6. 如果文档中没有某类型的内容，可以自主调整问题类型

【文档内容】：
{content}

请仅返回JSON格式的问题列表，每个问题包含'question'和'type'字段：
[
  {{"question": "问题1的具体内容", "type": "理论原理类"}},
  {{"question": "问题2的具体内容", "type": "技术方法类"}},
  ...
]"""

    def _get_answer_generation_prompt(self, question, content, metadata=None):
        """生成单个答案的prompt模板"""
        return f"""
你是一位卫星互联网领域的资深研究员和博士导师，需要针对以下问题，基于提供的学术文献内容，给出一个详尽、专业、体系化的答案。

【角色定位】：
你是卫星互联网领域的专家，精通卫星通信、信号处理、网络协议、轨道设计等相关技术。

【问题】：
{question}

【答案要求】：
1. **长度要求**：答案应至少500-800字，确保内容充分详尽
2. **内容结构**：
   - 首先给出清晰的定义或概念说明
   - 详细阐述理论原理和技术机制
   - 如涉及数学模型，给出公式推导和参数说明
   - 提供技术细节和实现要点
   - 必要时给出实例说明或应用场景
   - 说明与相关技术的关系或对比
3. **专业性要求**：
   - 使用准确的学术术语和技术词汇
   - 体现专家级的深度理解
   - 答案应成体系化，有清晰的逻辑层次
4. **准确性要求**：
   - 答案必须基于提供的文档内容
   - 不要编造文档中不存在的信息
   - 如果文档信息不足，说明已知部分即可

【参考文档】：
{content}

请直接返回详细的答案内容（不需要JSON格式，直接返回答案文本）："""

    def _generate_questions(self, content, num_questions, metadata=None):
        """
        生成问题列表

        Args:
            content (str): PDF内容
            num_questions (int): 问题数量
            metadata (dict): PDF元数据

        Returns:
            list: 问题列表（字符串列表）
        """
        logger.info(f"开始生成 {num_questions} 个问题")

        # 准备问题生成的提示词
        prompt = self._get_question_generation_prompt(
            content, num_questions, metadata)

        # 调用API生成问题
        questions = self.deepseek_client.generate_questions(
            prompt, num_questions)

        if questions:
            logger.info(f"成功生成 {len(questions)} 个问题")
        else:
            logger.warning("问题生成失败")

        return questions

    def _generate_answer(self, question, content, metadata=None):
        """
        为单个问题生成答案

        Args:
            question (str): 问题
            content (str): PDF内容
            metadata (dict): PDF元数据

        Returns:
            str: 生成的答案
        """
        import time
        start_time = time.time()

        logger.info(f"开始生成问题的答案: {question[:50]}...")

        # 准备答案生成的提示词
        prompt = self._get_answer_generation_prompt(
            question, content, metadata)

        # 调用API生成答案
        answer = self.deepseek_client.generate_single_answer(prompt)

        elapsed_time = time.time() - start_time

        if answer:
            logger.info(
                f"成功生成答案，长度: {len(answer)} 字符，耗时: {elapsed_time:.2f} 秒")
        else:
            logger.warning(f"答案生成失败，耗时: {elapsed_time:.2f} 秒")

        return answer

    def process_pdf(self, pdf_path):
        """
        处理单个PDF文件（两阶段生成：先生成问题，再并行生成答案）

        Args:
            pdf_path (str): PDF文件路径

        Returns:
            tuple: (问答对列表, 源文件名, 原始内容, 元数据, 是否成功)
        """
        filename = os.path.basename(pdf_path)
        try:
            logger.info(f"========== 开始处理PDF文件: {filename} ==========")

            # 第一步：从PDF提取文本
            logger.info(f"[第1步] 从PDF提取文本")
            content, filename, metadata = self.pdf_processor.extract_text_from_pdf(
                pdf_path)

            if not content:
                logger.warning(f"PDF文件 {filename} 没有提取到内容")
                return [], filename, "", {}, False

            logger.info(f"[第1步] 成功提取文本，长度: {len(content)} 字符")

            # 第二步：生成问题列表
            logger.info(f"[第2步] 生成 {self.num_qa_pairs} 个问题")
            questions = self._generate_questions(
                content, self.num_qa_pairs, metadata)

            if not questions:
                logger.warning(f"文件 {filename} 生成问题失败")
                return [], filename, content, metadata, False

            logger.info(f"[第2步] 成功生成 {len(questions)} 个问题")

            # 第三步：使用线程池并行生成答案
            logger.info(f"[第3步] 使用 {self.answer_max_workers} 个线程并行生成答案")
            qa_pairs = []
            failed_count = 0

            with ThreadPoolExecutor(max_workers=self.answer_max_workers) as executor:
                # 提交所有答案生成任务
                future_to_question = {
                    executor.submit(self._generate_answer, q, content, metadata): q
                    for q in questions
                }

                # 收集结果
                for future in as_completed(future_to_question):
                    question = future_to_question[future]
                    try:
                        answer = future.result()
                        if answer:
                            qa_pairs.append({
                                "question": question,
                                "answer": answer
                            })
                        else:
                            failed_count += 1
                            logger.warning(
                                f"问题答案生成失败（空答案）: {question[:50]}...")
                    except Exception as e:
                        failed_count += 1
                        logger.error(
                            f"生成问题答案时出现异常: {question[:50]}..., 错误: {str(e)}")

            logger.info(
                f"[第3步] 答案生成完成，成功: {len(qa_pairs)}, 失败: {failed_count}")

            if not qa_pairs:
                logger.warning(f"文件 {filename} 所有答案生成均失败")
                return [], filename, content, metadata, False

            # 计算统计信息
            total_answer_length = sum(len(qa['answer']) for qa in qa_pairs)
            avg_answer_length = total_answer_length / \
                len(qa_pairs) if qa_pairs else 0

            logger.info(f"========== 文件 {filename} 处理完成 ==========")
            logger.info(f"统计信息:")
            logger.info(f"  - 成功生成问答对: {len(qa_pairs)} 个")
            logger.info(f"  - 平均答案长度: {avg_answer_length:.0f} 字符")
            logger.info(
                f"  - 最短答案: {min(len(qa['answer']) for qa in qa_pairs)} 字符")
            logger.info(
                f"  - 最长答案: {max(len(qa['answer']) for qa in qa_pairs)} 字符")

            return qa_pairs, filename, content, metadata, True

        except Exception as e:
            logger.error(f"处理PDF文件 {pdf_path} 时出错: {str(e)}", exc_info=True)
            return [], filename, "", {}, False

    def generate_qa_from_pdfs(self):
        """
        从所有PDF文件生成问答对

        Returns:
            tuple: (问答对列表, 失败文件列表)
                  问答对列表: 每个元素是一个四元组 (问答对列表, 源文件名, 原始内容, 元数据)
                  失败文件列表: 处理失败的文件名列表
        """
        pdf_files = self.pdf_processor.get_pdf_files()

        if not pdf_files:
            logger.warning("没有找到PDF文件")
            return [], []

        results = []
        self.failed_files = []  # 重置失败文件列表

        # 使用线程池并行处理PDF文件
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_pdf = {executor.submit(
                self.process_pdf, pdf): pdf for pdf in pdf_files}

            # 收集结果
            for future in as_completed(future_to_pdf):
                pdf = future_to_pdf[future]
                pdf_name = os.path.basename(pdf)
                try:
                    qa_pairs, filename, content, metadata, success = future.result()
                    if success:
                        results.append((qa_pairs, filename, content, metadata))
                    else:
                        self.failed_files.append(filename)
                except Exception as e:
                    logger.error(f"获取文件 {pdf} 的处理结果时出错: {str(e)}")
                    self.failed_files.append(pdf_name)

        logger.info(
            f"共处理了 {len(pdf_files)} 个PDF文件，成功: {len(results)}，失败: {len(self.failed_files)}")

        # 打印处理失败的文件列表
        if self.failed_files:
            logger.warning("以下文件处理失败:")
            for failed_file in self.failed_files:
                logger.warning(f"  - {failed_file}")

        return results, self.failed_files
