#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import json
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from .pdf_processor import PDFProcessor
from .llm_client import LLMClient
from .prompts import PromptTemplates

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QAGenerator:
    """问答生成器类"""

    def __init__(self, pdf_dir="pdf_files", num_qa_pairs=20, max_workers=3,
                 api_max_retries=3, api_retry_delay=2,
                 use_latex_ocr=True, answer_max_workers=5, excel_writer=None, mode='normal'):
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
            excel_writer (ExcelWriter): Excel写入器实例，用于保存单个PDF结果
            mode (str): 模式: normal-正常模式生成大量较短的且相对常见基础的知识问答对, pro-专业模式生成少量但长篇的深入研讨问答对
        """
        self.pdf_processor = PDFProcessor(pdf_dir, use_latex_ocr)
        self.llm_client = LLMClient(
            max_retries=api_max_retries, retry_delay=api_retry_delay)
        self.prompt_templates = PromptTemplates()  # 初始化提示词模板
        self.num_qa_pairs = num_qa_pairs
        self.max_workers = max_workers
        self.answer_max_workers = answer_max_workers
        self.excel_writer = excel_writer  # 用于保存单个PDF结果
        self.failed_files = []  # 用于记录处理失败的文件
        self.mode = mode

        logger.info(
            f"问答生成器初始化，目录: {pdf_dir}，每个PDF生成 {num_qa_pairs} 个问答对，答案生成并行数: {answer_max_workers}")

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
        div_num_q = int(len(content) / 2000)
        # 如果按每2000字一个问题分出的问题数小于num_questions，则将num_questions设置为div_num_q
        if div_num_q < num_questions:
            num_questions = max(div_num_q, 1)
            logger.warning(
                f"按每2000字一个问题分出的问题数小于num_questions，将num_questions设置为{num_questions}")
        prompt = self.prompt_templates.get_pro_question_generation_prompt(
            content, num_questions, metadata)

        # 调用API生成问题
        questions = self.llm_client.generate_questions(
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

        # logger.info(f"开始生成问题的答案: {question[:50]}...")

        # 准备答案生成的提示词
        prompt = self.prompt_templates.get_pro_answer_generation_prompt(
            question, content, metadata)

        # 调用API生成答案
        answer = self.llm_client.generate_single_answer(prompt)

        elapsed_time = time.time() - start_time

        if answer:
            # logger.info(
            #     f"成功生成答案，长度: {len(answer)} 字符，耗时: {elapsed_time:.2f} 秒")
            pass
        else:
            logger.warning(f"答案生成失败，耗时: {elapsed_time:.2f} 秒")

        return answer

    def _is_pdf_processed(self, pdf_filename):
        """
        检查PDF文件是否已经处理过（通过检查输出目录中是否存在对应的JSON文件）

        Args:
            pdf_filename (str): PDF文件名（带扩展名）

        Returns:
            bool: 如果已处理过返回True，否则返回False
        """
        # 如果没有excel_writer，无法检查，返回False
        if not self.excel_writer:
            return False

        output_dir = self.excel_writer.output_dir

        # 检查输出目录是否存在
        if not os.path.exists(output_dir):
            return False

        # 遍历输出目录中的所有JSON文件
        json_files = glob.glob(os.path.join(output_dir, self.mode, "*.json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 检查source字段是否匹配当前PDF文件名
                    if 'source' in data and data['source'] == pdf_filename:
                        logger.info(f"PDF文件 {pdf_filename} 已处理过（找到已存在的JSON文件: {os.path.basename(json_file)}）")
                        return True
            except Exception as e:
                # 如果读取JSON文件失败，继续检查下一个文件
                logger.debug(f"读取JSON文件 {json_file} 时出错: {e}")
                continue

        return False

    def _process_document(self, content, filename, metadata=None):
        """
        统一的文档处理流程入口函数，根据mode分发到不同的处理函数
        
        Args:
            content (str): 文档内容
            filename (str): 文件名
            metadata (dict): 文档元数据
            
        Returns:
            tuple: (问答对列表, 文件名, 原始内容, 元数据, 是否成功)
        """
        try:
            # 根据mode选择不同的处理方式
            if self.mode == 'pro':
                qa_pairs, success = self._process_document_pro(content, filename, metadata)
            elif self.mode == 'normal':
                qa_pairs, success = self._process_document_normal(content, filename, metadata)
            else:
                logger.error(f"未知的模式: {self.mode}")
                return [], filename, content, metadata, False

            if not success or not qa_pairs:
                return [], filename, content, metadata, False

            # 统一的统计和保存逻辑
            self._log_and_save_results(qa_pairs, filename, metadata)

            return qa_pairs, filename, content, metadata, True

        except Exception as e:
            logger.error(f"处理文档 {filename} 时出错: {str(e)}", exc_info=True)
            return [], filename, content, metadata, False

    def _process_document_pro(self, content, filename, metadata=None):
        """
        Pro模式文档处理：先生成问题集，然后逐个生成答案
        
        Args:
            content (str): 文档内容
            filename (str): 文件名
            metadata (dict): 文档元数据
            
        Returns:
            tuple: (问答对列表, 是否成功)
        """
        try:
            # 第二步：生成问题列表
            logger.info(f"[第2步] 生成 {self.num_qa_pairs} 个问题")
            questions = self._generate_questions(
                content, self.num_qa_pairs, metadata)

            if not questions:
                logger.warning(f"文件 {filename} 生成问题失败")
                return [], False

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
                return [], False

            return qa_pairs, True

        except Exception as e:
            logger.error(f"Pro模式处理文档 {filename} 时出错: {str(e)}", exc_info=True)
            return [], False

    def _process_document_normal(self, content, filename, metadata=None):
        """
        Normal模式文档处理：一次性生成所有问答对
        
        Args:
            content (str): 文档内容
            filename (str): 文件名
            metadata (dict): 文档元数据
            
        Returns:
            tuple: (问答对列表, 是否成功)
        """
        try:
            # 计算问题数量限制（类似pro模式）
            num_questions = self.num_qa_pairs
            div_num_q = int(len(content) / 800)
            if div_num_q < num_questions:
                num_questions = max(div_num_q, 1)
                logger.warning(
                    f"按每800字一个问题分出的问题数小于num_qa_pairs，将num_questions设置为{num_questions}")

            # 生成prompt
            logger.info(f"[Normal模式] 一次性生成 {num_questions} 个问答对")
            prompt = self.prompt_templates.get_normal_qa_pair_generation_prompt(
                content, num_questions, metadata)

            # 调用API一次性生成所有问答对
            qa_pairs = self.llm_client.generate_qa_pairs(prompt, num_questions)

            if not qa_pairs:
                logger.warning(f"文件 {filename} 生成问答对失败")
                return [], False

            logger.info(f"[Normal模式] 成功生成 {len(qa_pairs)} 个问答对")

            # 验证问答对格式
            valid_qa_pairs = []
            for qa in qa_pairs:
                if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                    valid_qa_pairs.append({
                        "question": qa['question'],
                        "answer": qa['answer']
                    })
                else:
                    logger.warning(f"跳过无效的问答对格式: {qa}")

            if not valid_qa_pairs:
                logger.warning(f"文件 {filename} 没有有效的问答对")
                return [], False

            logger.info(f"[Normal模式] 验证后有效问答对: {len(valid_qa_pairs)} 个")
            return valid_qa_pairs, True

        except Exception as e:
            logger.error(f"Normal模式处理文档 {filename} 时出错: {str(e)}", exc_info=True)
            return [], False

    def _log_and_save_results(self, qa_pairs, filename, metadata):
        """
        统一的统计和保存逻辑
        
        Args:
            qa_pairs (list): 问答对列表
            filename (str): 文件名
            metadata (dict): 文档元数据
        """
        # 计算统计信息
        total_answer_length = sum(len(qa['answer']) for qa in qa_pairs)
        avg_answer_length = total_answer_length / len(qa_pairs) if qa_pairs else 0

        logger.info(f"========== 文件 {filename} 处理完成 ==========")
        logger.info(f"统计信息:")
        logger.info(f"  - 成功生成问答对: {len(qa_pairs)} 个")
        logger.info(f"  - 平均答案长度: {avg_answer_length:.0f} 字符")
        logger.info(
            f"  - 最短答案: {min(len(qa['answer']) for qa in qa_pairs)} 字符")
        logger.info(
            f"  - 最长答案: {max(len(qa['answer']) for qa in qa_pairs)} 字符")

        # 立即保存到单独的JSON文件
        if self.excel_writer:
            saved_path = self.excel_writer.save_single_pdf_qa(
                qa_pairs, filename, metadata, self.mode)
            if saved_path:
                logger.info(f"已保存到文件: {saved_path}")

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

            # 调用统一的文档处理流程
            return self._process_document(content, filename, metadata)

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

        # 过滤掉已经处理过的PDF文件
        pdf_files_to_process = []
        skipped_files = []
        
        for pdf_file in pdf_files:
            pdf_filename = os.path.basename(pdf_file)
            if self._is_pdf_processed(pdf_filename):
                skipped_files.append(pdf_filename)
            else:
                pdf_files_to_process.append(pdf_file)

        if skipped_files:
            logger.info(f"跳过 {len(skipped_files)} 个已处理的PDF文件:")
            for skipped_file in skipped_files:
                logger.info(f"  - {skipped_file}")

        if not pdf_files_to_process:
            logger.info("所有PDF文件都已处理过，无需重新处理")
            return [], []

        results = []
        self.failed_files = []  # 重置失败文件列表

        logger.info(f"开始处理 {len(pdf_files_to_process)} 个PDF文件（共 {len(pdf_files)} 个，跳过 {len(skipped_files)} 个）")

        # 使用线程池并行处理PDF文件
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_pdf = {executor.submit(
                self.process_pdf, pdf): pdf for pdf in pdf_files_to_process}

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
            f"处理完成：共 {len(pdf_files)} 个PDF文件，跳过 {len(skipped_files)} 个已处理的文件，"
            f"实际处理 {len(pdf_files_to_process)} 个，成功: {len(results)}，失败: {len(self.failed_files)}")

        # 打印处理失败的文件列表
        if self.failed_files:
            logger.warning("以下文件处理失败:")
            for failed_file in self.failed_files:
                logger.warning(f"  - {failed_file}")

        return results, self.failed_files

    def generate_qa_from_txt_files(self, txt_dir):
        """
        从txt文件生成问答对
        
        Args:
            txt_dir (str): txt文件所在目录
        
        Returns:
            tuple: (问答对列表, 失败文件列表)
                  问答对列表: 每个元素是一个四元组 (问答对列表, 源文件名, 原始内容, 元数据)
                  失败文件列表: 处理失败的文件名列表
        """
        
        # 获取所有txt文件
        txt_files = glob.glob(os.path.join(txt_dir, "*.txt"))
        
        if not txt_files:
            logger.warning(f"在目录 {txt_dir} 中没有找到txt文件")
            return [], []
        
        # 过滤掉已经处理过的txt文件
        txt_files_to_process = []
        skipped_files = []
        
        for txt_file in txt_files:
            txt_filename = os.path.basename(txt_file)
            # 从文件名推断原始PDF文件名（去除.txt扩展名，添加.pdf）
            pdf_filename = os.path.splitext(txt_filename)[0] + '.pdf'
            if self._is_pdf_processed(pdf_filename):
                skipped_files.append(txt_filename)
            else:
                txt_files_to_process.append(txt_file)
        
        if skipped_files:
            logger.info(f"跳过 {len(skipped_files)} 个已处理的txt文件:")
            for skipped_file in skipped_files:
                logger.info(f"  - {skipped_file}")
        
        if not txt_files_to_process:
            logger.info("所有txt文件都已处理过，无需重新处理")
            return [], []
        
        logger.info(f"开始处理 {len(txt_files_to_process)} 个txt文件（共 {len(txt_files)} 个，跳过 {len(skipped_files)} 个）")
        
        results = []
        self.failed_files = []  # 重置失败文件列表
        
        def process_single_txt(txt_path):
            """处理单个txt文件"""
            txt_filename = os.path.basename(txt_path)
            # 从文件名推断原始PDF文件名（去除.txt扩展名，添加.pdf）
            pdf_filename = os.path.splitext(txt_filename)[0] + '.pdf'
            
            try:
                logger.info(f"========== 开始处理txt文件: {txt_filename} ==========")
                
                # 读取txt文件内容
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if not content:
                    logger.warning(f"txt文件 {txt_filename} 内容为空")
                    return [], pdf_filename, "", {}, False
                
                logger.info(f"[第1步] 成功读取txt文件，长度: {len(content)} 字符")
                
                # 调用统一的文档处理流程
                return self._process_document(content, pdf_filename, {})
                
            except Exception as e:
                logger.error(f"处理txt文件 {txt_path} 时出错: {str(e)}", exc_info=True)
                return [], pdf_filename, "", {}, False
        
        # 使用线程池并行处理txt文件
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_txt = {
                executor.submit(process_single_txt, txt): txt 
                for txt in txt_files_to_process
            }
            
            # 收集结果
            for future in as_completed(future_to_txt):
                txt = future_to_txt[future]
                txt_name = os.path.basename(txt)
                try:
                    qa_pairs, filename, content, metadata, success = future.result()
                    if success:
                        results.append((qa_pairs, filename, content, metadata))
                    else:
                        self.failed_files.append(filename)
                except Exception as e:
                    logger.error(f"获取文件 {txt} 的处理结果时出错: {str(e)}")
                    self.failed_files.append(txt_name)
        
        logger.info(
            f"处理完成：共 {len(txt_files)} 个txt文件，跳过 {len(skipped_files)} 个已处理的文件，"
            f"实际处理 {len(txt_files_to_process)} 个，成功: {len(results)}，失败: {len(self.failed_files)}")
        
        # 打印处理失败的文件列表
        if self.failed_files:
            logger.warning("以下文件处理失败:")
            for failed_file in self.failed_files:
                logger.warning(f"  - {failed_file}")
        
        return results, self.failed_files