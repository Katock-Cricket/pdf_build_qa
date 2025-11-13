#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统计脚本：统计问答对涉及的文档数、问题长度分布、答案长度分布
"""

import os
import json
import glob
from collections import Counter, defaultdict
import argparse


def load_json_files(output_dir):
    """加载所有JSON文件"""
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    return json_files


def collect_statistics(output_dir):
    """收集统计信息"""
    json_files = load_json_files(output_dir)
    
    if not json_files:
        print(f"警告: 在 {output_dir} 目录下未找到JSON文件")
        return None
    
    # 统计变量
    documents = set()  # 文档集合（去重）
    question_lengths = []  # 所有问题的长度
    answer_lengths = []  # 所有答案的长度
    total_qa_pairs = 0  # 总问答对数
    
    # 遍历所有JSON文件
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 统计文档
                if 'source' in data:
                    documents.add(data['source'])
                
                # 统计问答对
                if 'qa_pairs' in data and isinstance(data['qa_pairs'], list):
                    for qa_pair in data['qa_pairs']:
                        if 'question' in qa_pair:
                            question_lengths.append(len(qa_pair['question']))
                        if 'answer' in qa_pair:
                            answer_lengths.append(len(qa_pair['answer']))
                        total_qa_pairs += 1
                        
        except Exception as e:
            print(f"警告: 读取文件 {json_file} 时出错: {e}")
            continue
    
    return {
        'documents': documents,
        'question_lengths': question_lengths,
        'answer_lengths': answer_lengths,
        'total_qa_pairs': total_qa_pairs,
        'total_files': len(json_files)
    }


def calculate_distribution(lengths):
    """计算长度分布统计"""
    if not lengths:
        return None
    
    lengths_sorted = sorted(lengths)
    n = len(lengths)
    
    stats = {
        'count': n,
        'min': lengths_sorted[0],
        'max': lengths_sorted[-1],
        'mean': sum(lengths) / n,
        'median': lengths_sorted[n // 2] if n % 2 == 1 else (lengths_sorted[n // 2 - 1] + lengths_sorted[n // 2]) / 2,
        'p25': lengths_sorted[n // 4] if n >= 4 else lengths_sorted[0],
        'p75': lengths_sorted[3 * n // 4] if n >= 4 else lengths_sorted[-1],
    }
    
    # 计算分区间分布
    if stats['max'] > stats['min']:
        bins = [0, 50, 100, 200, 500, 1000, 2000, float('inf')]
        bin_labels = ['0-50', '51-100', '101-200', '201-500', '501-1000', '1001-2000', '2000+']
        distribution = defaultdict(int)
        
        for length in lengths:
            for i, threshold in enumerate(bins[1:], 0):
                if length <= threshold:
                    distribution[bin_labels[i]] += 1
                    break
        
        stats['distribution'] = dict(distribution)
    else:
        stats['distribution'] = {}
    
    return stats


def print_statistics(stats):
    """打印统计结果"""
    print("=" * 80)
    print("问答对统计报告")
    print("=" * 80)
    
    # 文档统计
    print(f"\n【文档统计】")
    print(f"  涉及的文档数: {len(stats['documents'])}")
    print(f"  处理的JSON文件数: {stats['total_files']}")
    print(f"  总问答对数: {stats['total_qa_pairs']}")
    
    # 问题长度分布
    print(f"\n【问题长度分布】")
    q_stats = calculate_distribution(stats['question_lengths'])
    if q_stats:
        print(f"  总问题数: {q_stats['count']}")
        print(f"  最短: {q_stats['min']} 字符")
        print(f"  最长: {q_stats['max']} 字符")
        print(f"  平均: {q_stats['mean']:.2f} 字符")
        print(f"  中位数: {q_stats['median']:.2f} 字符")
        print(f"  25%分位数: {q_stats['p25']} 字符")
        print(f"  75%分位数: {q_stats['p75']} 字符")
        print(f"\n  长度区间分布:")
        for bin_label, count in sorted(q_stats['distribution'].items(), 
                                       key=lambda x: int(x[0].split('-')[0]) if '-' in x[0] else 9999):
            percentage = (count / q_stats['count']) * 100
            print(f"    {bin_label:12s}: {count:6d} ({percentage:5.2f}%)")
    
    # 答案长度分布
    print(f"\n【答案长度分布】")
    a_stats = calculate_distribution(stats['answer_lengths'])
    if a_stats:
        print(f"  总答案数: {a_stats['count']}")
        print(f"  最短: {a_stats['min']} 字符")
        print(f"  最长: {a_stats['max']} 字符")
        print(f"  平均: {a_stats['mean']:.2f} 字符")
        print(f"  中位数: {a_stats['median']:.2f} 字符")
        print(f"  25%分位数: {a_stats['p25']} 字符")
        print(f"  75%分位数: {a_stats['p75']} 字符")
        print(f"\n  长度区间分布:")
        for bin_label, count in sorted(a_stats['distribution'].items(),
                                       key=lambda x: int(x[0].split('-')[0]) if '-' in x[0] else 9999):
            percentage = (count / a_stats['count']) * 100
            print(f"    {bin_label:12s}: {count:6d} ({percentage:5.2f}%)")
    
    print("\n" + "=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='统计问答对的文档数、问题长度分布、答案长度分布')
    parser.add_argument('--output_dir', type=str, default='output/easy',
                        help='输出目录 (默认: output)')
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.output_dir):
        print(f"错误: 目录不存在: {args.output_dir}")
        return
    
    # 收集统计信息
    stats = collect_statistics(args.output_dir)
    
    if stats is None:
        return
    
    # 打印统计结果
    print_statistics(stats)


if __name__ == "__main__":
    main()
