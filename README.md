# PDF问答对生成器

使用LLM API从PDF文件自动生成多层次问答对，支持LaTeX公式提取，适用于知识库构建和模型微调。

## 功能特点

- 📄 **PDF文本提取**：支持从PDF文件提取文本内容，支持LaTeX公式OCR识别
- 🤖 **智能问答生成**：基于LLM自动生成高质量的问答对
- 🎯 **双模式支持**：
  - `normal`模式：生成大量较短的常见基础知识问答对
  - `pro`模式：生成少量但长篇的深入研讨问答对
- ⚡ **并行处理**：支持多文件并行处理和答案并行生成
- 💾 **自动保存**：每个PDF处理完成后自动保存为JSON文件

## 安装

1. 克隆或下载项目
2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 环境配置

创建`.env`文件（或在系统环境变量中设置）：

```env
API_KEY=your_api_key_here
BASE_URL=https://api.chatanywhere.tech/v1  # 可选，默认值
MODEL_NAME=llm-chat  # 可选，默认值
```

## 使用方法

### 1. 从PDF文件生成问答对

```bash
python main.py --pdf_dir pdf_files --output_dir output --mode normal --num_qa 15
```

### 2. 从txt文件生成问答对（推荐，更快）

```bash
# 第一步：提取PDF为txt文件
python main.py --extract-only --pdf_dir pdf_files --txt_dir txt_dir

# 第二步：从txt文件生成问答对
python main.py --from-txt --txt_dir txt_dir --output_dir output --mode normal --num_qa 15
```

### 3. 使用pro模式生成专业问答对

```bash
python main.py --from-txt --txt_dir txt_dir --output_dir output --mode pro --num_qa 10
```

## 输入输出位置

### 输入位置

- **PDF文件目录**：`--pdf_dir` 参数指定（默认：`pdf_files`）
- **txt文件目录**：`--txt_dir` 参数指定（默认：`./txt_dir`）

### 输出位置

- **输出目录**：`--output_dir` 参数指定（默认：`output`）
- **JSON文件路径**：`{output_dir}/{mode}/{文件名}_{时间戳}_{微秒}_{计数器}.json`
  - 例如：`output/normal/论文标题_20251113_194540_121262_0001.json`

### 输出文件格式

每个JSON文件包含：
```json
{
  "source": "原始PDF文件名.pdf",
  "metadata": {...},
  "qa_pairs": [
    {
      "question": "问题内容",
      "answer": "答案内容"
    }
  ],
  "generated_at": "20251113_194540"
}
```

## 参数说明

### 基本参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--pdf_dir` | str | `pdf_files` | PDF文件所在目录 |
| `--txt_dir` | str | `./txt_dir` | txt文件目录 |
| `--output_dir` | str | `output` | 输出目录 |
| `--mode` | str | `normal` | 生成模式：`normal` 或 `pro` |
| `--num_qa` | int | `10` | 每个PDF生成的问答对数量 |

### 性能参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--max_workers` | int | `5` | 最大并行处理的文件数 |
| `--answer_workers` | int | `15` | 答案生成的并行线程数（仅pro模式） |
| `--api_retries` | int | `5` | API调用失败时的最大重试次数 |
| `--retry_delay` | int | `4` | API重试间隔时间（秒） |

### 功能参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_latex_ocr` | flag | `True` | 启用LaTeX公式OCR识别 |
| `--model` | str | 环境变量 | 指定LLM模型名称 |
| `--extract-only` | flag | `False` | 仅提取PDF文本为txt，不生成问答对 |
| `--from-txt` | flag | `True` | 从txt文件生成问答对（而非PDF） |

## 参数设置建议

### 正常模式（normal）

- **问答对数量**：`--num_qa 15-30`（根据文档长度调整）
- **并行文件数**：`--max_workers 5-10`（根据API限制和机器性能）
- **适用场景**：快速生成大量基础知识问答对

### 专业模式（pro）

- **问答对数量**：`--num_qa 5-15`（生成更详细的答案）
- **并行文件数**：`--max_workers 3-5`（避免API限流）
- **答案并行数**：`--answer_workers 10-20`（加速答案生成）
- **适用场景**：生成深入的专业问答对

### API配置建议

- **重试次数**：`--api_retries 5`（网络不稳定时增加）
- **重试间隔**：`--retry_delay 4`（API限流时增加）
- **模型选择**：根据需求选择合适的LLM模型

## 使用示例

### 示例1：批量处理PDF文件（normal模式）

```bash
python main.py \
  --pdf_dir ./pdf_files \
  --output_dir ./output \
  --mode normal \
  --num_qa 20 \
  --max_workers 5 \
  --answer_workers 15
```

### 示例2：从txt文件生成专业问答对

```bash
python main.py \
  --from-txt \
  --txt_dir ./txt_dir \
  --output_dir ./output \
  --mode pro \
  --num_qa 10 \
  --max_workers 3 \
  --answer_workers 20 \
  --api_retries 5
```

### 示例3：仅提取PDF文本

```bash
python main.py \
  --extract-only \
  --pdf_dir ./pdf_files \
  --txt_dir ./txt_dir \
  --max_workers 5
```

## 日志文件

程序运行时会生成日志文件：
- `pdf_qa_generator.log`：主程序运行日志

## 注意事项

1. **API密钥**：确保正确设置`API_KEY`环境变量
2. **文件格式**：确保PDF文件可正常读取
3. **目录权限**：确保对输入输出目录有读写权限
4. **API限流**：注意API调用频率限制，适当调整并行参数
5. **模式选择**：
   - `normal`模式：一次性生成所有问答对，速度快
   - `pro`模式：分两步生成（先问题后答案），质量高但较慢

## 故障处理

- **API调用失败**：检查网络连接和API密钥，增加重试次数
- **文件处理失败**：查看日志文件`pdf_qa_generator.log`，失败文件列表保存在`output/failed_files.txt`
- **内存不足**：减少`--max_workers`和`--answer_workers`参数值

