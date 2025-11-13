import json
import importlib.util
import os

# 确保导入本地 stat.py 文件而不是标准库的 stat 模块
stat_path = os.path.join(os.path.dirname(__file__), 'stat.py')
spec = importlib.util.spec_from_file_location("stat_module", stat_path)
stat_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stat_module)
load_json_files = stat_module.load_json_files



def process_question(question):
    return question


def process_answer(answer):
    # 如果ans以"好的"开头，则删除第一句话，即删除到第一个句号“。”
    if answer.startswith("好的"):
        # 删除第一句话（即删除第一个“。”及其之前的所有内容）
        if "。" in answer:
            answer = answer[answer.find("。")+1:]
        # 删除开头所有的换行符
        answer = answer.lstrip("\n")
        # 删除开头所有的空格
        answer = answer.lstrip(" ")
        # 删除开头的“---”
        answer = answer.lstrip("\n---\n")
        print(f"删除了“好的”话术: {answer[:10]}...")

    while ("文献" in answer[:100] or "严格基于" in answer[:100] or "提供" in answer[:100] or "以下" in answer[:100]) and not answer.startswith("###"):
        
        # 删除第一句话，第一句话可能以“：”、“。”结尾
        idx = answer.find("：")
        if idx != -1:
            answer = answer[idx+1:]
        else:
            idx = answer.find("。")
            if idx != -1:
                answer = answer[idx+1:]
            else:
                break

        # 删除开头所有的换行符
        answer = answer.lstrip("\n")
        # 删除开头所有的空格
        answer = answer.lstrip(" ")
        # 删除开头的“---”
        answer = answer.lstrip("\n---\n")
        print(f"删除了“文献”、“严格基于”、“提供”等话术: {answer[:10]}...")

    if answer.startswith("### **问题：") or  answer.startswith("问题：") or answer.startswith("### 问题："):
        idx = answer.find("\n\n")
        if idx != -1:
            answer = answer[idx+1:]
            # 删除开头所有的换行符
            answer = answer.lstrip("\n")
            # 删除开头所有的空格
            answer = answer.lstrip(" ")
            # 删除开头的“---”
            answer = answer.lstrip("\n---\n")
            print(f"删除了“问题：”话术: {answer[:10]}...")
    return answer


def post_process(qa_pairs):
    new_qa_pairs = []
    for qa_pair in qa_pairs:
        qa_pair['question'] = process_question(qa_pair['question'])
        qa_pair['answer'] = process_answer(qa_pair['answer'])
        new_qa_pairs.append(qa_pair)

    return new_qa_pairs


def main():
    qa_dir = "./output/pro"
    json_files = load_json_files(qa_dir)

    for json_file in json_files:
        json_data = {}
        new_qa_pairs = []
        with open(json_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            try:
                new_qa_pairs = post_process(json_data['qa_pairs'])
            except Exception as e:
                print(f"处理文件 {json_file} 时出错: {e}")
                exit(1)
        json_data['qa_pairs'] = new_qa_pairs
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()