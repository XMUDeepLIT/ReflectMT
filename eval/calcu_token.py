import json
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import sys

# 配置
MODEL_PATH = ""
CONFIG_FILE = ""  # 配置文件路径

def get_text_from_data(data):
    """
    根据数据格式提取需要计算的文本
    优先级: full_response > translation_wconfusion
    """
    if "full_response" in data and data["full_response"]:
        return data["full_response"]
    elif "translation_wconfusion" in data and data["translation_wconfusion"]:
        return data["translation_wconfusion"]
    elif "answer1" in data and data["answer1"]:
        return data["answer1"]
    else:
        if "translation_wadacot" in data:
            return data["translation_wadacot"]
        return ""

def process_file(file_path, tokenizer):
    """
    处理单个文件，返回统计结果
    """
    if not os.path.exists(file_path):
        print(f"⚠️  文件不存在: {file_path}")
        return None
    
    total_tokens = 0
    count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"处理 {os.path.basename(file_path)}"):
                try:
                    data = json.loads(line)
                    text = get_text_from_data(data)
                    
                    if text:
                        tokens = tokenizer.encode(text, add_special_tokens=False)
                        total_tokens += len(tokens)
                        count += 1
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON解析错误: {e}")
                    continue
        
        return {
            'file_path': file_path,
            'total_tokens': total_tokens,
            'count': count,
            'avg_tokens': total_tokens / count if count > 0 else 0
        }
    
    except Exception as e:
        print(f"❌ 处理文件时出错 {file_path}: {e}")
        return None

def main():
    # 检查配置文件是否存在
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ 配置文件不存在: {CONFIG_FILE}")
        print(f"请创建 {CONFIG_FILE} 文件，每行一个待测试的文件路径")
        sys.exit(1)
    
    # 读取待测试文件列表
    print(f"📖 读取配置文件: {CONFIG_FILE}")
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        file_paths = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    if not file_paths:
        print("❌ 配置文件为空，请添加待测试的文件路径")
        sys.exit(1)
    
    print(f"✅ 找到 {len(file_paths)} 个待测试文件\n")
    
    # 加载tokenizer
    print("🔄 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("✅ Tokenizer加载完成\n")
    
    # 处理所有文件
    results = []
    for file_path in file_paths:
        print(f"\n{'='*80}")
        print(f"📁 处理文件: {file_path}")
        print(f"{'='*80}")
        
        result = process_file(file_path, tokenizer)
        if result:
            results.append(result)
    
    # 输出汇总结果
    print(f"\n\n{'='*80}")
    print("📊 统计结果汇总")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. 文件: {result['file_path']}")
        print(f"   总Token数: {result['total_tokens']:,}")
        print(f"   总行数: {result['count']:,}")
        print(f"   平均Token数: {result['avg_tokens']:.2f}")
        print()
    
    # 总体统计
    if results:
        total_all_tokens = sum(r['total_tokens'] for r in results)
        total_all_lines = sum(r['count'] for r in results)
        
        print(f"{'='*80}")
        print("🎯 总体统计")
        print(f"{'='*80}")
        print(f"处理文件数: {len(results)}")
        print(f"总Token数: {total_all_tokens:,}")
        print(f"总行数: {total_all_lines:,}")
        print(f"总体平均Token数: {total_all_tokens/total_all_lines:.2f}" if total_all_lines > 0 else "N/A")
    
    # 保存结果到文件
    output_file = "token_statistics_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_files': len(results),
                'total_tokens': sum(r['total_tokens'] for r in results),
                'total_lines': sum(r['count'] for r in results),
                'overall_avg_tokens': sum(r['total_tokens'] for r in results) / sum(r['count'] for r in results) if sum(r['count'] for r in results) > 0 else 0
            },
            'details': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 详细结果已保存到: {output_file}")

if __name__ == "__main__":
    main()

