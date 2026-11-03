import json
import re
import argparse
from pathlib import Path


def extract_answer(text):
    """从文本中提取<answer>标签内的内容"""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_score(response):
    """从模型响应中提取分数"""
    # 尝试多种模式匹配分数
    patterns = [
        r'Score:\s*(\d+(?:\.\d+)?)',
        r'score:\s*(\d+(?:\.\d+)?)',
        r'分数[：:]\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*分',
        r'(\d+(?:\.\d+)?)/100',
        r'评分[：:]\s*(\d+(?:\.\d+)?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            score = float(match.group(1))
            # 确保分数在0-100范围内
            if 0 <= score <= 100:
                return score
    
    # 如果没有找到分数，尝试提取最后一个数字
    numbers = re.findall(r'\d+(?:\.\d+)?', response)
    if numbers:
        score = float(numbers[-1])
        if 0 <= score <= 100:
            return score
    
    return None


def evaluate_translation(agent, src, hyp, src_lang="Chinese", trg_lang="English"):
    """评估单个翻译的GRF分数"""
    prompt = f""""""
    
    try:
        resp = agent.generate(query=prompt)
        response_text = resp.get('response', '')
        score = extract_score(response_text)
        
        if score is None:
            print(f"Warning: Could not extract score from response: {response_text}")
            return None
        
        return score
    
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return None


def process_file(file_path, agent):
    """处理文件并直接写回原文件"""
    
    # 读取所有数据
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Total lines to process: {len(lines)}")
    
    # 解析所有数据
    all_data = []
    for line in lines:
        try:
            data = json.loads(line.strip())
            all_data.append(data)
        except Exception as e:
            print(f"Error parsing line: {str(e)}")
            all_data.append({})
    
    # 检查是否已经有评分，找到需要处理的起始位置
    start_idx = 0
    for idx, data in enumerate(all_data):
        if 'GRF_score' not in data:
            start_idx = idx
            break
    else:
        # 所有数据都已评分
        print("All lines already have GRF scores.")
        return
    
    print(f"Starting from line {start_idx + 1}")
    
    # 处理每一行
    for idx in range(start_idx, len(all_data)):
        data = all_data[idx]
        
        try:
            # 提取源文本
            src = data.get('target', '')
            
            if not src:
                print(f"[{idx+1}] Skipping: no source text")
                data['GRF_score'] = {'adaCOT': None}
                continue
            
            # 提取adaCOT翻译并评分
            adacot_text = extract_answer(data.get('translation_wconfusion', ''))
            print(f"[{idx+1}/{len(all_data)}] Evaluating adaCOT: {adacot_text[:50]}...")
            adacot_score = evaluate_translation(agent, src, adacot_text)
            
            # 添加分数到数据中
            data['GRF_score'] = {
                'adaCOT': adacot_score
            }
            
            print(f"[{idx+1}/{len(all_data)}] Scores - adaCOT: {adacot_score}")
            
        except Exception as e:
            print(f"Error processing line {idx+1}: {str(e)}")
            data['GRF_score'] = {
                'adaCOT': None
            }
        
        # 每处理一条就立即写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            for result in all_data:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"[{idx+1}/{len(all_data)}] Progress saved to file")
    
    print(f"\nProcessing completed. All results saved to {file_path}")
    
    # 统计信息
    valid_adacot = sum(1 for r in all_data if r.get('GRF_score', {}).get('adaCOT') is not None)
    print(f"Valid adaCOT scores: {valid_adacot}/{len(all_data)}")


def process_file_list(file_list_path):
    """从txt文件读取文件列表并处理"""
    
    agent = 
    
    # 读取文件列表
    with open(file_list_path, 'r', encoding='utf-8') as f:
        file_paths = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(file_paths)} files to process\n")
    print("=" * 80)
    
    # 处理每个文件
    for i, file_path in enumerate(file_paths, 1):
        print(f"\n{'='*80}")
        print(f"Processing file {i}/{len(file_paths)}: {file_path}")
        print(f"{'='*80}\n")
        
        # 检查文件是否存在
        if not Path(file_path).exists():
            print(f"ERROR: File not found: {file_path}")
            print(f"Skipping...\n")
            continue
        
        try:
            process_file(file_path, agent)
            print(f"\n✓ Successfully processed: {file_path}")
        except Exception as e:
            print(f"\n✗ Error processing {file_path}: {str(e)}")
            continue
    
    print(f"\n{'='*80}")
    print("All files processed!")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="批量评估翻译质量（GRF分数）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单个文件
  python evaluate_grf.py --file /path/to/file.jsonl
  
  # 批量处理文件（从txt文件读取文件列表）
  python evaluate_grf.py --file-list files.txt
  
files.txt 格式示例:
  /path/to/file1.jsonl
  /path/to/file2.jsonl
  /path/to/file3.jsonl
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--file',
        type=str,
        help='单个JSONL文件路径'
    )
    group.add_argument(
        '--file-list',
        type=str,
        help='包含文件路径列表的txt文件（每行一个文件路径）'
    )
    
    args = parser.parse_args()
    
    if args.file:
        # 处理单个文件
        print(f"Processing single file: {args.file}")
        print("Starting GRF evaluation...\n")
        
        # 初始化agent
        agent = 
        process_file(args.file, agent)
        
    elif args.file_list:
        # 批量处理文件
        print(f"Processing files from list: {args.file_list}")
        print("Starting batch GRF evaluation...\n")
        
        process_file_list(args.file_list)


if __name__ == "__main__":
    main()