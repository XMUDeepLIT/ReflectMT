import json
import re


# 
DEEPSEEK_CLIENT = None  
DEEPSEEK_MODEL = ""
REFLECTION_MERGE_PROMPT = ""


def extract_reflection_tags(output, status):
    """
    从输出中提取标签内容，根据status决定提取哪些内容
    
    Args:
        output: 包含标签的原始输出文本
        status: 分类状态 (0, 1, 2, 3等)
        
    Returns:
        str: 根据status返回相应的内容
    """
    
    # 提取 <answer> 标签内容
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, output, re.DOTALL)
    answer_content = answer_match.group(1).strip() if answer_match else ''

    # 提取 <reflection> 标签内容
    reflection_pattern = r'<reflection>(.*?)</reflection>'
    reflection_match = re.search(reflection_pattern, output, re.DOTALL)
    reflection_content = reflection_match.group(1).strip() if reflection_match else ''
    
    # 提取 <need_revision> 标签内容
    need_revision_pattern = r'<need_revision>(.*?)</need_revision>'
    need_revision_match = re.search(need_revision_pattern, output, re.DOTALL)
    need_revision_content = need_revision_match.group(1).strip() if need_revision_match else ''
    
    # 提取 <final_answer> 标签内容
    final_answer_pattern = r'<final_answer>(.*?)</final_answer>'
    final_answer_match = re.search(final_answer_pattern, output, re.DOTALL)
    final_answer_content = final_answer_match.group(1).strip() if final_answer_match else ''
    
    # 组合完整结果
    result = f'<answer>{answer_content}</answer>\n<reflection>{reflection_content}</reflection>\n<need_revision>{need_revision_content}</need_revision>\n<final_answer>{final_answer_content}</final_answer>'
    
    return result


def merge_reflections_with_deepseek(
    reflections, initial_translation, final_translation, *,
    client=None, model=None, prompt=None,
):
    """调用 DeepSeek，将全部轮次的反思合并为针对初始译文的优化指令。

    Args:
        reflections: 按 history 顺序提取的所有轮次的 reflection 列表。
        initial_translation: 第一轮翻译。
        final_translation: 最后一轮翻译。
        client: 已初始化的客户端；未传时使用 DEEPSEEK_CLIENT。
        model: 模型名称；未传时使用 DEEPSEEK_MODEL。
        prompt: 合并任务的提示词；未传时使用 REFLECTION_MERGE_PROMPT。

    Returns:
        str: 指导模型从初始翻译优化到最终翻译的 prompt，写入 <reflection>。
    """
    client = DEEPSEEK_CLIENT if client is None else client
    model = DEEPSEEK_MODEL if model is None else model
    prompt = REFLECTION_MERGE_PROMPT if prompt is None else prompt

    # 保留每轮原文和轮次顺序，不截断、不只选最后一轮，也不去掉重复轮次。
    payload = {
        "reflections": [
            {"round": index, "reflection": reflection}
            for index, reflection in enumerate(reflections, 1)
        ],
        "initial_translation": initial_translation,
        "final_translation": final_translation,
    }
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
        )
    except Exception as exc:
        raise RuntimeError("DeepSeek 合并 reflection 的请求失败") from exc

    if not response.choices:
        raise RuntimeError("DeepSeek 未返回任何结果")
    choice = response.choices[0]
    if choice.finish_reason != "stop":
        raise RuntimeError(f"DeepSeek 未正常完成生成：{choice.finish_reason}")
    content = choice.message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("DeepSeek 返回的合并 reflection 为空")

    merged_reflection = content.strip()
    # 输出文件会统一添加标签；若模型返回了外层 reflection 标签，先去掉。
    match = re.fullmatch(r'<reflection>\s*(.*?)\s*</reflection>', merged_reflection, re.DOTALL)
    if match:
        merged_reflection = match.group(1).strip()
    if not merged_reflection:
        raise RuntimeError("DeepSeek 返回的 reflection 标签中没有内容")
    return merged_reflection


def convert_multiround_to_reflection_format(input_jsonl, output_json):
    """
    将多轮翻译数据转换为反思格式
    
    Args:
        input_jsonl: 输入的jsonl文件路径
        output_json: 输出的json文件路径
    """
    
    instruction = """"""
    
    data = []
    skipped_count = 0
    stats_count = {}
    round_count = {1: 0, 'multi': 0}
    
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                
                # 获取必要字段
                src = item.get('src', '')
                history = item.get('history', [])
                stats = item.get('stats', None)
                
                # 统计stats
                if stats is not None:
                    stats_count[stats] = stats_count.get(stats, 0) + 1
                
                # 检查数据完整性
                if not src:
                    print(f"行 {line_num}: 缺少 src 字段，跳过")
                    skipped_count += 1
                    continue
                
                if not history or len(history) == 0:
                    print(f"行 {line_num}: history 为空，跳过")
                    skipped_count += 1
                    continue
                
                # 以实际 history 长度为准，避免 total_rounds 不一致时漏轮或越界。
                total_rounds = len(history)
                if total_rounds == 1:
                    # 单轮情况
                    round_count[1] += 1
                    
                    # 获取第一轮的翻译和反思
                    first_round = history[0]
                    translation = first_round.get('translation', '')
                    reflection = first_round.get('reflection', '')
                    
                    # 构建输出
                    answer_content = translation
                    reflection_content = reflection
                    need_revision_content = '否'
                    final_answer_content = translation
                    
                else:
                    # 多轮情况 (total_rounds > 1)
                    round_count['multi'] += 1
                    
                    # answer: 第一轮的翻译
                    answer_content = history[0].get('translation', '')
                    
                    # 提取所有轮次的 reflection，保留原始顺序（包括最后一轮）。
                    reflections = [
                        round_item.get('reflection', '')
                        for round_item in history
                    ]

                    # final_answer: 最后一轮的翻译
                    final_answer_content = history[-1].get('translation', '')

                    # 将全部反思、初始翻译和最终翻译交给 DeepSeek 合并。
                    reflection_content = merge_reflections_with_deepseek(
                        reflections=reflections,
                        initial_translation=answer_content,
                        final_translation=final_answer_content,
                    )
                    
                    # need_revision: 是
                    need_revision_content = '是'
                    
                # 组合输出
                output = f'<answer>{answer_content}</answer>\n<reflection>{reflection_content}</reflection>\n<need_revision>{need_revision_content}</need_revision>\n<final_answer>{final_answer_content}</final_answer>'
                
                # 添加到数据列表
                data.append({
                    "instruction": instruction,
                    "input": src,
                    "output": output
                })
                
            except RuntimeError:
                # 配置缺失或合并失败时明确报错，不能静默丢弃多轮记录。
                raise
            except Exception as e:
                print(f"行 {line_num}: 处理记录时出错: {e}")
                skipped_count += 1
                continue
    
    # 保存结果
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print("\n" + "="*60)
    print("转换完成!")
    print("="*60)
    print(f"总共处理: {len(data)} 条数据")
    print(f"跳过记录: {skipped_count} 条")
    
    print("\n轮次统计:")
    print(f"  单轮 (total_rounds=1):  {round_count[1]} 条")
    print(f"  多轮 (total_rounds>1):  {round_count['multi']} 条")
    
    if stats_count:
        print("\nstats 分布:")
        for stat_value in sorted(stats_count.keys()):
            print(f"  stats={stat_value}: {stats_count[stat_value]} 条")
    
    print(f"\n已保存到: {output_json}")
    print("="*60 + "\n")
    
    return data


def verify_conversion(output_json, num_samples=3):
    """
    验证转换结果，显示几个样本
    
    Args:
        output_json: 输出的json文件路径
        num_samples: 显示的样本数量
    """
    
    print("\n" + "="*60)
    print("验证转换结果 - 显示样本")
    print("="*60 + "\n")
    
    with open(output_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for i, item in enumerate(data[:num_samples], 1):
        print(f"样本 {i}:")
        print(f"Input: {item['input'][:100]}...")
        print(f"\nOutput:")
        print(item['output'])
        print("\n" + "-"*60 + "\n")


if __name__ == "__main__":
    # 输入输出路径
    input_file = ""
    output_file = ""
    
    # 执行转换
    result = convert_multiround_to_reflection_format(input_file, output_file)
    
    # 验证结果
    if result:
        verify_conversion(output_file, num_samples=3)
