import json
import re
from openai import OpenAI
from tqdm import tqdm


def extract_scores(response):
    """从模型响应中提取两个分数"""
    # 尝试提取 score1 和 score2
    patterns = [
        r'score1:\s*(\d+(?:\.\d+)?)',
        r'Score1:\s*(\d+(?:\.\d+)?)',
        r'分数1[：:]\s*(\d+(?:\.\d+)?)',
    ]
    
    score1 = None
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            score1 = float(match.group(1))
            if 0 <= score1 <= 100:
                break
            score1 = None
    
    patterns2 = [
        r'score2:\s*(\d+(?:\.\d+)?)',
        r'Score2:\s*(\d+(?:\.\d+)?)',
        r'分数2[：:]\s*(\d+(?:\.\d+)?)',
    ]
    
    score2 = None
    for pattern in patterns2:
        match = re.search(pattern, response)
        if match:
            score2 = float(match.group(1))
            if 0 <= score2 <= 100:
                break
            score2 = None
    
    return score1, score2


def extract_score(response):
    """从模型响应中提取单个分数（用于反思质量评估）"""
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
            if 0 <= score <= 100:
                return score
    
    numbers = re.findall(r'\d+(?:\.\d+)?', response)
    if numbers:
        score = float(numbers[-1])
        if 0 <= score <= 100:
            return score
    
    return None


def obtain_request(messages):
    client = OpenAI(
        api_key="",
        base_url="",
    )
    try_num = 0
    while True:
        try:
            completion = client.chat.completions.create(
                model="",
                messages=messages,
                temperature=0.1,
                top_p=0.8,
                max_tokens=512,
                extra_body={
                    "repetition_penalty": 1.05,
                },
            )
            response = completion.choices[0].message.content
            response = response.strip()
            return response
        except:
            try_num += 1
            if try_num >= 10:
                return None


# ==================== Prompt模板 ====================

PROMPT_DUAL_TRANSLATION_QUALITY = """"""


PROMPT_REFLECTION_QUALITY = """"""


# ==================== 反思相关函数（来自 v1） ====================

def compute_reflection_quality_score(source_sentence, answer_content, reflection_content):
    """
    评估反思质量，返回反思质量得分（0 ~ 1）。
    需要在确认 needs_revision=True 且 answer_changed=True 后才有意义。
    """
    if len(reflection_content) <= 20:
        print("反思内容过短，质量分为0")
        return 0.0

    messages = [{
        "role": "user",
        "content": PROMPT_REFLECTION_QUALITY.format(
            source=source_sentence,
            initial_translation=answer_content,
            reflection=reflection_content,
        )
    }]
    response = obtain_request(messages)
    if not response:
        print(f"LLM调用失败，反思质量给  分")
        return 

    raw = extract_score(response)
    if raw is not None:
        raw = max(0.0, min(100.0, raw))
        score = raw / 100.0
        print(f"反思质量分: {score:.2f} (原始分={raw})")
        return score
    else:
        print(f"无法解析反思质量，给  分")
        return 


def compute_reflection_improvement_score(score_i, score_f):
    """
    评估反思改进奖励，通过对比 answer 与 final_answer 翻译质量分（0~1 尺度）的差来计算。
    与 v1 一致的分段处理：
    - 双正：(score_f - score_i) + 1
    - 否则：score_f - score_i
    最后乘以 0.1。
    """
    if score_f > 0 and score_i > 0:
        score_improvement = (score_f - score_i) + 1
    else:
        score_improvement = score_f - score_i
    return score_improvement * 0.1


# ==================== 反思奖励函数 ====================

def compute_score(
    source_sentence, 
    solution_str_raw, 
    ground_truth=None,
    stats=None,
):
    """
    反思版奖励函数
    
    Args:
        source_sentence: 原文（英文）
        solution_str_raw: 模型生成的完整回复
        ground_truth: 参考翻译（可选）
        stats: 保留参数（兼容性）
    
    Returns:
        float: 总分（0-1分制）
    """
    
    # 权重配置（格式与初始翻译保留原权重，反思部分对齐 v1）
    weights = {
        'format': ,           # 格式正确奖励
        'initial_translation': ,  # 初始翻译质量
    }
    
    total_score = 0.0
    score_details = {}
    
    # ========== 1. 格式检查 ==========
    solution_str = solution_str_raw.split("Assistant:")[-1].strip()
    print(f"\n{'='*50}")
    print(f"原始输出:\n{solution_str[:500]}...")
    
    format_valid = True
    
    # 检查必需标签
    required_tags = ['<answer>', '</answer>', '<reflection>', '</reflection>', 
                     '<need_revision>', '</need_revision>', '<final_answer>', '</final_answer>']
    
    for tag in required_tags:
        if tag not in solution_str:
            print(f"[格式错误] 缺少标签: {tag}")
            format_valid = False
            break
    
    if format_valid:
        # 检查标签数量
        for tag in ['<answer>', '<reflection>', '<need_revision>', '<final_answer>']:
            if solution_str.count(tag) != 1:
                print(f"[格式错误] 标签 {tag} 数量不正确")
                format_valid = False
                break
    
    if format_valid:
        # 检查标签顺序
        answer_start = solution_str.find("<answer>")
        answer_end = solution_str.find("</answer>")
        reflection_start = solution_str.find("<reflection>")
        reflection_end = solution_str.find("</reflection>")
        need_revision_start = solution_str.find("<need_revision>")
        need_revision_end = solution_str.find("</need_revision>")
        final_answer_start = solution_str.find("<final_answer>")
        final_answer_end = solution_str.find("</final_answer>")
        
        if not (answer_start < answer_end < reflection_start < reflection_end < 
                need_revision_start < need_revision_end < final_answer_start < final_answer_end):
            print("[格式错误] 标签顺序不正确")
            format_valid = False
    
    # 提取内容
    if format_valid:
        answer_content = solution_str[answer_start+8:answer_end].strip()
        reflection_content = solution_str[reflection_start+12:reflection_end].strip()
        need_revision_content = solution_str[need_revision_start+15:need_revision_end].strip()
        final_answer_content = solution_str[final_answer_start+14:final_answer_end].strip()
        
        score_details['format'] = weights['format']
        print(f"[格式正确] +{weights['format']:.2f}分")
    else:
        # 容错提取
        answer_content = solution_str[:200]
        reflection_content = ""
        need_revision_content = "未知"
        final_answer_content = solution_str[:200]
        score_details['format'] = 0
        print(f"[格式错误] 0分")
    
    total_score += score_details['format']
    
    # ========== 2. 双翻译质量评分（一次调用获取两个分数）==========
    print("\n[同时评估初始翻译和最终翻译质量]")
    messages_dual = [{
        "role": "user",
        "content": PROMPT_DUAL_TRANSLATION_QUALITY.format(
            source=source_sentence,
            translation1=answer_content,
            translation2=final_answer_content
        )
    }]
    
    response_dual = obtain_request(messages_dual)
    initial_score_raw = 0
    final_score_raw = 0
    
    if response_dual:
        print(f"LLM评分响应: {response_dual}")
        score1, score2 = extract_scores(response_dual)
        
        if score1 is not None:
            initial_score_raw = max(0, min(100, score1))
            initial_translation_score = (initial_score_raw / 100.0) * weights['initial_translation']
            score_details['initial_translation'] = initial_translation_score
            score_details['initial_translation_raw'] = initial_score_raw
            print(f"初始翻译分(score1): {initial_translation_score:.2f} (原始分={initial_score_raw})")
        else:
            print("无法解析score1，给x分")
            score_details['initial_translation'] = 0.1 * weights['initial_translation']
            initial_score_raw = x
        
        if score2 is not None:
            final_score_raw = max(0, min(100, score2))
            print(f"最终翻译分(score2):(原始分={final_score_raw})")
        else:
            print("无法解析score2，给x分")
            final_score_raw = x
    else:
        print("LLM调用失败，两个翻译都给x分")
        score_details['initial_translation'] = 0.1 * weights['initial_translation']
        initial_score_raw = x
        final_score_raw = x
    
    total_score += score_details['initial_translation']
    
    # ========== 3. 反思奖励计算（v1 风格） ==========
    print("\n[评估反思（v1 风格）]")
    
    # 判断是否需要反思
    needs_revision = need_revision_content.strip() in ['是', 'yes', 'Yes', 'YES', '需要', 'true', 'True']
    answer_changed = answer_content.strip() != final_answer_content.strip()
    
    # 将翻译质量分归一化到 0~1，用于 v1 风格的反思打分
    score_i_norm = initial_score_raw / 100.0
    score_f_norm = final_score_raw / 100.0
    
    reflection_score = 0.0
    quality_score = 0.0
    improvement_score = 0.0
    
    if needs_revision and answer_changed:
        # 声明需要反思且确实修改了
        quality_score = compute_reflection_quality_score(
            source_sentence, answer_content, reflection_content
        )
        improvement_score = compute_reflection_improvement_score(score_i_norm, score_f_norm)
        # 若改进分 <= 0（分数下降），则质量分按 v1 折减
        if improvement_score <= 0:
            quality_score = quality_score * y

        reflection_score = quality_score * y
        print(f"[需要反思且已修改] quality={quality_score:.4f}, improvement={improvement_score:.4f}, "
              f"reflection_score={reflection_score:.4f}")
        
    elif not needs_revision and not answer_changed:
        if score_i_norm >= 0.9:
            reflection_score = z
        elif score_i_norm >= 0.6:
            reflection_score = z
        elif score_i_norm > 0:
            reflection_score = 0.0
        else:
            reflection_score = z
    else:
        # 其他情况（判断错误或不一致）
        reflection_score = z
    
    score_details['reflection_total'] = reflection_score
    score_details['reflection_quality'] = quality_score
    score_details['reflection_improvement'] = improvement_score
    
    total_score += reflection_score * 100.0
    
    # ========== 输出详细信息 ==========
    print(f"\n{'='*50}")
    print(f"总分: {total_score:.2f}/100")
    print(f"\n详细得分:")
    print(f"  - 格式分: {score_details.get('format', 0):.2f}")
    print(f"  - 初始翻译: {score_details.get('initial_translation', 0):.2f} (原始={score_details.get('initial_translation_raw', 0)})")
    print(f"  - 最终翻译raw: {final_score_raw}")
    print(f"  - 反思质量(0~1): {score_details.get('reflection_quality', 0):.4f}")
    print(f"  - 反思改进(0~1): {score_details.get('reflection_improvement', 0):.4f}")
    print(f"  - 反思总分(0~1): {score_details.get('reflection_total', 0):.4f}")
    print(f"\n需要反思: {needs_revision}")
    print(f"翻译改变: {answer_changed}")
    print(f"分数变化: {final_score_raw - initial_score_raw:.1f}")
    print(f"{'='*50}\n")
    
    return total_score / 100  # 返回0-1分制