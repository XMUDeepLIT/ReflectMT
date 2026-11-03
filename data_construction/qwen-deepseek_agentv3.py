import os
import json
import logging
from openai import OpenAI
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, Lock
import re
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============ 配置变量 ============
QWEN_URL = ""
DEEPSEEK_URL = ""
INPUT_FILE = ""  # 修改为你的输入文件路径
OUTPUT_FILE = ""  # 修改为你的输出文件路径
MAX_ROUNDS = 5
SCORE_THRESHOLD = 90
MAX_RETRIES = 10
qwen_model = ""

# 多个vllm服务地址（用于多进程并发负载均衡）
qwen_vllm_urls = [
    "",
]
deepseek_vllm_urls = [
    "",
]
# ==================================


def get_parser():
    parser = argparse.ArgumentParser(description="Multi-round translation with reflection")
    parser.add_argument("--workers", "-w", default=32, type=int,
                        help="number of parallel workers")
    return parser


def init_worker():
    """初始化进程的锁"""
    global write_lock
    write_lock = Lock()


args = get_parser().parse_args()


# def obtain_qwen_request(messages, vllm_url, max_retries=MAX_RETRIES):
#     """调用Qwen模型进行翻译"""
#     client = OpenAI(
#         api_key="EMPTY",
#         base_url=vllm_url,
#     )

#     for try_num in range(max_retries):
#         try:
#             completion = client.chat.completions.create(
#                 model="",  # 根据实际模型名称修改
#                 messages=messages,
#                 temperature=0.7,
#                 top_p=0.8,
#                 max_tokens=1024,
#                 extra_body={
#                     "repetition_penalty": 1.05,
#                 },
#             )
#             response = completion.choices[0].message.content
#             return response.strip()
#         except Exception as e:
#             logging.warning(f"Qwen request failed (attempt {try_num + 1}/{max_retries}): {str(e)}")
#             if try_num >= max_retries - 1:
#                 return None
#     return None


def obtain_deepseek_request(messages, vllm_url, max_retries=MAX_RETRIES):
    """调用DeepSeek模型进行反思"""
    client = OpenAI(
        api_key="EMPTY",
        base_url=vllm_url,
    )

    for try_num in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="",
                messages=messages,
                temperature=0.1,
                top_p=0.8,
                max_tokens=2048,
                extra_body={
                    "repetition_penalty": 1.05,
                },
            )
            response = completion.choices[0].message.content
            return response.strip()
        except Exception as e:
            logging.warning(f"DeepSeek request failed (attempt {try_num + 1}/{max_retries}): {str(e)}")
            if try_num >= max_retries - 1:
                return None
    return None


def extract_answer(text):
    """提取<answer>标签中的内容"""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_reflection_and_score(text):
    """提取<reflect>和<score>标签中的内容"""
    reflect_match = re.search(r'<reflect>(.*?)</reflect>', text, re.DOTALL)
    score_match = re.search(r'<score>(\d+)</score>', text, re.DOTALL)

    reflection = reflect_match.group(1).strip() if reflect_match else ""
    score = int(score_match.group(1)) if score_match else 0

    return reflection, score


def initial_translate(src_text, qwen_url):
    """第一轮翻译"""
    prompt = '''''' # 请输入你的翻译prompt

    msgs = [{"role": "user", "content": prompt}]
    response = obtain_deepseek_request(msgs, qwen_url)

    if response is None:
        return None

    translation = extract_answer(response)
    logging.info(f"Initial translation: {translation[:100]}...")
    return translation


def reflect_and_score(src_text, translation, deepseek_url, previous_reflection=None):
    """对翻译进行反思和评分"""
    if previous_reflection:
        prompt = ''''''# 请输入你的首轮反思prompt
    else:
        prompt = ''''''# 请输入你的非首轮反思prompt

    msgs = [{"role": "user", "content": prompt}]
    response = obtain_deepseek_request(msgs, deepseek_url)

    if response is None:
        return None, 0

    reflection, score = extract_reflection_and_score(response)
    logging.info(f"Reflection score: {score}")
    return reflection, score


def refine_translation(src_text, previous_translation, reflection, qwen_url):
    """根据反思意见改进翻译"""
    prompt = '''''' # 请输入你的精炼prompt

    msgs = [{"role": "user", "content": prompt}]
    response = obtain_deepseek_request(msgs, qwen_url)

    if response is None:
        return previous_translation

    new_translation = extract_answer(response)
    logging.info(f"Refined translation: {new_translation[:100]}...")
    return new_translation


def multi_round_translate_with_reflection(data, qwen_url, deepseek_url):
    """多轮翻译和反思"""
    src_text = data['src']
    history = []

    # 第一轮翻译
    translation = initial_translate(src_text, qwen_url)
    if translation is None:
        return None

    # 第一轮反思
    reflection, score = reflect_and_score(src_text, translation, deepseek_url)
    if reflection is None:
        return None

    history.append({
        'round': 1,
        'translation': translation,
        'reflection': reflection,
        'score': score
    })

    logging.info(f"Round 1 - Score: {score}")

    # 如果第一轮就达到阈值，直接返回
    if score >= SCORE_THRESHOLD:
        logging.info(f"Score {score} reached threshold {SCORE_THRESHOLD} in round 1, stopping.")
        return {
            'src': src_text,
            'final_translation': translation,
            'total_rounds': 1,
            'final_score': score,
            'history': history
        }

    # 多轮迭代
    previous_score = score
    current_translation = translation

    for round_num in range(2, MAX_ROUNDS + 1):
        # 根据反思改进翻译
        new_translation = refine_translation(
            src_text,
            current_translation,
            history[-1]['reflection'],
            qwen_url
        )

        # 对新翻译进行反思
        reflection, score = reflect_and_score(
            src_text,
            new_translation,
            deepseek_url,
            previous_reflection=history[-1]['reflection']
        )

        if reflection is None:
            break

        history.append({
            'round': round_num,
            'translation': new_translation,
            'reflection': reflection,
            'score': score
        })

        logging.info(f"Round {round_num} - Score: {score} (previous: {previous_score})")

        # 判断是否停止
        if score >= SCORE_THRESHOLD:
            logging.info(f"Score {score} reached threshold {SCORE_THRESHOLD}, stopping.")
            current_translation = new_translation
            break

        if score > previous_score:
            # 分数上涨，继续
            current_translation = new_translation
            previous_score = score
        else:
            # 分数没有上涨，继续尝试但不更新best translation
            logging.info(f"Score did not improve ({score} <= {previous_score}), continuing...")
            # 仍然更新当前翻译以便下一轮改进
            current_translation = new_translation
            previous_score = score

    # 找出最高分的翻译
    best_round = max(history, key=lambda x: x['score'])

    return {
        'src': src_text,
        'final_translation': best_round['translation'],
        'total_rounds': len(history),
        'final_score': best_round['score'],
        'best_round': best_round['round'],
        'history': history
    }


def process_single_item(data, fw, qwen_urls, deepseek_urls):
    """处理单条数据"""
    qwen_url = qwen_urls[data['id'] % len(qwen_urls)]
    deepseek_url = deepseek_urls[data['id'] % len(deepseek_urls)]
    try:
        result = multi_round_translate_with_reflection(data, qwen_url, deepseek_url)

        if result is None:
            logging.error(f"Failed to process: {data.get('src', '')[:50]}...")
            result = {
                'src': data.get('src', ''),
                'error': 'Processing failed',
                'final_translation': '',
                'total_rounds': 0,
                'final_score': 0,
                'history': []
            }

        # 合并原始数据
        output_data = {**data, **result}

    except Exception as e:
        logging.error(f"Error processing data - {str(e)}")
        output_data = {
            **data,
            'error': str(e),
            'final_translation': '',
            'total_rounds': 0,
            'final_score': 0,
            'history': []
        }

    # 写入文件（多进程加锁，单进程直接写）
    if args.workers > 1:
        with write_lock:
            with open(fw, 'a', encoding='utf-8') as f:
                output_line = json.dumps(output_data, ensure_ascii=False) + '\n'
                f.write(output_line)
                f.flush()
    else:
        with open(fw, 'a', encoding='utf-8') as f:
            output_line = json.dumps(output_data, ensure_ascii=False) + '\n'
            f.write(output_line)
            f.flush()

    return output_data


def process_file():
    """处理输入文件"""
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # 检查已处理的行数
    already_count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as fr:
            already_count = sum(1 for _ in fr)

    fw = f"{OUTPUT_FILE}"
    logging.info(f"Skip {already_count} lines as already been generated")

    items = []
    for index, line in enumerate(lines):
        if index < already_count:
            continue

        data = json.loads(line.strip())
        data['id'] = index
        items.append(data)

    process_func = partial(
        process_single_item,
        fw=fw,
        qwen_urls=qwen_vllm_urls,
        deepseek_urls=deepseek_vllm_urls,
    )

    if args.workers > 1:
        with Pool(processes=args.workers, initializer=init_worker) as pool:
            list(tqdm(
                pool.imap(process_func, items),
                total=len(items),
                desc="Processing items"
            ))
    else:
        for data in tqdm(items, desc="Processing items"):
            process_func(data)


def main():
    logging.info(f"Qwen URL: {QWEN_URL}")
    logging.info(f"DeepSeek URL: {DEEPSEEK_URL}")
    logging.info(f"Qwen vLLM URLs: {qwen_vllm_urls}")
    logging.info(f"DeepSeek vLLM URLs: {deepseek_vllm_urls}")
    logging.info(f"Workers: {args.workers}")
    logging.info(f"Max rounds: {MAX_ROUNDS}")
    logging.info(f"Score threshold: {SCORE_THRESHOLD}")
    logging.info(f"Input file: {INPUT_FILE}")
    logging.info(f"Output file: {OUTPUT_FILE}")
    logging.info("Starting processing...")

    process_file()

    logging.info("Processing completed.")


if __name__ == "__main__":
    main()