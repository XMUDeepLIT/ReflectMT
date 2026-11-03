import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", "-i", 
                        default="",
                        help="Input JSONL file path")
    parser.add_argument("--output", "-o",
                        default="",
                        help="Output JSONL file path")
    parser.add_argument("--model_path", "-m",
                        default="",
                        help="Model path")
    parser.add_argument("--max_new_tokens", type=int, default=8196,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for processing")
    return parser


def load_model(model_path):
    """加载模型和tokenizer"""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("Model loaded successfully!")
    return tokenizer, model

def create_translation_messages_with_confusion(source_text):
    messages = [
        {
            "role": "system",
            "content": """"""
        },
        {
            "role": "user",
            "content": f""
        }
    ]
    return messages

def translate_with_model(tokenizer, model, messages, max_new_tokens=512):
    """使用模型进行翻译"""
    messages = messages
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 使用贪婪解码以保证一致性
            temperature=1.0,
            top_p=1.0
        )
    
    # 只解码新生成的部分
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], 
        skip_special_tokens=True
    )
    aa = tokenizer.decode(
        outputs[0], 
        skip_special_tokens=False
    )
    
    return generated_text.strip()


def process_jsonl(input_file, output_file, tokenizer, model, max_new_tokens=512):
    """处理JSONL文件"""
    
    # 统计信息
    total = 0
    success = 0
    errors = 0
    
    # 读取输入文件并计算总行数
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"\nProcessing {len(lines)} samples...")
    
    with open(output_file, 'w', encoding='utf-8') as fout:
        for line_num, line in enumerate(tqdm(lines, desc="Translating"), 1):
            try:
                data = json.loads(line.strip())
                
                # 获取源文本
                source_text = data.get('src')
                
                if not source_text:
                    print(f"\nWarning: No source text found in line {line_num}")
                    errors += 1
                    continue
                
                
                messages_wconcot = create_translation_messages_with_confusion(source_text)
                translation_wconcot = translate_with_model(
                    tokenizer, model, messages_wconcot, max_new_tokens
                )
                print("confusion cot", translation_wconcot)
                
                
                data['translation_wconfusion'] = translation_wconcot
                
                # 写入输出文件
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                
                total += 1
                success += 1
                
                # 每10个样本打印一次示例
                if line_num % 10 == 0:
                    print(f"\n{'='*80}")
                    print(f"Sample {line_num}:")
                    print(f"Source: {source_text[:100]}...")
                    # print(f"w/oCOT: {translation_wocot[:100]}...")
                    # print(f"w/COT:  {translation_wcot[:100]}...")
                    print(f"conCOT: {translation_wconcot[:100]}...")
                    print(f"{'='*80}\n")
                
            except Exception as e:
                print(f"\nError processing line {line_num}: {str(e)}")
                errors += 1
                continue
    
    # 打印统计信息
    print("\n" + "="*80)
    print("Translation Complete!")
    print("="*80)
    print(f"Total samples: {total}")
    print(f"Successful: {success}")
    print(f"Errors: {errors}")
    print(f"Output saved to: {output_file}")
    print("="*80)


def main():
    args = get_parser().parse_args()
    
    # 加载模型
    tokenizer, model = load_model(args.model_path)
    
    # 处理文件
    process_jsonl(
        args.input,
        args.output,
        tokenizer,
        model,
        args.max_new_tokens
    )


if __name__ == "__main__":
    main()