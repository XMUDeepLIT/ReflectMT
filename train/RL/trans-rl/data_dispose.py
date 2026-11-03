# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import json
from tqdm import tqdm
import re
import os
import datasets
from datasets import load_dataset

import argparse


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution


def dispose_gsm8k():
    data_source = 'openai/gsm8k'

    dataset = datasets.load_dataset(data_source, 'main')

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    instruction_following = "Let's think step by step and output the final answer after \"####\"."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('question')

            question = question_raw + ' ' + instruction_following

            answer_raw = example.pop('answer')
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer_raw,
                    "question": question_raw,
                }
            }
            return data

        return process_fn

    # print(type(train_dataset))
    # print(len(train_dataset))
    # print(json.dumps(train_dataset[0], ensure_ascii=False, indent=4))
    # print(type(train_dataset[0]))
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)


    train_dataset.to_parquet('')
    test_dataset.to_parquet('')


def dispose_littrans():
    
    train_dataset = load_dataset("json", data_files="")["train"] 
    valid_dataset = load_dataset("json", data_files="")["train"]


    instruction_following = ""
    vllm_url = ""
    
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            src = example.pop('text')
            tgt = example.pop('trans')
            thought = example.pop('thought')
            data = {
                "data_source": "drt_data",
                "prompt": [{
                    "role": "user",
                    "content": instruction_following.replace("{text}", src),
                }],
                "ability": "translation",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": tgt
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'url': vllm_url,
                    'src': src
                }
            }
            return data

        return process_fn

    # print(type(train_dataset))
    # print(len(train_dataset))
    # print(json.dumps(train_dataset[0], ensure_ascii=False, indent=4))
    # print(type(train_dataset[0]))


    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    valid_dataset = valid_dataset.map(function=make_map_fn('valid'), with_indices=True)


    train_dataset.to_parquet('')
    valid_dataset.to_parquet('')






def dispose_s1s_data():
    
    train_dataset = load_dataset("json", data_files="")['']
    valid_dataset = load_dataset("json", data_files="")['']


    vllm_url = ""
    
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            current_src = example.pop('input')
            # current_stats = example.pop('classification').pop('status')
            current_prompt = example.pop('instruction')
            current_query = f""""""
            
            data = {
                "data_source": "con",
                "prompt": [{
                    "role": "system",
                    "content": current_prompt
                },
                {
                    "role": "user",
                    "content": current_query
                }],
                "ability": "con_translation",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ""
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'url': vllm_url,
                    'src': current_src,
                    'stats': 0
                }
            }
            return data

        return process_fn


    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    valid_dataset = valid_dataset.map(function=make_map_fn('valid'), with_indices=True)


    train_dataset.to_parquet('')
    valid_dataset.to_parquet('')


if __name__ == '__main__':
    # dispose_gsm8k()
    
    # dispose_littrans()
    dispose_s1s_data()

    