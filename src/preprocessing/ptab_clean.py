from glob import glob
import os
import sys
from pathlib import Path

import argparse
import json
import re
import ast

import pdb
import pandas as pd

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import asyncio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config_utils import load_config

def load_valid_identifier(valid_path):
    valid_identifier = []
    with open(valid_path) as f:
        for i, line in enumerate(f, start=1):
            try:
                valid_identifier.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[오류] {i}번째 {line} 파일 파싱 실패: {e}")
    return valid_identifier

def split_by_before_blocks(pages):
    def fuzzy(word):
        return r'\s*'.join(list(word)) + r'\s*'

    pattern = re.compile(
        r'BEFORE\s*(?:THE\s*)?'
        r'(?:' +
            fuzzy('PATENT') + r'(?:AND\s*)?' +
            fuzzy('TRIAL') + r'(?:AND\s*)?' +
            fuzzy('APPEAL') + r'S?\s*' + fuzzy('BOARD') +
            r'|' +
            r'BOARD\s*OF\s*' + fuzzy('PATENT') + r'\s*APPEALS(?:\s*AND(?:\s*INTERFERENCES)?)?' +
        r')',
        re.IGNORECASE
    )

    filtered_data = {}
    status = False
    page_keys = sorted(pages.keys(), key=int)

    prev_key = [
        i for i, key in enumerate(page_keys)
        if re.search(pattern, pages[key][:500])
    ]

    if len(prev_key) > 1:
        # If there are two or more info pages
        filtered_data = None
    elif len(prev_key) == 1:
        status = True
        for x, key in enumerate(pages):
            if prev_key[0] <= x: # drop an info page
                filtered_data[key] = pages[key]
    else:
        filtered_data = None

    return status, filtered_data, prev_key

def drop_redundant_header(data, num_lines=5):
    def find_common_headers(data, num_lines):
        buffer_dict = {}
        # key : header, values : 등장한 pages

        for page_num, text in data.items():
            buffer_texts = text.strip().split('\n')[:num_lines]
            for i in buffer_texts:
                buffer_dict.setdefault(i, []).append(page_num)
        return {line for line, pages in buffer_dict.items() if len(pages) > 1}

    result = {}
    common_headers = find_common_headers(data, num_lines)

    for page_num, text in data.items():
        lines = text.strip().split('\n')
        cleaned_head = [
            line for line in lines[:num_lines] if line not in common_headers
        ]
        cleaned_tail = lines[num_lines:]
        result[page_num] = '\n'.join(cleaned_head + cleaned_tail)

    return result

def drop_redundant_page_no(data):
    cleaned_pages = {}
    for key, text in data.items():
        # 끝에 "\n숫자"가 붙어있다면 제거
        cleaned_text = re.sub(r'\n\d{1,3}$', '', text.strip())
        cleaned_pages[key] = cleaned_text
    return cleaned_pages

def main(args):
    ### Init
    config = load_config(args.config)
    root_path = Path(config["path"]["root_path"])

    input_dir = root_path / config["path"]["input_dir"]
    output_dir = root_path / config["path"]["output_dir"]
    valid_identifier_dir = root_path / config["path"]["valid_identifier"]
    output_dir.mkdir(parents=True, exist_ok=True)

    ### Load valid identifier files
    valid_identifier = load_valid_identifier(valid_identifier_dir)
    documentName_set = {os.path.splitext(item["documentName"])[0] for item in valid_identifier}

    #### drop duplicates
    seen = {}
    for fname in os.listdir(input_dir):
        base_name = os.path.splitext(fname.split("_ocr_result")[0])[0]
        if base_name in documentName_set and base_name not in seen:
            seen[base_name] = fname  # save a first file
    print(f"Total data count without duplicates : {len(seen)}")

    batch_size = 100
    err = []
    files = list(seen.values())

    for i in tqdm(range(0, len(files), batch_size), desc="Processing PTAB"):
        batch_files = files[i:i + batch_size]
        for fname in batch_files:
            base_name = os.path.splitext(fname.split("_ocr_result")[0])[0]
            file_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, f"{base_name}.json")

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            pages = data.get("all_text_by_page", "")

            ### Drop First Page (Patent & Ptab decision info page)
            drop_page_status, filtered_data, prev_key = split_by_before_blocks(pages)

            ### 3 Drop Header & Page number
            if drop_page_status:
                filtered_data = drop_redundant_header(filtered_data)
                filtered_data = drop_redundant_page_no(filtered_data)
                with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
                continue
            else:
                err.append({
                    "filename":base_name,
                    "prev_key" : prev_key
                })

    print(f"처리된 Data 개수 : {len(os.listdir(output_dir))}")
    pd.DataFrame(err).to_csv("dropped_ocr_files.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="config/ptab_clean.json", help="Path of configuration file (e.g., ptab_clean.json)")

    args = parser.parse_args()

    main(args)