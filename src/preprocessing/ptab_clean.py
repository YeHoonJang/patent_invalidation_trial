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

def load_valid_identifier(valid_identifier_dir) -> list:
    valid_identifier = []
    with open(valid_identifier_dir) as f:
        for i, line in enumerate(f, start=1):
            try:
                valid_identifier.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[오류] {i}번째 {line} 파일 파싱 실패: {e}")
    return valid_identifier

def split_by_before_blocks(pages: dict) -> dict:
    def fuzzy(word):
        return r'\s*'.join(list(word)) + r'\s*'

    ptab_intro_pattern = re.compile(
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

    page_keys_sorted = sorted(pages.keys(), key=int)
    ptab_intro_page_idxs = [
        i for i, key in enumerate(page_keys_sorted)
        if re.search(ptab_intro_pattern, pages[key][:500])
    ]

    if len(ptab_intro_page_idxs) == 1:
        intro_idx = ptab_intro_page_idxs[0]
        return {
            key: pages[key]
            for i, key in enumerate(page_keys_sorted)
            if i >= intro_idx  # keep pages from PTAB intro
        }
    return None # if no intro page is found or multiple are found

def preprocess(pages: dict) -> dict:
    def find_common_headers(pages: dict, header_line_limit: int=5) -> set:
        # key: line (potential header), value: list of page numbers where it appears
        line_to_pages = {}

        for page_idx, text in pages.items():
            top_lines = text.strip().split('\n')[:header_line_limit]
            for line_idx, line in enumerate(top_lines):
                line_to_pages.setdefault((line_idx, line), []).append(page_idx)

       # Keep only (idx, line) pairs that appear in 2 or more pages
        common_headers = {
            (i, l) for (i, l), pages_set in line_to_pages.items()
            if len(pages_set) > 1
        }
        return common_headers

    common_headers = find_common_headers(pages)
    cleaned_pages = {}

    for page_num, page_text in pages.items():
        lines = page_text.strip().split('\n')

        # Remove common header lines from the top of each page
        cleaned_lines = [
            line for idx, line in enumerate(lines)
            if (idx, line) not in common_headers
        ]

        text = '\n'.join(cleaned_lines)

        # Remove a trailing page number (e.g., "\n23") at the end of the text
        text = re.sub(r'\n\d{1,3}$', '', text)

        # Remove Korean characters to keep only English
        text = re.sub(r'[가-힣]', '', text)

        # Remove \n
        cleaned_text = text.replace('\n', ' ')

        cleaned_pages[page_num] = cleaned_text

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
    valid_identifiers = load_valid_identifier(valid_identifier_dir)
    document_name_set = {os.path.splitext(item["documentName"])[0] for item in valid_identifiers}

    #### Drop duplicates, only keep the first occurrence
    unique_file_map = {}
    for fname in os.listdir(input_dir):
        base_name = os.path.splitext(fname.split("_ocr_result")[0])[0]
        if base_name in document_name_set and base_name not in unique_file_map:
            unique_file_map[base_name] = fname  # save the first occurremce

    print(f"==== Total data count without duplicates: {len(unique_file_map)}")

    batch_size = 100
    files = list(unique_file_map.values())
    korean_count = []
    ### Preprocess --------
    for i in tqdm(range(0, len(files), batch_size), desc="Processing PTAB"):
        batch_files = files[i:i + batch_size]
        for fname in batch_files:
            file_id = os.path.splitext(fname.split("_ocr_result")[0])[0]
            file_path = input_dir / fname
            output_path = output_dir / f"{file_id}.json"

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            pages = data.get("all_text_by_page", "")

            ### Drop PTAB introduction page (Patent & Ptab decision info page)
            filtered_pages = split_by_before_blocks(pages)
            if filtered_pages is None:
                continue

            ### Preprocess
            preprocessed_data = preprocess(filtered_pages)

            ## Save
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(preprocessed_data, f, ensure_ascii=False, indent=2)

    print(f"==== Total preprocessed data count : {len(os.listdir(output_dir))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="config/ptab_clean.json", help="Path of configuration file (e.g., ptab_clean.json)")

    args = parser.parse_args()

    main(args)