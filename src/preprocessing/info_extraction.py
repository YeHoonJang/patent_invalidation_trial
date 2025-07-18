import glob
import os
import sys
from pathlib import Path

import argparse
import json
import re
import ast

import pdb
import time

from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config_utils import load_config

def process_file(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    # content = (
    #     data.get("main_body_text", {})
    #         .get("STATEMENT OF THE CASE", {})
    #         .get("text", "")
    # )
    content = " ".join(data.values())

    PATENT_RE = re.compile(
        r"""(?ix)
        \b(?P<country>U\.?S\.?)\s*          # ← country
        (?:Pat(?:ent)?\.?\s*(?:No\.?)?\s*)? # Pat. / Patent No.
        (?P<number>
            \d{4}/\d{7}               # 2003/0108643
        | \d{1,3}(?:\s*,\s*\d{3})+  # 5,330,627
        | \d{7,9}                   # 7117449
        )
        \s*
        (?P<kind>[A-H]\d?)?           # A1, B2 …
        \b
        """
    )

    result = []
    for m in PATENT_RE.finditer(content):
        country_raw = m.group('country')
        country = re.sub(r'\W', '', country_raw).upper()

        patent_number_raw = m.group('number').strip()
        patent_number = patent_number_raw.replace(',', '').replace(' ', '').replace('/', '')

        result.append(
            {
                "Document": path.stem,
                "country": country,
                "patent_no": patent_number,
                "kind": m.group('kind') or None
            }
        )
    return result

def main(args):
    ### Init
    config = load_config(args.config)
    root_path = Path(config["path"]["root_path"])

    input_dir = root_path / config["path"]["input_dir"]
    output_dir = root_path / config["path"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    files = input_dir.glob("*.json")

    patent_infos = []
    for p in tqdm(files):
        patent_info = process_file(p)
        patent_infos += patent_info

    patent_infos_dict = {i: info for i, info in enumerate(patent_infos)}
    (output_dir/"information_extraction.json").write_text(json.dumps(patent_infos_dict, indent=2), encoding="utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="config/information_extraction.json", help="Path of configuration file (e.g., information_extraction.json)")

    args = parser.parse_args()

    main(args)