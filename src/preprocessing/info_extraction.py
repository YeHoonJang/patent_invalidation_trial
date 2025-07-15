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

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def process_file(path):
    data = json.loads(path.read_text(encoding="utf-8"))

    PATENT_RE = re.compile(
        r"""(?ix)
        \bU\.?S\.?\s*                 # "US" or "U.S."
        (?:Pat(?:ent)?\.?\s*          #  Pat. / Patent
        (?:No\.?)?\s*                 #  No. / No
        )?
        (                             # ── patent number format (example)
            \d{4}/\d{7}               # 2003/0108643
        | \d{1,3}(?:\s*,\s*\d{3})+    # 5,330,627  / 2,738,915
        | \d{7,9}                     # 7117449
        )
        \s*
        ([A-H]\d?)?                   # Kind Code (A1, B2 …)
        \b
    """)


    result = {
        "patent_no" : path.name,
        "claims" : [m.group(0) for m in PATENT_RE.finditer(
            " ".join(data.values())
        )]
    }

    return result

def main(args):
    ### Init
    config = load_config(args.config)
    root_path = Path(config["path"]["root_path"])

    input_dir = root_path / config["path"]["input_dir"]
    output_dir = root_path / config["path"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    files = input_dir.glob("*.json")

    result = []
    for p in tqdm(files):
        patent_numbers = process_file(p)
        result.append(patent_numbers)

    df = pd.DataFrame(result)
    df.to_csv(output_dir/"information_extraction.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="config/information_extraction.json", help="Path of configuration file (e.g., information_extraction.json)")

    args = parser.parse_args()

    main(args)