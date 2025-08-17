import re
import os
import glob
import json
import sys
from pathlib import Path

import argparse
import pandas as pd
from datetime import datetime, timedelta
import ast

import xml.etree.ElementTree as ET
from collections import Counter

import zipfile

from tqdm import tqdm
import pdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config_utils import load_config

def extract_xml_file(path: Path) -> Path:
    with zipfile.ZipFile(path, 'r') as zip_ref:
        xml_file = next(f for f in zip_ref.namelist() if f.endswith('.xml'))  # first xml
        zip_ref.extract(xml_file, path=path.parent)
        return path.parent / xml_file

def extract_content(xml_path: Path, content: str) -> list[str]:
    # xml pattern
    if xml_path.stem.startswith("pa"):
        pattern = r"<patent-application-publication.*?</patent-application-publication>"
    else:
        pattern = r"<us-patent-application.*?</us-patent-application>"

    return re.findall(pattern, content, re.DOTALL)

def save_patent_if_matched(patent_xml: str, output_dir: Path, app_nums: list[dict], ref_nums: list[dict]) -> None:
    root = ET.fromstring(patent_xml)
    apl_num = None

    ### Appellant Application Number
    doc_id = root.find(".//application-reference/document-id")
    if doc_id is None:
        return

    country = doc_id.findtext("country", "").strip()
    if country.upper() == "US":
        apl_num = doc_id.findtext("doc-number")

    ### Prior Art Number
    pub_nums = []
    for doc_id in root.findall(".//publication-reference/document-id"):
        country = doc_id.findtext("country", "").strip()

        if country.upper() == "US":
            pub_nums.append({
                "pat_no":doc_id.findtext("doc-number"),
                "kind":doc_id.findtext("kind"),
            })

    if not apl_num:
        return
    if not pub_nums:
        return

    full_file = root.attrib.get("file")
    filename   = full_file.split("-", 1)[0]

    save_paths = []
    for item in app_nums:
        if str(apl_num) == str(item.get("patent_no")):
            path = output_dir / item.get("Document") / "ApplicantPatent" / f"{filename}.xml"
            save_paths.append(path)

    for ref in ref_nums.values():
        pat_ref  = ref["patent_no"]
        kind_ref = (ref.get("kind") or "").strip().upper()

        for pub in pub_nums:
            pat_pub  = pub["pat_no"]
            if str(pat_ref) == str(pat_pub):
                if kind_ref:
                    kind_pub = (pub.get("kind") or "").strip().upper()
                    if str(kind_ref) == str(kind_pub):
                        path = output_dir / ref.get("Document") / "PriorArtPatent" / f"{filename}.xml"
                        save_paths.append(path)
                else:
                    path = output_dir / ref.get("Document") / "PriorArtPatent" / f"{filename}.xml"
                    save_paths.append(path)

    for path in save_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(patent_xml, encoding="utf-8")

def main(args):
    ### Init
    config = load_config(args.config)
    root_path = Path(config["path"]["root_path"])

    input_dir = root_path / config["path"]["input_dir"]
    output_dir = root_path / config["path"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    ### Patent Number Data
    # Appellant Patents
    cleaned_ptab_dir = root_path / config["path"]["cleaned_ptab_dir"]
    cleaned_ptab = [f.name.removesuffix(".json") for f in cleaned_ptab_dir.glob("*.json")]
    valid_identifier_dir = root_path / config["path"]["valid_identifier"]
    valid_identifiers = json.loads(valid_identifier_dir.read_text(encoding="utf-8"))

    doc_counts = Counter(
        item.get("documentName", "").removesuffix(".pdf")
        for item in valid_identifiers.values()
    )

    appellants = []
    for item in valid_identifiers.values():
        doc = item.get("documentName", "").removesuffix(".pdf")

        # Remove Duplicates
        if doc_counts[doc] > 1:
            continue

        if doc not in cleaned_ptab:
            continue

        pat = item.get("appellantApplicationNumberText") # Application Number
        appellants.append({
            "Document": doc,
            "country": "US",
            "patent_no": pat,
            "kind":None
        })

    print(f"Successfully extracted {len(appellants)} unique appellant Application Number values from {len(valid_identifiers)} total identifiers.")

    # Prior Art Patents
    references_dir = root_path / config["path"]["prior_art_references"]
    references = json.loads(references_dir.read_text(encoding="utf-8")) # Publication Number


    files = [f for f in input_dir.glob("*/*.zip")]

    for file in tqdm(files, desc="Processing ZIP files ..."):
        ### unzip ...
        xml_path = extract_xml_file(file)
        data = xml_path.read_text(encoding="utf-8")

        ### extract patents
        patents = extract_content(xml_path, data)

        ### Patent Number Matching
        for patent in patents:
            save_patent_if_matched(
                patent_xml = patent,
                output_dir = output_dir,
                app_nums    = appellants,
                ref_nums    = references
            )

        os.remove(xml_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="config/uspto_collect.json", help="Path of configuration file (e.g., uspto_collect.json)")

    args = parser.parse_args()
    main(args)
