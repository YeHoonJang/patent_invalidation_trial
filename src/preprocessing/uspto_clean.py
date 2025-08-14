import re
import os
import pdb
import json
import sys
from pathlib import Path
import glob

import argparse
import pandas as pd
from datetime import datetime

from tqdm import tqdm
from html import unescape

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config_utils import load_config

import xml.etree.ElementTree as ET

def parse_description_sections(root):
    description_root = root.find(".//description")
    if description_root is None:
        return []

    drawings_section = description_root.find("description-of-drawings")
    drawings_ids = set()
    if drawings_section is not None:
        for elem in drawings_section.iter():
            if "id" in elem.attrib:
                drawings_ids.add(elem.attrib["id"])

    sections = []
    current_section = None

    stack = list(description_root)
    while stack:
        elem = stack.pop(0)
        if "id" in elem.attrib and elem.attrib["id"] in drawings_ids:
            continue

        if elem.tag == "heading" and elem in list(description_root):
            if current_section:
                sections.append(current_section)
            current_section = {
                "heading": (elem.text or "").strip(),
                "paragraphs": []
            }

        elif current_section:
            text_items = extract_full_text(elem)
            if text_items:
                current_section["paragraphs"].extend(text_items)

            stack[0:0] = list(elem)


    if current_section:
        sections.append(current_section)

    return sections

def extract_full_text(elem):
    def get_self_text(elem):
        return unescape(elem.text.strip()) if elem.text else ''

    def extract_table(elem):
        table_xml_str = ET.tostring(elem, encoding="unicode", method="xml").strip()
        return [{"table": table_xml_str}]

    if 'num' in elem.attrib:
        if elem.tag == 'tables':
            return extract_table(elem)
        else:
            if any('num' in child.attrib for i, child in enumerate(elem.iter()) if i > 0):
                return [get_self_text(elem)] if get_self_text(elem) else []
            else:
                full_text = unescape(''.join(elem.itertext()).strip())
                return [full_text] if full_text else []
    return []

def parse_claim_text(elem):
    parts = [(elem.text or "").strip()]

    for child in elem:
        if child.tag == "claim-text":
            parts.append(parse_claim_text(child))
        else:
            if child.text:
                parts.append(child.text.strip())
            if child.tail:
                parts.append(child.tail.strip())

    if all(isinstance(p, str) for p in parts):
        full_text = " ".join(p for p in parts if p)
        return {"text": full_text}

    root_text = " ".join(p for p in parts if isinstance(p, str) and p)
    components = [p for p in parts if isinstance(p, dict)]

    result = {"text": root_text}
    if components:
        result["components"] = components
    return result

def extract_claims_text(root):
    def extract_claim_dependency(claim_elem):
        first_text_elem = claim_elem.find("claim-text")
        if first_text_elem is None:
            return None

        claim_ref_elem = first_text_elem.find("claim-ref")
        if claim_ref_elem is not None:
            return claim_ref_elem.attrib.get("idref")  # e.g., "CLM-00001"

        return None

    claims_root = root.find(".//claims")
    if claims_root is None:
        return []

    claim_blocks = []

    for claim in claims_root.findall("claim"):
        claim_id = claim.attrib.get("id")
        claim_num = claim.attrib.get("num", "").lstrip("0")
        depends_on = extract_claim_dependency(claim)

        root_claim_text = claim.find("claim-text")
        if root_claim_text is None:
            continue

        structured_claim = {
            "claim_id": claim_id,
            "claim_num": claim_num,
            "depends_on": depends_on,
            "claim_text": parse_claim_text(root_claim_text)
        }

        claim_blocks.append(structured_claim)
    return claim_blocks

def main(args):
    ### Init
    config = load_config(args.config)
    root_path = Path(config["path"]["root_path"])

    input_dir = root_path / config["path"]["input_dir"]
    output_dir = root_path / config["path"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in input_dir.glob("**/*.xml")]
    for file in tqdm(files, desc="Parsing XML files ..."):
        file_path = file.relative_to(input_dir).with_suffix(".json")
        output_path = output_dir /file_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            tree = ET.parse(file)
            root = tree.getroot()

            file_attr = root.attrib.get("file")

            title_element = root.find(".//invention-title")
            title = title_element.text.strip() if title_element is not None else None

            claims = extract_claims_text(root)
            descriptions = parse_description_sections(root)

            data = {
                "filename" : file_attr,
                "title": title,
                "claims":claims,
                "description":descriptions
            }

            output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except:
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="config/uspto_clean.json", help="Path of configuration file (e.g., uspto_clean.json)")

    args = parser.parse_args()
    main(args)