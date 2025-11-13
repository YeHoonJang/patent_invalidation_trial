# -*- coding: utf-8 -*-
import os
import json
import glob
import sys
import pdb
import pandas as pd
from pathlib import Path

from dataclasses import dataclass
from typing import Any, Dict, List
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config_utils import load_config


def _join(lines: List[str]) -> str:
    return "\n".join([s.strip() for s in lines if isinstance(s, str) and s.strip()])


def build_source_text(sample: Dict[str, Any]) -> Dict[str, str]:
    return {
        "app":  _join(sample.get("appellant_arguments", [])),
        "exm":  _join(sample.get("examiner_findings", [])),
        "ptab": _join(sample.get("ptab_opinion", [])),
    }


def load_prompt(task: str) -> str:
    config_path = PROJECT_ROOT / "config" / "prompt.json"
    prompt_config = load_config(config_path)["prompt_path"]
    prompt_path = PROJECT_ROOT / prompt_config["base"][task]
    return prompt_path.read_text(encoding="utf-8")


def build_prompt(task: str, src_text: Dict[str, str]) -> str:
    prompt_text = load_prompt(task)
    label_json_path = PROJECT_ROOT / "data" / "json"

    if task in {"issue", "board"}:
        instruction = prompt_text
    elif task == "fine_sub":
        label_dict = json.loads((label_json_path / "decision_type.json").read_text(encoding="utf-8"))
        instruction = prompt_text.format(decision_type=label_dict)
    elif task == "coarse_sub":
        label_dict = json.loads((label_json_path / "coarse_decision_type.json").read_text(encoding="utf-8"))
        instruction = prompt_text.format(decision_type=label_dict)
    else:
        raise ValueError(task)
    
    return instruction.format(appellant=src_text["app"], examiner=src_text["exm"])


def _json_compact(d: Dict[str, Any]) -> str:
    return json.dumps(d, ensure_ascii=False, separators=(",", ":"))


def encode_target(task:str, raw_label, label_space: List[str]):
    lab2idx = {k: i for i, k in enumerate(label_space)}

    if task == "issue":
        labs = [l for l in (raw_label or []) if l in lab2idx]
        if not labs and "Others" in lab2idx: labs = ["Others"]
        labs_sorted = sorted(set(labs), key=lambda x: lab2idx[x])
        out = {"issue_type": labs_sorted}
        return _json_compact(out), out
    
    if task == "board":
        labs = [l for l in (raw_label or []) if l in lab2idx]
        if not labs and "Others" in lab2idx: labs = ["Others"]
        labs_sorted = sorted(set(labs), key=lambda x: lab2idx[x])
        out = {"board_ruling": labs_sorted}
        return _json_compact(out), out
    
    if task in {"fine_sub", "coarse_sub"}:
        name = raw_label if isinstance(raw_label, str) else ""
        if name not in lab2idx:
            name = "Others" if "Others" in lab2idx else label_space[0]
        idx = lab2idx[name]
        out = {"decision_type": name, "decision_number": idx}
        return _json_compact(out), out

    raise ValueError(task)
    

class PTABCaseDataset(Dataset):
    def __init__(self, root_dir:str | None, task:str, label_key: str, label_space: List[str], file_list: List[str] | None = None):
        if file_list is not None:
            self.files = sorted(file_list)
        else:
            assert root_dir is not None, "root_dir 또는 file_list 중 하나는 필요합니다."
            self.files = sorted(glob.glob(os.path.join(root_dir, "*.json")))
        if not self.files:
            raise FileNotFoundError(f"No JSON files in {root_dir} or '(from file_list)'")
        assert task in {"issue", "board", "fine_sub", "coarse_sub"}
        self.task = task
        self.label_key = label_key
        self.label_space = label_space

        self.label_df = pd.read_csv(PROJECT_ROOT / "data/csv/20250818_ptab.csv")
        self.label_df.set_index("file_name", inplace=True)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        file_path = self.files[index]
        file_name = Path(file_path).stem

        with open(self.files[index], "r", encoding="utf-8") as f:
            sample = json.load(f)
        
        if file_name in self.label_df.index:
            row = self.label_df.loc[file_name]
            sample["issue_type"] = row["issue_type"]
            sample["board_authorities"] = row["board_rulings"]
            sample["subdecision"] = row["subdecision"]
            sample["subdecision_coarse"] = row["subdecision_coarse"]

        src = build_source_text(sample)
        prompt = build_prompt(self.task, src)
        target_text, y_true_struct = encode_target(self.task, sample.get(self.label_key), self.label_space)
        return {"prompt": prompt, "target_text": target_text, "gt": y_true_struct}
    