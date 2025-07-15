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
from tqdm.asyncio import tqdm_asyncio
import asyncio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config_utils import load_config
from llms.llm_client import get_llm_client


async def process_file(path, system_prompt, base_prompt, client, output_dir, model):
    data = json.loads(path.read_text(encoding="utf-8"))

    state_of_case_text = (
        data.get('main_body_text', {})
            .get('STATEMENT OF THE CASE', {})
            .get('text', '')
    )

    full_prompt = base_prompt.format(data=state_of_case_text)

    prompt = {
        "system": system_prompt,
        "user": full_prompt
    }

    result_json = await client.generate_valid_json(prompt)
    output_path = output_dir/f"{path.name}"
    output_path.write_text(json.dumps(result_json, indent=2), encoding="utf-8")

def main(args):
    ### Init
    config = load_config(args.config)
    root_path = Path(config["path"]["root_path"])

    prompt_dir_name = config["prompt"]["prompt_dir"]
    system_prompt_file = config["prompt"]["system"]
    user_prompt_file = config["prompt"][args.prompt]

    prompt_dir = root_path / prompt_dir_name
    system_prompt_path = prompt_dir / system_prompt_file
    user_prompt_path = prompt_dir / user_prompt_file

    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    base_prompt = user_prompt_path.read_text(encoding="utf-8")

    load_dotenv(PROJECT_ROOT / "config" / ".env")

    model = args.model.lower()
    if (model == "gpt") or (model == "gpt-o"):
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        raise ValueError(f"Unsupported model: {model}")

    if not api_key:
        raise RuntimeError(f"환경변수 {model.upper()}_API_KEY가 설정되지 않았습니다.")

    input_dir = root_path / config["path"]["input_dir"]
    output_dir = root_path / config["path"]["output_dir"] / args.prompt / config[model]["llm_params"]["model"]
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_params = config[model]["llm_params"]

    client = get_llm_client(model, api_key, **llm_params)

    all_files = input_dir.glob("*.json")
    files = [p for p in all_files if not (output_dir / f"{model}_{p.name}").exists()]
    files = files[:50]

    sem = asyncio.Semaphore(config["async"]["concurrency"])

    async def sem_task(path):
        async with sem:
            await process_file(path, system_prompt, base_prompt, client, output_dir, model)

    asyncio.run(tqdm_asyncio.gather(*[sem_task(p) for p in files], desc="(Async) Extract Information ..."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=False, default="config/information_extraction_prompt.json", help="Path of configuration file (e.g., information_extraction_prompt.json)")
    parser.add_argument("--model", choices=["gpt", "gpt-o"], required=False, default="gpt", help="LLM Model for Information Extraction")
    parser.add_argument("--prompt", type=str, required=True, default=None, help="Prompt for inferencing")

    args = parser.parse_args()

    main(args)