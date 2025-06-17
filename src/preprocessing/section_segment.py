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


async def process_file(path, base_prompt, client, output_dir, model):
    data = json.loads(path.read_text(encoding="utf-8"))
    full_prompt = base_prompt.format(data=data)

    response = await client.split_opinion(full_prompt)

    if model == "gpt" or model == "gpt-o":
        result_json = response.choices[0].message.function_call.arguments
    elif model == "gemini":
        result_json = response.text

    try:
        result = json.loads(result_json)
    except Exception:
        print(f"JSON Load Failed ...: {path.name}")

    output_path = output_dir/f"{model}_{path.name}"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main(args):
    ### Init
    config = load_config(args.config)
    root_path = Path(config["path"]["root_path"])

    prompt_dir_name = config["prompt"]["prompt_dir"]
    prompt_file_name = config["prompt"][args.prompt]

    prompt_dir = root_path / prompt_dir_name
    prompt_path = prompt_dir / prompt_file_name

    with open(prompt_path, "r") as f:
        base_prompt = f.read()

    load_dotenv(PROJECT_ROOT / "config" / ".env")

    model = args.model.lower()
    if model == "gpt":
        api_key = os.getenv("OPENAI_API_KEY")
    elif model == "gpt-o":
        api_key = os.getenv("OPENAI_API_KEY")
    elif model == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
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

    sem = asyncio.Semaphore(config["async"]["concurrency"])

    async def sem_task(path):
        async with sem:
            await process_file(path, base_prompt, client, output_dir, model)

    asyncio.run(tqdm_asyncio.gather(*[sem_task(p) for p in files], desc="(Async) Splitting Opinion ..."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=False, default="config/section_segment.json", help="Path of configuration file (e.g., section_segment.json)")
    parser.add_argument("--model", choices=["gpt", "gpt-o", "gemini"], required=False, default="gpt", help="LLM Model for spliting opinion")
    parser.add_argument("--prompt", type=str, required=True, default=None, help="Prompt for inferencing")

    args = parser.parse_args()

    main(args)