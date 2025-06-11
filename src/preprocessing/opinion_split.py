import argparse
import glob
import json
import os
import re
import sys
import pdb
from pathlib import Path
import time

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config_utils import load_config
from llms.llm_client import get_llm_client


def main(args):
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
    elif model == "claude":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif model == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
    else:
        raise ValueError(f"Unsupported model: {model}")
    
    if not api_key:
        raise RuntimeError(f"환경변수 {model.upper()}_API_KEY가 설정되지 않았습니다.")
    
    
    input_dir = root_path / config["path"]["input_dir"]
    output_dir = root_path / config["path"]["output_dir"] / args.prompt / config[model]["llm_params"]["model"]
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.json"))
    for path in tqdm(files, desc="Splitting Opinion ..."):
        data = json.loads(path.read_text(encoding="utf-8"))
        statement_of_case_text = data["contents"]["STATEMENT OF THE CASE"]["text"]
        analysis_text = data["contents"]["ANALYSIS"]["text"]

        full_prompt = base_prompt.format(
            statement_of_the_case=statement_of_case_text.strip(),
            analysis=analysis_text.strip()
        )

        llm_params = config[model]["llm_params"]
        client = get_llm_client(model, api_key, **llm_params)

        response = client.split_opinion(full_prompt)

        if model == "gpt" or model == "gpt-o":
            result_json = response.choices[0].message.function_call.arguments
        elif model == "claude":
            result_json = response.content[0].text.replace("```json","").replace("```", "").strip()
        elif model == "gemini":
            result_json = response.text

        try:
            result = json.loads(result_json)
        except:
            print(f"JSON Load Failed ...: {os.path.basename(path)}")

        output = json.dumps(result, indent=2)

        with open(f"{output_dir}/{model}_{os.path.basename(path)}", "w", encoding="utf-8") as f:
            f.write(output)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=False, default="config/opinion_split.json", help="Path of configuration file (e.g., opinion_split.json)")
    parser.add_argument("--model", choices=["gpt", "gpt-o", "claude", "gemini"], required=False, default="gpt", help="LLM Model for spliting opinion")
    parser.add_argument("--prompt", type=str, required=True, default=None, help="Prompt for inferencing")

    args = parser.parse_args()

    main(args)

