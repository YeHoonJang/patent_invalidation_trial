import argparse
import asyncio
import glob
import json
import os
import sys
import pdb
from pathlib import Path

import re

from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config_utils import load_config
from llms.llm_client import get_llm_client


async def predict_subdecision(path, system_prompt, base_prompt, client, labels, output_dir, model):
    data = json.loads(path.read_text(encoding="utf-8"))

    appellant = data["appellant_arguments"]
    examiner = data["examiner_findings"]

    full_prompt = base_prompt.format(
        appellant=appellant,
        examiner=examiner,
        decision_type=labels
    )

    prompt = {
        "system": system_prompt,
        "user": full_prompt
    }


    if model == "gpt" or model == "gpt-o":
        mod = await client.client.moderations.create(input=full_prompt)
        if mod.results[0].flagged:
            print(f"[BLOCKED] {path.name}")
            (output_dir / "blocked.log").open("a").write(f"{path.name}\n")
            return

    response = await client.generate_valid_json(prompt)
    result = {}
    json_result = None

    if model in ["llama", "qwen", "mistral", "t5", "deepseek"]:
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            cleaned = re.sub(r"^```json\s*|\s*```$", "", response.strip())
            try:
                result = json.loads(cleaned)
            except json.JSONDecodeError:
                if response.strip().isdigit():
                    result = {"decision_type": int(response.strip())}
    else:
        result = response

    if isinstance(result, dict) and "decision_type" in result.keys():
        try:
            json_result = {"decision_type": int(result["decision_type"])}
        except (ValueError, TypeError):
            pass

    # if model == "gpt" or model == "gpt-o":
    #     result = response
    # elif model == "claude":
    #     result = response
    # elif model == "gemini":
    #     result = response

    if json_result:
        output_path = output_dir/f"{os.path.basename(path)}"
        output_path.write_text(json.dumps(json_result, indent=2), encoding="utf-8")


def main(args):
    config = load_config(args.config)
    root_path = Path(config["path"]["root_path"])

    labels_file = config["path"]["decision_type"]
    labels_path = root_path / labels_file
    labels = json.loads(labels_path.read_text(encoding="utf-8")).keys()
    idx2labels = {i:k for i, k in enumerate(labels)}

    prompt_dir_name = config["prompt"]["prompt_dir"]
    system_prompt_file = config["prompt"]["system"]
    user_prompt_file = config["prompt"][args.prompt]

    prompt_dir = root_path / prompt_dir_name
    system_prompt_path = prompt_dir / system_prompt_file
    user_prompt_path = prompt_dir / user_prompt_file

    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()
    with open(user_prompt_path, "r") as f:
        base_prompt = f.read()

    load_dotenv(PROJECT_ROOT / "config" / ".env")
    model = args.inference_model.lower()

    use_api = True
    api_key = None
    if model == "gpt":
        api_key = os.getenv("OPENAI_API_KEY")
    elif model == "gpt-o":
        api_key = os.getenv("OPENAI_API_KEY")
    elif model == "claude":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif model == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
    elif model == "solar":
        api_key = os.getenv("UPSTAGE_API_KEY")
    elif model in ["llama", "qwen", "mistral", "deepseek", "t5"]:
       use_api = False
    else:
        raise ValueError(f"Unsupported model: {model}")

    if (use_api) and (not api_key):
        raise RuntimeError(f"환경변수 {model.upper()}_API_KEY가 설정되지 않았습니다.")

    input_dir = root_path / config["path"]["input_dir"] / args.input_model
    opinion_split_version = input_dir.parent.name

    output_dir = root_path / config["path"]["output_dir"] / args.prompt / opinion_split_version / f"input_{args.input_model}" / f"output_{config[model]['llm_params']['model']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    llm_params = config[model]["llm_params"]
    client = get_llm_client(model, api_key, **llm_params)

    all_files = sorted(input_dir.glob("*.json"))
    files = [p for p in all_files if not (output_dir / f"{p.name}").exists()]
    sem = asyncio.Semaphore(config["async"]["concurrency"])

    async def sem_task(path):
        async with sem:
            await predict_subdecision(path, system_prompt, base_prompt, client, idx2labels, output_dir, model)

    print(f"[Async] {args.input_model} ({args.prompt}) -> {config[model]['llm_params']['model']}")
    asyncio.run(tqdm_asyncio.gather(*[sem_task(p) for p in files], desc="Predict Subdecision ..."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=False, default="config/decision_predict.json", help="Path of configuration file (e.g., decision_predict.json)")
    parser.add_argument("--input_model", type=str, choices=["gpt-4o", "o3-2025-04-16", "claude-opus-4-20250514", "claude-sonnet-4-20250514", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-pro", "reg_ex"], required=True, default=None, help="LLM Model that makes input data")
    parser.add_argument("--inference_model", choices=["gpt", "gpt-o", "claude", "gemini", "llama", "qwen", "solar", "mistral", "deepseek", "t5"], required=False, default="gpt", help="LLM Model for decision prediction")
    parser.add_argument("--prompt", type=str, required=True, default=None, help="Prompt for inferencing")

    args = parser.parse_args()

    main(args)