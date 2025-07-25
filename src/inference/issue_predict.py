import argparse
import asyncio
import glob
import json
import os
import sys
import pdb
from pathlib import Path

from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config_utils import load_config
from llms.llm_client import get_llm_client


async def predict_issue_type(path, system_prompt, base_prompt, client, labels, meaning_dict, output_dir, model):

    data = json.loads(path.read_text(encoding="utf-8"))
    # pdb.set_trace()
    appellant = data["appellant_arguments"]
    examiner = data["examiner_findings"]

    full_prompt = base_prompt.format(
        appellant=appellant,
        examiner=examiner,
        issue_type=labels,
        meaning_dict=meaning_dict
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

    # if model == "gpt" or model == "gpt-o":
    #     result = response
    # elif model == "claude":
    #     result = response
    # elif model == "gemini":
    #     result = response


    output_path = output_dir/f"{os.path.basename(path)}"
    output_path.write_text(json.dumps(response, indent=2), encoding="utf-8")


def main(args):
    config = load_config(args.config)
    root_path = Path(config["path"]["root_path"])

    labels_file = config["path"]["issue_type"]
    labels_path = root_path / labels_file
    meaning_dict = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = meaning_dict.keys()
    idx2labels = {i:k for i, k in enumerate(labels)}
    # pdb.set_trace()

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
    opinion_split_version = input_dir.parent.name
    output_dir = root_path / config["path"]["output_dir"] / args.prompt / opinion_split_version/ config[model]["llm_params"]["model"]
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_params = config[model]["llm_params"]
    client = get_llm_client(model, api_key, **llm_params)
    
    all_files = sorted(input_dir.glob("*.json"))
    files = [p for p in all_files if not (output_dir / f"{p.name}").exists()]
    sem = asyncio.Semaphore(config["async"]["concurrency"])

    async def sem_task(path):
        async with sem:
            await predict_issue_type(path, system_prompt, base_prompt, client, idx2labels, meaning_dict, output_dir, model)

    asyncio.run(tqdm_asyncio.gather(*[sem_task(p) for p in files], desc=f"(Async) [{config[model]["llm_params"]["model"]}] Predict Issue Type ..."))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=False, default="config/issue_predict.json", help="Path of configuration file (e.g., issue_predict.json)")
    parser.add_argument("--model", choices=["gpt", "gpt-o", "claude", "gemini"], required=False, default="gpt", help="LLM Model for issue type prediction")
    parser.add_argument("--prompt", type=str, required=True, default=None, help="Prompt for inferencing")

    args = parser.parse_args()

    main(args)