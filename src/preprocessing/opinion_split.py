import argparse
import asyncio
import atexit
import glob
import json
import os
import re
import sys
import signal
import pdb
from pathlib import Path
import time
import wandb

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config_utils import load_config
from llms.llm_client import get_llm_client


async def split_opinion(path, system_prompt, base_prompt, client, output_dir, model, stats, lock):
    data = json.loads(path.read_text(encoding="utf-8"))
    statement_of_case_text = data["main_body_text"]["STATEMENT OF THE CASE"]["text"].strip()
    analysis_text = data["main_body_text"]["ANALYSIS"]["text"].strip()

    full_prompt = base_prompt.format(
        statement_of_the_case=statement_of_case_text,
        analysis=analysis_text
        )
    
    prompt = {
        "system": system_prompt,
        "user": full_prompt
    }
    
    t0 = time.perf_counter()

    try:
        if "gpt" in model:
            mod = await client.client.moderations.create(input=full_prompt)
            if mod.results[0].flagged:
                print(f"[BLOCKED] {path.name}")
                (output_dir / "blocked.log").open("a").write(f"{path.name}\n")
                return
        
        if "gemini" in model:
            response, input_token, cached_token, output_token, reasoning_token = await client.generate_valid_json(prompt)

        else:
            response = await client.generate_valid_json(prompt)
        
        latency_ms = round((time.perf_counter() - t0) * 1000)

        output_path = output_dir/f"{model}_{path.name}"
        output_path.write_text(json.dumps(response, indent=2), encoding="utf-8")

        wandb.log({
            "name": path.name,
            "status": "ok",
            "input_tokens": input_token if input_token is not None else -1,
            "cached_tokens": cached_token if cached_token is not None else -1,
            "output_token": output_token if output_token is not None else -1,
            "reasoning_token": reasoning_token if reasoning_token is not None else -1,
            "latency_ms": latency_ms
        })

        async with lock:
            stats["processed"] += 1
            stats["succeeded"] += 1
            if input_token: stats["sum_input_tokens"] += input_token
            if cached_token: stats["sum_cached_tokens"] += cached_token
            if output_token: stats["sum_output_tokens"] += output_token
            if reasoning_token: stats["sum_reasoning_tokens"] += reasoning_token
            stats["sum_latency_ms"] += latency_ms

    except Exception as e:
        latency_ms = round((time.perf_counter() - t0) * 1000)
        wandb.log({
            "name": path.name,
            "status": f"error:{type(e).__name__}",
            "latency_ms": latency_ms
        })

        async with lock:
            stats["processed"] += 1
            stats["failed"] += 1
            stats["sum_latency_ms"] += latency_ms


def main(args):
    config = load_config(args.config)
    root_path = Path(config["path"]["root_path"])

    prompt_dir = root_path /config["prompt"]["prompt_dir"]
    system_prompt_path = prompt_dir / config["prompt"]["system"]
    user_prompt_path = prompt_dir / config["prompt"][args.prompt]

    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()

    with open(user_prompt_path, "r") as f:
        base_prompt = f.read()

    load_dotenv(PROJECT_ROOT / "config" / ".env")

    model = args.model.lower()
    if "gpt" in model:
        api_key = os.getenv("OPENAI_API_KEY")
    elif "claude" in model:
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif "gemini" in model:
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
    
    all_files = sorted(input_dir.glob("*.json"))
    files = [p for p in all_files if not (output_dir / f"{model}_{p.name}").exists()]

    run_name = f"opinion_split_{config[model]["llm_params"]["model"]}_{args.prompt}"
    wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=run_name, config={
        "task": "opinion_split",
        "model_alias": model,
        "provider_model": config[model]["llm_params"]["model"],
        "prompt_name": args.prompt,
        "concurrency": config["async"]["concurrency"],
        "num_files": len(files),
    })

    t_run0 = time.perf_counter()
    stats = {
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "sum_input_tokens": 0,
        "sum_cached_tokens": 0,
        "sum_output_tokens": 0,
        "sum_reasoning_tokens": 0,
        "sum_latency_ms": 0
    }

    FINALIZED = False
    EXIT_REASON = "completed"
    
    def finalize():
        nonlocal FINALIZED, EXIT_REASON
        if FINALIZED:
            return
        FINALIZED = True

        if wandb.run is None:
            return
        
        elapsed_s = round(time.perf_counter() - t_run0, 3)

        wandb.summary["run_status"] = EXIT_REASON
        wandb.summary["run_elapsed_s"] = elapsed_s
        wandb.summary["files_processed"] = stats["processed"]
        wandb.summary["files_succeeded"] = stats["succeeded"]
        wandb.summary["files_failed"] = stats["failed"]
        wandb.summary["sum_input_tokens"] = stats["sum_input_tokens"]
        wandb.summary["sum_cached_tokens"] = stats["sum_cached_tokens"]
        wandb.summary["sum_output_tokens"] = stats["sum_output_tokens"]
        wandb.summary["sum_reasoning_tokens"] = stats["sum_reasoning_tokens"]
        wandb.summary["total_latency_ms"] = stats["sum_latency_ms"]
        if stats["processed"]:
            wandb.summary["avg_input_tokens"] = round(stats["sum_input_tokens"]  / stats["processed"], 2)
            wandb.summary["avg_cached_tokens"] = round(stats["sum_cached_tokens"]  / stats["processed"], 2)
            wandb.summary["avg_output_tokens"] = round(stats["sum_output_tokens"] / stats["processed"], 2)
            wandb.summary["avg_reasoning_tokens"] = round(stats["sum_reasoning_tokens"] / stats["processed"], 2)
            wandb.summary["avg_latency_ms"] = round(stats["sum_latency_ms"] / stats["processed"], 2)
        wandb.finish()

    def handle_sig(sig, frame):
        nonlocal EXIT_REASON
        EXIT_REASON = f"signal:{sig.name}"
        finalize()
        sys.exit(0)

    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, handle_sig)

    sem = asyncio.Semaphore(config["async"]["concurrency"])
    lock = asyncio.Lock()
    
    async def sem_task(path):
        async with sem:
            await split_opinion(path, system_prompt, base_prompt, client, output_dir, model, stats, lock)

    asyncio.run(tqdm_asyncio.gather(*[sem_task(p) for p in files], desc=f"(Async) [{config[model]["llm_params"]["model"]}] Splitting Opinion ..."))

    EXIT_REASON = "completed"
    finalize()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=False, default="config/opinion_split.json", help="Path of configuration file (e.g., opinion_split.json)")
    parser.add_argument("--model", choices=["gpt", "gpt-o", "gpt-5", "claude", "gemini"], required=False, default="gpt", help="LLM Model for spliting opinion")
    parser.add_argument("--prompt", type=str, required=True, default=None, help="Prompt for inferencing")
    parser.add_argument("--wandb_entity", default="patent_project")
    parser.add_argument("--wandb_project", default="opinion_split")
    parser.add_argument("--run_name", default=None)

    args = parser.parse_args()

    main(args)

