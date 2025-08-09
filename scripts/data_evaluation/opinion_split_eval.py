import argparse
import asyncio
import json
import os
import sys
import pdb
from pathlib import Path


from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


from utils.config_utils import load_config
from llms.llm_client import get_llm_client


def load_prompts(system_prompt_path, user_prompt_path):
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    with open(user_prompt_path, "r", encoding="utf-8") as f:
        user_prompt_template = f.read().strip()
    return system_prompt, user_prompt_template


async def evaluate_one(split_path: Path, original_dir: Path, gt_dir: Path, system_prompt: str, user_template: str, judge_clients: list, judge_names: list, output_dir: Path, diff_threshold: int = 10,):
    
    split_fname = split_path.name
    base_fname = ("_").join(split_fname.split("_")[1:])
    original_path = original_dir / base_fname
    gt_path = gt_dir / base_fname

    if not (original_path.exists() and gt_path.exists()):
        print(f"[SKIP]: original / GT skip: {base_fname}")
        return
    
    original_text = original_path.read_text(encoding="utf-8")
    model_split = split_path.read_text(encoding="utf-8")
    gt_split = gt_path.read_text(encoding="utf-8")

    user_prompt = user_template.format(
        original_text = original_text,
        model_split = model_split,
        gt_split = gt_split
    )

    prompt = {"system": system_prompt, "user": user_prompt}

    results = []

    for i in range(len(judge_clients)):
        response = await judge_clients[i].generate_valid_json(prompt)
        results.append(response)

    def avg(key):
        return round(sum(r["scores"][key] for r in results) / len(results))
    
    def calc_overall(s):
        return round(
            0.35 * s["critical_leakage_rate"]
            + 0.15 * s["precision"]
            + 0.25 * s["recall"]
            + 0.15 * s["role_accuracy"]
            + 0.05 * (100 - s["missing_rate"])
            + 0.05 * (100 - s["extra_rate"])
        )
    
    score_keys = ["precision", "recall", "missing_rate", "extra_rate", "role_accuracy", "critical_leakage_rate"]
    merged_scores = {k: avg(k) for k in score_keys}

    merged = {
        "scores": merged_scores,
        "hallucination_detected": any(r["scores"]["hallucination_detected"] for r in results),
        "overall_score": calc_overall(merged_scores),
        "error_analysis": {cat: [] for cat in ["critical_leakage", "missing", "extra", "misrole"]},
        "judge_detail": {name: r for name, r in zip(judge_names, results)},
    }

    for cat in merged["error_analysis"]:
        seen = set()
        for r in results:
            for item in r["error_analysis"].get(cat, []):
                key = item.get("text") or item.get("gt_text") or item.get("model_text")
                if key and key not in seen:
                    merged["error_analysis"][cat].append(item)
                    seen.add(key)
    
    output_path = output_dir / base_fname
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=4), encoding="utf-8")


def main(args):
    config = load_config(args.config)
    root_path = Path(config["path"]["root_path"])

    prompt_dir = root_path / config["prompt"]["prompt_dir"]
    system_prompt, user_template = load_prompts(
        prompt_dir / config["prompt"]["system"],
        prompt_dir / config["prompt"]["user"]
    )

    load_dotenv(PROJECT_ROOT / "config" / ".env")
    judge_names = [m.strip().lower() for m in args.judge_models.split(",")]
    judge_clients = []

    split_models = [m.strip().lower() for m in args.split_models.split(",")]

    for s_model in split_models:
        for j_name in judge_names:
            if "gpt" in j_name:
                api_key = os.getenv("OPENAI_API_KEY")
            elif "claude" in j_name:
                api_key = os.getenv("ANTHROPIC_API_KEY")
            elif "gemini" in j_name:
                api_key = os.getenv("GOOGLE_API_KEY")

            # model = config[name]["llm_params"]["model"]
            llm_params = config[j_name]["llm_params"]
            judge_clients.append(get_llm_client(j_name, api_key, **llm_params))

        split_dir = root_path / config["path"]["split_dir"] / s_model
        prompt_ver = split_dir.parent.parent.name
        original_dir = root_path / config["path"]["original_dir"]
        gt_dir = root_path / config["path"]["gt_dir"]

        output_dir = root_path / config["path"]["output_dir"] / prompt_ver / s_model
        output_dir.mkdir(parents=True, exist_ok=True)

        all_files = sorted(split_dir.glob("*.json"))
        files = [p for p in all_files if not (output_dir / p.name).exists()]
        print(f"[Async] Evaluate {s_model} with {judge_names}")

        sem = asyncio.Semaphore(config["async"]["concurrency"])

        async def sem_task(p):
            async with sem:
                await evaluate_one(p, original_dir, gt_dir, system_prompt, user_template, judge_clients, judge_names, output_dir)

        asyncio.run(tqdm_asyncio.gather(*[sem_task(p) for p in files], desc="LLM Judge Evaluating ..."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=False, default="config/eval_split.json", help="Evaluation config file")
    parser.add_argument("--split_models", type=str, required=True, default="gpt-4o, gpt-o3, claude-sonnet, gemini-2.5-pro, gemini-2.5-flash", help="Opinion Split LLM Models")
    parser.add_argument("--judge_models", default="gpt, claude, gemini", help="Judge LLMs")

    args = parser.parse_args()
    main(args)
