# -*- coding: utf-8 -*-
import os
import json
import argparse
import datetime
import numpy as np
import sys
import pdb
import wandb
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing.case_dataset import PTABCaseDataset
from finetune.collator import Collator
from utils.config_utils import load_config
from utils.seed import set_seed_all, seed_worker
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score


def resolve_splits(data_dir: str, split_json: str):
    base = Path(data_dir)
    spec = json.load(open(split_json, "r", encoding="utf-8"))
    def to_abs(names):
        out = []
        for name in names:
            p = base / name
            if not p.exists():
                raise FileNotFoundError(f"[split] not found: {p}")
            out.append(str(p))
        return out
    train_files = to_abs(spec.get("train", []))
    valid_files = to_abs(spec.get("valid", []))
    test_files  = to_abs(spec.get("test",  []))
    if not train_files or not valid_files:
        raise ValueError("The split_json must contain at least train and valid.")
    return train_files, valid_files, test_files


def build_model(model_name: str, bf16: bool):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        bias="none"
    )
    model = get_peft_model(model, lora)
    return tok, model


def extract_json(text: str) -> dict | None:
    s = text.find("{")
    e = text.rfind("}")
    
    if s == -1 or e == -1 or e <= s:
        return None
    
    frag = text[s:e+1]
    try:
        return json.loads(frag)
    except Exception:
        frag = frag.replace("“","\"").replace("”","\"").replace("’","'")
        try:
            return json.loads(frag)
        except Exception:
            return None


def clip_to_label_space_list(xs, label_space):
    return [x for x in xs if x in label_space]


def postprocess_pred(task: str, pred_text: str, label_space: list[str]) -> dict:
    obj = extract_json(pred_text) or {}

    if task == "issue":
        raw = obj.get("issue_type", [])
        raw = raw if isinstance(raw, list) else []
        valid = [x for x in raw if x in label_space]
        invalid = [x for x in raw if x not in label_space]
        return {"issue_type": valid, "_invalid": invalid}
    
    if task == "board":
        raw = obj.get("board_authorities", [])
        raw = raw if isinstance(raw, list) else []
        valid = [x for x in raw if x in label_space]
        invalid = [x for x in raw if x not in label_space]
        return {"board_authorities": valid, "_invalid": invalid}
    
    name = obj.get("decision_type", None)
    if name in label_space:
        return {"decision_type": name, "decision_number": label_space.index(name), "_invalid": None}
    else:
        return {"decision_type": None, "decision_number": -1, "_invalid": name}


def metrics_multilabel(y_true_list: list[list[str]], y_pred_list: list[list[str]], label_space: list[str]) -> dict:
    lab2idx = {k:i for i,k in enumerate(label_space)}

    def binarize(ls):
        v = [0]*len(label_space)
        for l in ls:
            if l in lab2idx: v[lab2idx[l]] = 1
        return v
    
    Yt = np.array([binarize(x) for x in y_true_list], dtype=int)
    Yp = np.array([binarize(x) for x in y_pred_list], dtype=int)
    
    report = {
        "exact_match": (Yt==Yp).all(axis=1).mean().item(),
        "micro_precision": precision_score(Yt, Yp, average="micro", zero_division=0),
        "micro_recall": recall_score(Yt, Yp, average="micro", zero_division=0),
        "micro_f1": f1_score(Yt, Yp, average="micro", zero_division=0),
        "macro_precision": precision_score(Yt, Yp, average="macro", zero_division=0),
        "macro_recall": recall_score(Yt, Yp, average="macro", zero_division=0),
        "macro_f1": f1_score(Yt, Yp, average="macro", zero_division=0),
        "hamming_loss": (Yt != Yp).mean().item(),
        "coverage_gt": (Yt.sum(axis=1)>0).mean().item(),
        "coverage_pred": (Yp.sum(axis=1)>0).mean().item(),
    }
    return report


def metrics_multiclass(y_true_idx: list[int], y_pred_idx: list[int]) -> dict:
    Yt = np.array(y_true_idx)
    Yp = np.array(y_pred_idx)

    report = {
        "accuracy": accuracy_score(Yt, Yp),
        "balanced_acc": balanced_accuracy_score(Yt, Yp),
        "macro_precision": precision_score(Yt, Yp, average="macro", zero_division=0),
        "macro_recall":    recall_score(Yt, Yp, average="macro", zero_division=0),
        "macro_f1":        f1_score(Yt, Yp, average="macro", zero_division=0),
        "micro_f1":        f1_score(Yt, Yp, average="micro", zero_division=0),
        "weighted_f1":     f1_score(Yt, Yp, average="weighted", zero_division=0),
        "coverage_gt": 1.0, "coverage_pred": 1.0,
    }
    return report


def evaluate_generate(model, tok, dataset, device, task: str, label_space: list[str]) -> dict:
    preds_struct, gts_struct = [], []
    model.eval()
    with torch.no_grad():
        for ex in dataset:
            prompt = ex["prompt"]
            y_true = ex["gt"]
            inpt = tok(prompt, return_tensors="pt").to(device)
            gen = model.generate(
                **inpt, max_new_tokens=64, do_sample=False, num_beams=1, eos_token_id=tok.eos_token_id)
            out_text = tok.decode(gen[0][inpt["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = postprocess_pred(task, out_text, label_space)
            preds_struct.append(pred); gts_struct.append(y_true)

    if task in {"issue", "board"}:
        gt_list = [d["issue_type"] if "issue_type" in d else d["board_authorities"] for d in gts_struct]
        pred_list = [d["issue_type"] if "issue_type" in d else d["board_authorities"] for d in preds_struct]
        return metrics_multilabel(gt_list, pred_list, label_space)
    else:
        gt_idx = [label_space.index(d["decision_type"]) for d in gts_struct]
        pred_idx = [p["decision_number"] if isinstance(p.get("decision_number"), int) else -1 for p in preds_struct]
        mask = np.array([i >= 0 for i in pred_idx])
        if mask.any():
            rep_valid = metrics_multiclass([gt_idx[i] for i in np.where(mask)[0]], [pred_idx[i] for i in np.where(mask)[0]])
        else:
            rep_valid = {k:0.0 for k in ["accuracy","balanced_acc","macro_precision","macro_recall","macro_f1","micro_f1","weighted_f1","coverage_gt","coverage_pred"]}
        overall_acc = float((mask & (np.array(pred_idx)==np.array(gt_idx))).sum() / len(gt_idx))
        rep_valid["overall_accuracy"] = overall_acc
        rep_valid["coverage_pred_valid"] = float(mask.mean())
        return rep_valid
    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["issue","board","fine_sub","coarse_sub"], required=True)
    ap.add_argument("--data_dir",  required=True)    # JSON들이 있는 폴더
    ap.add_argument("--split_json", required=True)

    ap.add_argument("--label_key", required=True)
    ap.add_argument("--label_space", required=True)  # labels json 파일
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--max_length", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--train_bsz", type=int, default=1)
    ap.add_argument("--eval_bsz", type=int, default=1)  # generate 평가라 보통 1이 안전
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--outdir", default="runs/clm_json")

    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--wandb_entity", default="patent_project")
    ap.add_argument("--wandb_project", default="ptab_finetune")
    ap.add_argument("--wandb_task", default="finetune_qwen")
    
    ap.add_argument("--log_interval", type=int, default=50)  

    args = ap.parse_args()
    
    if args.use_wandb:
        run_name = f"{args.wandb_task}_{args.model}_{args.task}"
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            resume="allow",
        )

    set_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    raw = json.load(open(args.label_space, "r", encoding="utf-8"))
    if isinstance(raw, dict):
        label_space = sorted({ k.strip() for k in raw.keys() })
    elif isinstance(raw, list):
        label_space = sorted({ str(k).strip() for k in raw })
    else:
        raise ValueError("labels json must be dict or list")
    
    # TODO data Label key 붙이기!!
    train_files, valid_files, test_files = resolve_splits(args.data_dir, args.split_json)

    ds_train = PTABCaseDataset(None, args.task, args.label_key, label_space, file_list=train_files)
    ds_valid = PTABCaseDataset(None, args.task, args.label_key, label_space, file_list=valid_files)
    ds_test  = PTABCaseDataset(None, args.task, args.label_key, label_space, file_list=test_files) if len(test_files)>0 else None


    def normalize_label(s: str) -> str:
        return " ".join(str(s).split()).strip()


    def peek(ds, n=64):
        if ds is None:
            print("[SANITY] dataset=None")
            return
        cov = 0
        seen = Counter()
        for i, ex in enumerate(ds):
            gt = ex["gt"]
            if args.task in {"issue", "board"}:
                key = "issue_type" if args.task == "issue" else "board_ruling"
                labs = [normalize_label(x) for x in (gt.get(key, []) or [])]
                cov += int(len(labs) > 0)
                seen.update(labs)
            else:
                lab = normalize_label(gt.get("decision_type", "") or "")
                cov += int(bool(lab))
                if lab: seen.update([lab])
            if i + 1 >= n:
                break
        total = i + 1
        print(f"[SANITY] first {total} samples coverage_gt={cov/total:.3f}")
        print("[SANITY] sample label_space head:", list(label_space)[:10])
        print("[SANITY] top seen labels:", seen.most_common(10))

    print("=== SANITY on valid/train ===")
    peek(ds_valid)
    peek(ds_train)


    tok, model = build_model(args.model, bf16=args.bf16)
    coll = Collator(tok, max_length=args.max_length)

    def mk(ds, bsz, shuf):
        if ds is None:
            return None
        g = torch.Generator()
        g.manual_seed(args.seed)
        return DataLoader(
            ds,
            batch_size=bsz,
            shuffle=shuf,
            collate_fn=coll,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=g
        )
    train_loader = mk(ds_train, args.train_bsz, True)

    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params":[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay":0.01},
        {"params":[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay":0.0},
    ]
    optim = torch.optim.AdamW(params, lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    model.to(device)

    best, best_dir = -1.0, None

    for ep in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {ep}")
        optim.zero_grad(set_to_none=True)

        running_loss = 0.0
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                out = model(**batch)              # labels는 collator가 구성
                loss = out.loss / args.grad_accum


            # <<< ADD: supervised token debug >>>
            if (step % args.log_interval == 0) or (step == 1):
                if "labels" in batch:
                    lbl = batch["labels"]
                    n_sup = (lbl != -100).sum().item()
                    print(f"[DBG] supervised_tokens={n_sup}")
                    if args.use_wandb:
                        wandb.log({"train/supervised_tokens": n_sup, "epoch": ep})
            # <<< END >>>

            scaler.scale(loss).backward()
            running_loss += loss.item() * args.grad_accum

            if step % args.grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            if (step % args.log_interval == 0) or (step == 1):
                avg_loss = running_loss / step
                pbar.set_postfix({
                    "loss": f"{loss.item()*args.grad_accum:.4f}",
                    "avg_loss": f"{avg_loss:.4f}"
                })
                if args.use_wandb:
                    wandb.log({
                        "train/loss": loss.item()*args.grad_accum,
                        "train/avg_loss": avg_loss,
                        "train/lr": optim.param_groups[0]["lr"],
                        "train/gpu_mem_alloc_GB": (torch.cuda.memory_allocated()/1024**3) if torch.cuda.is_available() else 0.0,
                        "train/gpu_mem_peak_GB": (torch.cuda.max_memory_allocated()/1024**3) if torch.cuda.is_available() else 0.0,
                        "global_step": (ep-1)*len(train_loader) + step,
                        "epoch": ep,
                    })
            pbar.update(1)
        pbar.close()

        epoch_loss = running_loss / len(train_loader)
        print(f"[Train][Ep{ep}] avg_loss={epoch_loss:.4f}")

        if args.use_wandb:
            wandb.log({
                "train/epoch_loss": epoch_loss,
                "epoch": ep
            })

        # -------- Validation (generate → parse → metrics) --------
        val_metrics = evaluate_generate(model, tok, ds_valid, device, args.task, label_space)
        print(f"[Val][Ep{ep}] {val_metrics}")

        if args.use_wandb:
            log_payload = {f"val/{k}": v for k, v in val_metrics.items()}
            log_payload["epoch"] = ep
            wandb.log(log_payload)

        # primary: 멀티라벨이면 micro_f1, 아니면 balanced_acc
        primary = val_metrics.get("micro_f1") if args.task in {"issue","board"} else val_metrics.get("balanced_acc")
        os.makedirs(args.outdir, exist_ok=True)
        if float(primary) > best:
            best = float(primary)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            best_dir = os.path.join(args.outdir, f"best_ep{ep}_{timestamp}")
            
            model.save_pretrained(best_dir)
            tok.save_pretrained(best_dir)
            json.dump(val_metrics, open(os.path.join(best_dir, "val_metrics.json"), "w", encoding="utf-8"), indent=2)

            if args.use_wandb:
                wandb.summary["best_primary"] = best
                wandb.summary["best_ckpt_dir"] = best_dir

    # -------- Test --------
    if ds_test:
        test_metrics = evaluate_generate(model, tok, ds_test, device, args.task, label_space)
        print("[Test]", test_metrics)
        json.dump(test_metrics, open(os.path.join(best_dir or args.outdir, "test_metrics.json"), "w", encoding="utf-8"), indent=2)
        if args.use_wandb:
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
            for k, v in test_metrics.items():
                wandb.summary[f"test/{k}"] = v

if __name__ == "__main__":
    main()

