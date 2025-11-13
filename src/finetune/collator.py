from typing import Dict, List
import torch

class Collator:
    def __init__(self, tok, max_length=8192, pad_to_multiple_of=8, add_eos=True):
        self.tok = tok
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.add_eos = add_eos

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        prompts = [b["prompt"] for b in batch]
        targets = [b["target_text"] + (self.tok.eos_token if (self.add_eos and self.tok.eos_token) else "") for b in batch]

        enc_p = self.tok(prompts, add_special_tokens=False)
        enc_t = self.tok(targets, add_special_tokens=False)

        input_ids, labels, attn = [],  [], []
        for p_ids, t_ids in zip(enc_p["input_ids"], enc_t["input_ids"]):
            ids = p_ids + t_ids
            lab = [-100] * len(p_ids) + t_ids[:]
            
            if len(ids) > self.max_length:
                ids = ids[-self.max_length:]
                lab = lab[-self.max_length:]
            input_ids.append(ids)
            labels.append(lab)
            attn.append([1] * len(ids))

        padded = self.tok.pad(
            {"input_ids": input_ids, "attention_mask": attn},
            padding=True, max_length=self.max_length, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
        )

        L = padded["input_ids"].shape[1]
        lab_pad = torch.full((len(labels), L), -100, dtype=torch.long)
        for i, l in enumerate(labels):
            lab_pad[i, :len(l)] = torch.tensor(l, dtype=torch.long)
        padded["labels"] = lab_pad

        return padded