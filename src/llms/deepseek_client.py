import asyncio
import json
import pdb
import random
import time

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import transformers
import torch


class DeepSeekClient:
    def __init__(self, **model_config):
        self.model_name = model_config.pop("model")
        self.load_model(model_config=model_config)

    def load_model(
        self, device_map: str = "cuda", model_config: dict = {}, cache_dir: str = None
    ):
        config = AutoConfig.from_pretrained(self.model_name, **model_config)

        torch_dtype = model_config.pop("torch_dtype", None)
        if torch_dtype and isinstance(torch_dtype, str) and torch_dtype != "auto":
            torch_dtype = getattr(torch, torch_dtype)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                  **model_config,
                                                  trust_remote_code=True)
        if "cuda" in device_map or "cpu" in device_map:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                config=config,
                trust_remote_code=True,
                **({ "cache_dir": cache_dir } if cache_dir else {}),
            )

        model.generation_config = GenerationConfig.from_pretrained(model_name)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

        self.tokenizer = tokenizer
        self.model = model

    async def _call(self, prompt):
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]

        input_tensor = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        pdb.set_trace()

        response = self.model.generate(
            input_tensor,
            max_new_tokens=100)
        generated_tokens = response[0][input_tensor.shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    async def generate_valid_json(self, prompt: str) -> dict:
        retry_count = 0
        while True:
            try:
                response = await self._call(prompt)
                return response
            except Exception as e:
                retry_count += 1

                if retry_count >= 50:
                    wait = 3600
                    retry_count = 0
                    print("f[RateLimit] Retries exceeded 5 times. Please wait an hour.")
                else:
                    wait = (2 ** (retry_count - 1)) + random.random()
                    print(f"[WARN] {type(e).__name__}, Retry after {wait:.1f}s ({retry_count}/5)...")
                await asyncio.sleep(wait)
