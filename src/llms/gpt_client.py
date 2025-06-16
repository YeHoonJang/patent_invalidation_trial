import asyncio
import time
import random
from openai import AsyncOpenAI, OpenAI
from openai import RateLimitError

import pdb


class GPTClient:
    def __init__(self, api_key, model, temperature, functions, function_call):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.functions = functions
        self.function_call = function_call


    async def _call(self, prompt):
        if self.model.startswith("gpt"):
            return await self.client.chat.completions.create(
                model = self.model,
                messages= [
                    {"role": "system", "content": "You are a legal assistant who classifies PTAB legal text by speaker."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                functions=self.functions,
                function_call=self.function_call
                )
        elif self.model.startswith("o"):
            return await self.client.chat.completions.create(
                model = self.model,
                messages = [
                    {"role": "system", "content": "You are a legal assistant who classifies PTAB legal text by speaker."},
                    {"role": "user", "content": prompt}
                ],
                functions=self.functions,
                function_call=self.function_call
                )

    async def split_opinion(self, prompt):
        retry_count = 0
        while True:
            try:
                return await self._call(prompt)
                        
            except RateLimitError as e:
                retry_count += 1
                if retry_count >= 5:
                    wait = 3600
                    retry_count = 0
                    print("f[RateLimit] Retries exceeded 5 times. Please wait an hour.")
                else:
                    wait = (2 ** (retry_count - 1)) + random.random()
                    print(f"[WARN] {type(e).__name__}, Retry after {wait:.1f}s ({retry_count}/5)...")
                    await asyncio.sleep(wait)