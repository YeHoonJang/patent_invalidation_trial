import time
import random
from anthropic import HUMAN_PROMPT, AI_PROMPT
import anthropic
from anthropic._exceptions import OverloadedError, RateLimitError

class ClaudeClient:
    def __init__(self, api_key, model, temperature, max_tokens, functions, timeout):
        self.client = anthropic.Client(api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.functions = functions


    def split_opinion(self, prompt):
        system = (
            "You are a legal assistant who classifies PTAB legal text by speaker.\n\n"
            "Please output **only** a valid JSON object matching the following schema (no extra text, no markdown fences):\n"
            f"{self.functions}"
        )

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model = self.model,
                    system=system,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                return response
            
            except (OverloadedError, RateLimitError) as e:
                wait = (2 ** attempt) + random.random()
                print(f"[WARN] {type(e).__name__}, Retry after {wait:.1f}s ({attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue

        