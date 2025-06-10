import time
import random
from openai import OpenAI
from openai import RateLimitError


class GPTClient:
    def __init__(self, api_key, model, temperature, functions, function_call):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.functions = functions
        self.function_call = function_call


    def split_opinion(self, prompt):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model = self.model,
                    messages= [
                        {"role": "system", "content": "You are a legal assistant who classifies PTAB legal text by speaker."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    functions=self.functions,
                    function_call=self.function_call
                )

                return response
            
            except RateLimitError as e:
                wait = (2 ** attempt) + random.random()
                print(f"[WARN] {type(e).__name__}, Retry after {wait:.1f}s ({attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue