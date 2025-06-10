import time
import random
import pdb
from google import genai
from google.genai import types

class GeminiClient:
    def __init__(self, api_key, model, temperature, functions):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.functions = functions


    def split_opinion(self, prompt):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model = self.model,
                    contents = [
                        types.Content(
                            role="user",
                            parts=[types.Part(text=prompt)]
                        )
                    ],
                    config = types.GenerateContentConfig(
                        system_instruction=(
                            "You are a legal assistant who classifies PTAB legal text by speaker."
                        ),
                        response_mime_type="application/json",
                        response_schema=self.functions
                    )
                )

                return response
            
            except Exception as e:
                pdb.set_trace()
                wait = (2 ** attempt) + random.random()
                print(f"[WARN] {type(e).__name__}, Retry after {wait:.1f}s ({attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue