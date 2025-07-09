import json, datetime, os
from pathlib import Path

import pdb
import time

from openai import OpenAI
from jsonschema import ValidationError, validate

class GPTBatchClient:
    def __init__(self, api_key, model, temperature, window, functions, function_call):
        self.client = OpenAI(api_key=api_key)
        self.model  = model
        self.window = window
        self.temperature = temperature
        self.functions = functions
        self.function_call = function_call

    def _call(self, file):
        return self.client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window=self.window,
        )

    def make_request_line(self, prompt, custom_id) -> dict:
        body = {
            "model":        self.model,
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user",   "content": prompt["user"]},
            ],
            "tools":        self.functions,
            "tool_choice":  self.function_call,
        }

        if self.temperature is not None and self.model.startswith("o"):
            print(f"temperature is ignored for model {self.model} (parameter unsupported)")

        supports_temp = (
            self.temperature is not None
            and not self.model.lower().startswith(("o"))
        )

        if supports_temp:
            body["temperature"] = self.temperature

        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }

    def validate_with_schema(self, result: dict):
        target_name = self.function_call["function"]["name"]
        try:
            schema = next(fn["function"]["parameters"] for fn in self.functions if fn["function"]["name"] == target_name)
        except StopIteration:
            raise ValueError(f"Function '{target_name}' not found in functions list")

        validate(instance=result, schema=schema)
        return result

    def generate_valid_json(self, batch_path):
        batch_input_files = self.client.files.create(
            file=open(batch_path, "rb"),
            purpose="batch"
        )

        res = self._call(batch_input_files)
        print(f"[Batch] created id={res.id}")

        while True:
            info = self.client.batches.retrieve(res.id)
            status = info.status

            rc = info.request_counts
            pct = 0 if rc.total == 0 else rc.completed / rc.total * 100
            logging.info("status=%-10s %5.1f%% (%s/%s)", status, pct, rc.completed, rc.total)

            if status == "completed":
                break
            if status == "expired":
                if not info.output_file_id:
                    raise RuntimeError(
                        f"[Batch] {res.id} expired but no output_file_id was provided; "
                        "the batch produced no downloadable result."
                    )
                break

            if status in ("failed", "cancelled"):
                raise RuntimeError(f"[Batch] {res.id} ended with status={status}")
            time.sleep(5)

        if info.output_file_id:
            raw = self.client.files.retrieve_content(info.output_file_id)
        else:
            return {}

        validated: Dict[str, dict] = {}
        dropped:   list[str] = []

        for ln in raw.splitlines():
            rec   = json.loads(ln)
            document_id   = rec["custom_id"]

            try:
                result = rec["response"]["body"]["choices"][0]["message"] \
                              ["tool_calls"][0]["function"]["arguments"]
                result_json = json.loads(result)
            except (KeyError, json.JSONDecodeError):
                dropped.append(document_id); continue

            try:
                self.validate_with_schema(result_json)
                validated[document_id] = result_json
            except ValidationError:
                dropped.append(document_id)

        if dropped:
            print(f"[WARN] {len(dropped)} responses invalid → skipped (e.g. {dropped[:2]}...)")

        return validated