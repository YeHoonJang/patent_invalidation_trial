from .gpt_client import GPTClient
from .gpt_batch_client import GPTBatchClient
from .claude_client import ClaudeClient
from .gemini_client import GeminiClient
from .llama_client import LlamaClient
from .qwen_client import QwenClient
from .solar_client import SolarClient
from .mistral_client import MistralClient


def get_llm_client(model_name, api_key, **kwargs):
    name = model_name.lower()

    if (name == "gpt") or (name == "gpt-o"):
        return GPTClient(api_key, **kwargs)
    elif name == "claude":
        return ClaudeClient(api_key, **kwargs)
    elif name == "gemini":
        return GeminiClient(api_key, **kwargs)
    elif name == "solar":
        return SolarClient(api_key, **kwargs)
    elif name == "llama":
        return LlamaClient(**kwargs)
    elif name == "qwen":
        return QwenClient(**kwargs)
    elif name == "mistral":
        return MistralClient(**kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def get_llm_batch_client(model_name, api_key, **kwargs):
    name = model_name.lower()

    if (name == "gpt") or (name == "gpt-o"):
        return GPTBatchClient(api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")