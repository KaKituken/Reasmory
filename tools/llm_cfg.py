import os
from typing import Any, Dict

# Endpoints can be overridden per deployment (e.g. a private Azure resource).
GEMINI_BASE_URL = os.environ.get(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")


def _require_key(env_var: str, provider: str) -> str:
    """Read an API key from the environment.

    Keys are never hardcoded: set them in your shell or in a local `.env`
    (see `.env.example`). `.env` is gitignored.
    """
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise RuntimeError(
            f"Missing {provider} API key. Set the {env_var} environment variable "
            f"(see .env.example) before running with this model."
        )
    return key


def is_remote_api_model(model_path: str) -> bool:
    model_path_lower = model_path.lower()
    return any(provider in model_path_lower for provider in ("gpt", "gemini"))


def build_llm_cfg(model_path: str, temp: float, top_p: float) -> Dict[str, Any]:
    model_path_lower = model_path.lower()

    if "gemini" in model_path_lower:
        return {
            "model": model_path,
            "model_server": GEMINI_BASE_URL,
            "model_type": "qwenvl_oai",
            "api_key": _require_key("GEMINI_API_KEY", "Gemini"),
            "generate_cfg": {
                "top_p": 0.8,
                "temperature": temp,
            },
        }

    if "gpt" in model_path_lower:
        if "gpt-5-mini" in model_path_lower:
            if not AZURE_OPENAI_ENDPOINT:
                raise RuntimeError(
                    "gpt-5-mini is served through Azure OpenAI. Set AZURE_OPENAI_ENDPOINT "
                    "(and AZURE_OPENAI_API_KEY) -- see .env.example."
                )
            return {
                "model": "gpt-5-mini",
                "model_server": AZURE_OPENAI_ENDPOINT,
                "model_type": "qwenvl_azure",
                "api_version": AZURE_OPENAI_API_VERSION,
                "api_key": _require_key("AZURE_OPENAI_API_KEY", "Azure OpenAI"),
                "generate_cfg": {
                    "temperature": temp,
                },
            }
        if "gpt-5" in model_path_lower:
            return {
                "model": "gpt-5",
                "model_server": OPENAI_BASE_URL,
                "model_type": "qwenvl_oai",
                "api_key": _require_key("OPENAI_API_KEY", "OpenAI"),
                "generate_cfg": {
                    "temperature": temp,
                },
            }
        raise ValueError(f"Unsupported GPT model {model_path}.")

    if "claude" in model_path_lower:
        return {
            "model": "claude-sonnet-4-6",
            "model_server": ANTHROPIC_BASE_URL,
            "model_type": "qwenvl_oai",
            "api_key": _require_key("ANTHROPIC_API_KEY", "Anthropic"),
            "generate_cfg": {
                "temperature": temp,
            },
        }

    if "gemma-4-31b-it" in model_path_lower or "gemma-4-26b-a4b-it" in model_path_lower:
        # Served through the Gemini API surface.
        return {
            "model": model_path,
            "model_server": GEMINI_BASE_URL,
            "model_type": "qwenvl_oai",
            "api_key": _require_key("GEMINI_API_KEY", "Gemini"),
            "generate_cfg": {
                "top_p": 0.8,
                "temperature": temp,
            },
        }

    if model_path == "Qwen3-VL-8B-Instruct":
        model_name = "Qwen/Qwen3-VL-8B-Instruct"
    elif "gemma-4-E4B-it" in model_path:
        model_name = "google/gemma-4-E4B-it"
    elif "gemma-4-E2B-it" in model_path:
        model_name = "google/gemma-4-E2B-it"
    elif "spaceom" in model_path_lower:
        model_name = "remyxai/SpaceOm"
    elif "spatialladder" in model_path_lower:
        model_name = "hongxingli/SpatialLadder-3B"
    else:
        model_name = "Qwen/Qwen3-VL-4B-Instruct"
    # import ipdb; ipdb.set_trace()
    return {
        "model": model_name,
        "model_type": "transformers",
        "api_key": "EMPTY",
        "device": "cuda",
        "generate_cfg": {
            "top_p": top_p,
            "top_k": 20,
            "temperature": temp,
            "repetition_penalty": 1.0,
            "do_sample": False if temp == 0 else True,
        },
    }
