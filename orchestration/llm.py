from __future__ import annotations

import logging
import os
import time

from langchain_openai import ChatOpenAI


log = logging.getLogger("virtual_lab")

_llm_instances: dict[float, ChatOpenAI] = {}


def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """
    Return cached LLM client for the given temperature.
    """
    global _llm_instances

    if temperature not in _llm_instances:
        vllm_url = os.getenv("VLLM_URL", "http://localhost:8000/v1")
        model_name = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-32B-Instruct")

        for attempt in range(3):
            try:
                _llm_instances[temperature] = ChatOpenAI(
                    base_url=vllm_url,
                    api_key=os.getenv("VLLM_API_KEY", "empty"),
                    model=model_name,
                    temperature=temperature,
                    request_timeout=600,
                    frequency_penalty=float(os.getenv("VLLM_FREQ_PENALTY", "0.3")),
                    presence_penalty=float(os.getenv("VLLM_PRES_PENALTY", "0.1")),
                    top_p=float(os.getenv("VLLM_TOP_P", "0.9")),
                    max_tokens=int(os.getenv("VLLM_MAX_NEW_TOKENS", "512")),
                )

                log.info(
                    "LLM client (temp=%.2f) initialised on attempt %d",
                    temperature,
                    attempt + 1,
                )
                break

            except Exception as e:
                log.warning("LLM init attempt %d/3 failed: %s", attempt + 1, e)
                time.sleep(5)

        if temperature not in _llm_instances:
            raise RuntimeError(
                f"Could not connect to vLLM server at {vllm_url} after 3 attempts."
            )

    return _llm_instances[temperature]