# integration/llm_client.py
from __future__ import annotations
import os, requests

class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER","openai")   # openai|openrouter|anthropic
        self.model    = os.getenv("LLM_MODEL","gpt-4o-mini")
        self.api_key  = os.getenv("LLM_API_KEY")
        self.api_base = os.getenv("LLM_API_BASE","https://api.openai.com")
        self.anthropic_version = os.getenv("ANTHROPIC_VERSION","2023-06-01")
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is required")

    def chat(self, messages, temperature=0.2, max_tokens=800) -> str:
        if self.provider in ("openai","openrouter","openai-compatible","azure-openai"):
            url = self.api_base.rstrip("/") + "/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}","Content-Type":"application/json"}
            payload = {"model": self.model, "messages": messages,
                       "temperature": temperature, "max_tokens": max_tokens}
            r = requests.post(url, headers=headers, json=payload, timeout=60); r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        elif self.provider=="anthropic":
            url = self.api_base.rstrip("/") + "/v1/messages"
            headers = {"x-api-key": self.api_key, "anthropic-version": self.anthropic_version,
                       "content-type":"application/json"}
            sys = next((m["content"] for m in messages if m["role"]=="system"), None)
            user = next((m["content"] for m in messages if m["role"]=="user"), "")
            payload = {"model": self.model, "max_tokens": max_tokens,
                       "messages":[{"role":"user","content": user}],"system": sys}
            r = requests.post(url, headers=headers, json=payload, timeout=60); r.raise_for_status()
            data = r.json()
            return "".join(b.get("text","") for b in data.get("content",[]) if b.get("type")=="text")
        else:
            raise RuntimeError(f"Unsupported LLM_PROVIDER='{self.provider}'")
