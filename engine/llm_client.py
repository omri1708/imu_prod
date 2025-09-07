# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json
from typing import Dict, Any, Optional, List
import urllib.request, urllib.error

class LLMError(RuntimeError): ...

def _http(method:str, url:str, headers:Dict[str,str], data:Optional[bytes]) -> Dict[str,Any]:
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise LLMError(f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')}")
    except Exception as e:
        raise LLMError(str(e))

class LLMClient:
    """ ספק אחיד: openai | azure | anthropic | gemini | ollama | oai_compatible """
    def __init__(self, provider:str|None=None, model:str|None=None, base_url:str|None=None):
        cfg = self._load_cfg()
        self.provider = provider or cfg.get("provider") or os.environ.get("LLM_PROVIDER","")
        self.model    = model    or cfg.get("model")    or os.environ.get("LLM_MODEL","gpt-4o-mini")
        self.base_url = base_url or cfg.get("base_url") or os.environ.get("LLM_BASE_URL","")
        self.keys = {
            "OPENAI_API_KEY":    os.environ.get("OPENAI_API_KEY"),
            "AZURE_OPENAI_KEY":  os.environ.get("AZURE_OPENAI_KEY"),
            "AZURE_OPENAI_EP":   os.environ.get("AZURE_OPENAI_ENDPOINT"),
            "AZURE_OPENAI_DEP":  os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
            "GOOGLE_API_KEY":    os.environ.get("GOOGLE_API_KEY"),
        }
    def _load_cfg(self)->Dict[str,Any]:
        p="config/imu.local.json"
        if os.path.exists(p):
            try: return json.loads(open(p,"r",encoding="utf-8").read())
            except: pass
        return {}
    # ---- API אחיד ----
    def chat(self, messages:List[Dict[str,str]], temperature:float=0.0, max_tokens:int=1024) -> str:
        prov = (self.provider or "").lower()
        if prov in ("openai","oai","oai_compatible") and self.base_url:
            return self._oai_compatible(messages, temperature, max_tokens)
        if prov in ("openai","oai"):
            return self._openai(messages, temperature, max_tokens)
        if prov in ("azure","azure_openai"):
            return self._azure(messages, temperature, max_tokens)
        if prov in ("anthropic","claude"):
            return self._anthropic(messages, temperature, max_tokens)
        if prov in ("gemini","google","palm"):
            return self._gemini(messages, temperature, max_tokens)
        if prov in ("ollama",):
            return self._ollama(messages, temperature, max_tokens)
        raise LLMError("LLM provider not configured. Set LLM_PROVIDER/LLM_MODEL env or config/imu.local.json.")
    # ---- מימושים ----
    def _openai(self, messages, temperature, max_tokens)->str:
        key = self.keys["OPENAI_API_KEY"];  url="https://api.openai.com/v1/chat/completions"
        if not key: raise LLMError("OPENAI_API_KEY missing")
        body={"model":self.model,"temperature":temperature,"max_tokens":max_tokens,"messages":messages}
        r=_http("POST",url,{"Authorization":f"Bearer {key}","Content-Type":"application/json"},json.dumps(body).encode())
        return r["choices"][0]["message"]["content"]
    def _azure(self,messages,temperature,max_tokens)->str:
        key=self.keys["AZURE_OPENAI_KEY"]; ep=self.keys["AZURE_OPENAI_EP"]; dep=self.keys["AZURE_OPENAI_DEP"]
        if not (key and ep and dep): raise LLMError("AZURE_OPENAI_{KEY,ENDPOINT,DEPLOYMENT} missing")
        url=f"{ep}/openai/deployments/{dep}/chat/completions?api-version=2024-02-15-preview"
        body={"temperature":temperature,"max_tokens":max_tokens,"messages":messages}
        r=_http("POST",url,{"api-key":key,"Content-Type":"application/json"},json.dumps(body).encode())
        return r["choices"][0]["message"]["content"]
    def _anthropic(self,messages,temperature,max_tokens)->str:
        key=self.keys["ANTHROPIC_API_KEY"]; url="https://api.anthropic.com/v1/messages"
        if not key: raise LLMError("ANTHROPIC_API_KEY missing")
        content=[{"type":"text","text": m["content"]} for m in messages if m["role"]!="system"]
        system="\n".join([m["content"] for m in messages if m["role"]=="system"])
        body={"model":self.model,"max_tokens":max_tokens,"temperature":temperature,"messages":[{"role":"user","content":content}],"system":system}
        r=_http("POST",url,{"x-api-key":key,"anthropic-version":"2023-06-01","Content-Type":"application/json"},json.dumps(body).encode())
        return "".join([c.get("text","") for c in r["content"] if c.get("type")=="text"])
    def _gemini(self,messages,temperature,max_tokens)->str:
        key=self.keys["GOOGLE_API_KEY"]
        if not key: raise LLMError("GOOGLE_API_KEY missing")
        url=f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={key}"
        parts=[]; system=[]
        for m in messages:
            if m["role"]=="system": system.append(m["content"])
            else: parts.append({"role":"user" if m["role"]!="assistant" else "model","parts":[{"text":m["content"]}]})
        body={"contents":parts,"generationConfig":{"temperature":temperature,"maxOutputTokens":max_tokens},"safetySettings":[]}
        r=_http("POST",url,{"Content-Type":"application/json"},json.dumps(body).encode())
        return r["candidates"][0]["content"]["parts"][0]["text"]
    def _ollama(self,messages,temperature,max_tokens)->str:
        base=os.environ.get("OLLAMA_BASE_URL","http://localhost:11434"); url=f"{base}/api/chat"
        body={"model":self.model,"messages":messages,"options":{"temperature":temperature,"num_predict":max_tokens},"stream":False}
        r=_http("POST",url,{"Content-Type":"application/json"},json.dumps(body).encode())
        return r["message"]["content"]
    def _oai_compatible(self,messages,temperature,max_tokens)->str:
        url=f"{self.base_url.rstrip('/')}/v1/chat/completions"; key=os.environ.get("LLM_API_KEY","")
        headers={"Content-Type":"application/json"};  headers.update({"Authorization":f"Bearer {key}"} if key else {})
        body={"model":self.model,"temperature":temperature,"max_tokens":max_tokens,"messages":messages}
        r=_http("POST",url,headers,json.dumps(body).encode())
        return r["choices"][0]["message"]["content"]
