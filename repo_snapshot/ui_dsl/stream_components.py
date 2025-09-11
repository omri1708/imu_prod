# ui_dsl/stream_components.py
from __future__ import annotations

def progress_bar(topic: str, label: str="Progress"):
    return {"type":"progress_bar","topic":topic,"label":label}

def event_timeline(topic: str, max_items:int=200):
    return {"type":"event_timeline","topic":topic,"max_items":max_items}