# examples/unity_build_example.py
import os
from provenance.store import ProvenanceStore
from adapters.unity_cli import build_unity_project

if __name__ == "__main__":
    os.environ.setdefault("UNITY_PATH", "/Applications/Unity/Hub/Editor/2022.3.XXf1/Unity.app/Contents/MacOS/Unity")
    store = ProvenanceStore()
    art = build_unity_project(project_dir="./samples/unity/MyGame", build_target="StandaloneOSX", store=store)
    print("Built:", art)