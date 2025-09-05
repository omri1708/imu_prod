# examples/android_build_example.py
from provenance.store import ProvenanceStore
from adapters.android_build import build_gradle

if __name__ == "__main__":
    store = ProvenanceStore()
    art = build_gradle(project_dir="./samples/android/MyApp", task="assembleRelease", store=store)
    print("Built:", art)