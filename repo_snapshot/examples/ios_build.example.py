# examples/ios_build_example.py
from provenance.store import ProvenanceStore
from adapters.ios_build import build_xcode

if __name__ == "__main__":
    store = ProvenanceStore()
    art = build_xcode(project_dir="./samples/ios/MyApp", scheme="MyApp", store=store)
    print("Built:", art)