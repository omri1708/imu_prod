# examples/k8s_apply_example.py
from provenance.store import ProvenanceStore
from adapters.k8s_deploy import apply_manifests

if __name__ == "__main__":
    store = ProvenanceStore()
    art = apply_manifests(["./samples/k8s/deploy.yaml"], namespace="default", store=store)
    print("Release:", art)