# examples/cuda_compile_example.py
from provenance.store import ProvenanceStore
from adapters.cuda_jobs import compile_cuda_kernel

if __name__ == "__main__":
    store = ProvenanceStore()
    art = compile_cuda_kernel("./samples/cuda/vec_add.cu", store=store)
    print("Built:", art)