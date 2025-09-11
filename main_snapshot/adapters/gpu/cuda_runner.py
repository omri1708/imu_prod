import os, shutil, subprocess, tempfile, textwrap, multiprocessing as mp
from ..contracts import ResourceRequired

CUDA_SAMPLE = r"""
#include <stdio.h>
__global__ void addOne(int *a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    a[idx] += 1;
}
int main() {
    const int N = 1024;
    int *a, *d_a;
    a = (int*)malloc(N*sizeof(int));
    for (int i=0;i<N;++i) a[i]=i;
    cudaMalloc(&d_a, N*sizeof(int));
    cudaMemcpy(d_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    addOne<<<N/256,256>>>(d_a);
    cudaMemcpy(a, d_a, N*sizeof(int), cudaMemcpyDeviceToHost);
    printf("OK %d\\n", a[10]);
    cudaFree(d_a);
    free(a);
    return 0;
}
"""

def _compile_run_single_gpu(device_id: int) -> str:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise ResourceRequired("CUDA Toolkit (nvcc)", "Install NVIDIA CUDA Toolkit; ensure 'nvcc' in PATH")
    with tempfile.TemporaryDirectory() as td:
        cu = os.path.join(td, "a.cu")
        exe = os.path.join(td, f"a_{device_id}")
        with open(cu, "w") as f: f.write(CUDA_SAMPLE)
        subprocess.run([nvcc, cu, "-o", exe, "-arch=sm_70"], check=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device_id)
        out = subprocess.run([exe], check=True, stdout=subprocess.PIPE, env=env, text=True).stdout.strip()
        return out

def compile_and_run_cuda(devices: int = 1):
    """
    מריץ מקבילי על מספר כרטיסים אם יש. אין nvcc → ResourceRequired.
    """
    if devices <= 1:
        return {"results": [_compile_run_single_gpu(0)]}
    with mp.Pool(processes=devices) as pool:
        outs = pool.map(_compile_run_single_gpu, list(range(devices)))
    return {"results": outs}