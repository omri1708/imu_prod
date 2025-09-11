# examples/usage_android_ios_unity_cuda_k8s.py
from engine.integrations_registry import REGISTRY
from engine.contracts_policy import policy_wrap

def ex_android(project_dir: str):
    return policy_wrap("android.build", REGISTRY["android.build"], project_dir, "Release")

def ex_ios(project_path: str, scheme: str):
    return policy_wrap("ios.build", REGISTRY["ios.build"], project_path, scheme, "iphoneos", "Release")

def ex_unity(project_path: str, method: str):
    return policy_wrap("unity.build", REGISTRY["unity.build"], project_path, method)

def ex_cuda():
    code = r'''
    #include <stdio.h>
    __global__ void add1(int *a){ int i = blockIdx.x*blockDim.x + threadIdx.x; if(i<32) a[i]+=1; }
    int main(){ const int N=32; int h[N]; for(int i=0;i<N;++i) h[i]=i;
      int *d; cudaMalloc(&d, N*sizeof(int)); cudaMemcpy(d,h,N*sizeof(int),cudaMemcpyHostToDevice);
      add1<<<1,32>>>(d); cudaMemcpy(h,d,N*sizeof(int),cudaMemcpyDeviceToHost);
      for(int i=0;i<N;++i) printf("%d ",h[i]); printf("\n"); cudaFree(d); return 0; }
    '''
    return policy_wrap("cuda.run", REGISTRY["cuda.run"], code, "add1")

def ex_k8s():
    return policy_wrap("k8s.job", REGISTRY["k8s.job"], name="hello-job", image="busybox",
                       command=["/bin/sh","-lc","echo hi && sleep 1 && echo bye"], namespace="default")