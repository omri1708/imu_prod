# run_examples.py
from examples.usage_android_ios_unity_cuda_k8s import ex_android, ex_ios, ex_unity, ex_cuda, ex_k8s

print(ex_cuda())      # יריץ nvcc אם קיים
print(ex_k8s())       # ישלח Job אמיתי אם kubectl מחובר לקלאסטר