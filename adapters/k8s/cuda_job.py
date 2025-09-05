#  adapters/k8s/cuda_job.py
# -*- coding: utf-8 -*-
from adapters.k8s.deploy import deploy_image
from adapters.contracts import ResourceRequired

def run_cuda_job(image: str, namespace: str = None):
    # דורש קלאסטר עם GPU + פלגין nvidia-device-plugin מותקן
    return deploy_image(image=image, name="imu-cuda-job", namespace=namespace, gpu=True)