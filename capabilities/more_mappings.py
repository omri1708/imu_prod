# capabilities/more_mappings.py
# Additional mappings for cosign, helm, minikube, etc.
WINGET_EXTRA = {
    "cosign": "Sigstore.cosign",
    "helm":   "Helm.Helm",
    "minikube": "Googlecloudsdk.Minikube"
}

BREW_EXTRA = {
    "cosign": "cosign",
    "helm": "helm",
    "minikube": "minikube"
}

APT_EXTRA = {
    # cosign via snap recommended; leave apt empty or fallback to script
    "helm": "helm",
    "minikube": "minikube"
}