# Argo CD Image Updater

Install (cluster-wide):
kubectl create namespace argocd-image-updater
kubectl apply -n argocd-image-updater -f https://raw.githubusercontent.com/argoproj-labs/argocd-image-updater/v0.12.0/manifests/install.yaml


Configure registry auth (GHCR uses PAT via imagePullSecret or env):
kubectl -n argocd-image-updater create secret generic image-updater-secret
--from-literal=argocd.token=$ARGOCD_TOKEN
--from-literal=github.token=$GITHUB_TOKEN

The annotations on `umbrella-prod` Application guide updates:
- `argocd-image-updater.argoproj.io/image-list` — tracked images
- `.../helm.image-values.<name>` — where to write tag value in Helm values
- `.../write-back-method: git` — commits back to repo (main)