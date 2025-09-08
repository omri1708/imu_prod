# PATH: engine/blueprints/ci_github_actions.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any

"""
Legal / Use Anchor:
- No ToS violations; lawful CI workflow generation.
- User controls secrets/registries; no hidden behavior.
- Steps are explicit, auditable, and can be removed/modified safely.

Blueprint ID: "ci.github_actions"
Produces:
  .github/workflows/ci.yml
"""

BLUEPRINT_ID = "ci.github_actions"


def _val(d: Dict[str, Any], *keys: str, default: str = "") -> str:
    cur: Any = d
    for k in keys:
        cur = (cur or {}).get(k)
    return (str(cur).strip() if cur else default) or default


def generate(spec: Dict[str, Any]) -> Dict[str, bytes]:
    """
    Workflow assumptions:
      - Python tests live under services/api/test_*.py
      - Dockerfiles:
          docker/prod/api/Dockerfile
          docker/ws/Dockerfile
          docker/ui/Dockerfile
      - Terraform files under infra/terraform/
      - Registry: GHCR by default; can be overridden via secrets/env.

    Required repo secrets (recommended):
      - CR_PAT (optional if using GHCR + GITHUB_TOKEN)
      - TF_API_TOKEN (optional; only if using Terraform Cloud/Remote)
    """
    project = _val(spec, "infra", "project", default="imu")
    environment = _val(spec, "infra", "environment", default="dev")

    workflow_yml = f"""name: CI

on:
  push:
    branches: [ main, master ]
  pull_request:

env:
  PROJECT: {project}
  ENVIRONMENT: {environment}
  PY_VERSION: "3.11"
  REGISTRY: ghcr.io
  IMAGE_NAMESPACE: ${{{{ github.repository_owner }}}}
  IMAGE_TAG: ${{{{ github.sha }}}}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{{{ env.PY_VERSION }}}}

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          if [ -f services/api/requirements.txt ]; then pip install -r services/api/requirements.txt; fi
          pip install pytest
      - name: Run pytest
        run: |
          if [ -d services/api ]; then pytest -q services/api; else echo "No API tests folder"; fi

  build_and_push:
    needs: [ test ]
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{{{ env.REGISTRY }}}}
          username: ${{{{ github.actor }}}}
          password: ${{{{ secrets.GITHUB_TOKEN }}}}

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build & Push WS
        uses: docker/build-push-action@v6
        with:
          context: .
          file: docker/ws/Dockerfile
          push: true
          tags: |
            ${{{{ env.REGISTRY }}}}/${{{{{ env.IMAGE_NAMESPACE }}}}/ws:${{{{ env.IMAGE_TAG }}}}
            ${{{{ env.REGISTRY }}}}/${{{{{ env.IMAGE_NAMESPACE }}}}/ws:latest

      - name: Build & Push API
        uses: docker/build-push-action@v6
        with:
          context: .
          file: docker/prod/api/Dockerfile
          push: true
          tags: |
            ${{{{ env.REGISTRY }}}}/${{{{{ env.IMAGE_NAMESPACE }}}}/api:${{{{ env.IMAGE_TAG }}}}
            ${{{{ env.REGISTRY }}}}/${{{{{ env.IMAGE_NAMESPACE }}}}/api:latest

      - name: Build & Push UI
        uses: docker/build-push-action@v6
        with:
          context: .
          file: docker/ui/Dockerfile
          push: true
          tags: |
            ${{{{ env.REGISTRY }}}}/${{{{{ env.IMAGE_NAMESPACE }}}}/ui:${{{{ env.IMAGE_TAG }}}}
            ${{{{ env.REGISTRY }}}}/${{{{{ env.IMAGE_NAMESPACE }}}}/ui:latest

      - name: Export images for deploy
        id: img
        run: |
          echo "API_IMAGE=${{{{ env.REGISTRY }}}}/${{{{{ env.IMAGE_NAMESPACE }}}}/api:${{{{ env.IMAGE_TAG }}}}" >> $GITHUB_OUTPUT
          echo "WS_IMAGE=${{{{ env.REGISTRY }}}}/${{{{{ env.IMAGE_NAMESPACE }}}}/ws:${{{{ env.IMAGE_TAG }}}}" >> $GITHUB_OUTPUT
          echo "UI_IMAGE=${{{{ env.REGISTRY }}}}/${{{{{ env.IMAGE_NAMESPACE }}}}/ui:${{{{ env.IMAGE_TAG }}}}" >> $GITHUB_OUTPUT

  terraform_apply:
    needs: [ build_and_push ]
    runs-on: ubuntu-latest
    permissions:
      contents: read
    defaults:
      run:
        working-directory: infra/terraform
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3

      - name: Terraform Init
        run: |
          terraform init -input=false

      - name: Terraform Apply
        env:
          API_IMAGE: ${{{{ needs.build_and_push.outputs.API_IMAGE || steps.img.outputs.API_IMAGE }}}}
          WS_IMAGE:  ${{{{ needs.build_and_push.outputs.WS_IMAGE  || steps.img.outputs.WS_IMAGE }}}}
          UI_IMAGE:  ${{{{ needs.build_and_push.outputs.UI_IMAGE  || steps.img.outputs.UI_IMAGE }}}}
        run: |
          terraform apply -auto-approve \
            -var "project=${{{{ env.PROJECT }}}}" \
            -var "environment=${{{{ env.ENVIRONMENT }}}}" \
            -var "api_image=${{{{ env.API_IMAGE }}}}" \
            -var "ws_image=${{{{ env.WS_IMAGE }}}}" \
            -var "ui_image=${{{{ env.UI_IMAGE }}}}"
"""
    return {".github/workflows/ci.yml": workflow_yml.encode("utf-8")}
