# PATH: engine/blueprints/iac_terraform.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List

"""
Legal / Use Anchor:
- No violation of any ToS; lawful, user-controlled usage.
- Generates Infrastructure-as-Code (Terraform) files only.
- No implicit installs/executions; output is auditable.

Blueprint ID: "iac.terraform"
Produces:
  infra/terraform/provider.tf
  infra/terraform/variables.tf
  infra/terraform/main.tf
  infra/terraform/outputs.tf
"""

BLUEPRINT_ID = "iac.terraform"


def _safe(s: str, default: str) -> str:
    s = (s or "").strip()
    return s if s else default


def _int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def generate(spec: Dict[str, Any]) -> Dict[str, bytes]:
    """
    Generate Terraform scaffold targeting the 'docker' provider:
    - Network + three containers (ws/api/ui) with healthchecks and ports.
    - Variables allow overriding images/ports/network/env names at apply-time.

    Expected to integrate with CI that pushes images to a registry and runs:
      terraform init && terraform apply -auto-approve \
        -var "api_image=REG/OWNER/api:TAG" \
        -var "ws_image=REG/OWNER/ws:TAG" \
        -var "ui_image=REG/OWNER/ui:TAG"
    """
    infra = spec.get("infra") or {}
    project = _safe(infra.get("project") or spec.get("project") or spec.get("title"), "imu")
    environment = _safe(infra.get("environment") or spec.get("environment"), "dev")

    # Derive reasonable defaults if not provided in spec
    api_image = _safe((infra.get("images") or {}).get("api"), "imu/api:latest")
    ws_image = _safe((infra.get("images") or {}).get("ws"), "imu/ws:latest")
    ui_image = _safe((infra.get("images") or {}).get("ui"), "imu/ui:latest")

    api_port = _int((infra.get("ports") or {}).get("api"), 8000)
    ws_port = _int((infra.get("ports") or {}).get("ws"), 8766)
    ui_port = _int((infra.get("ports") or {}).get("ui"), 8080)

    network_name = _safe(infra.get("network") or "imu_net", "imu_net")
    ws_internal_url = _safe(infra.get("ws_internal_url"), "ws://ws:8766")
    log_level = _safe(infra.get("log_level"), "INFO")

    provider_tf = f"""terraform {{
  required_version = ">= 1.5.0"
  required_providers {{
    docker = {{
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }}
  }}
}}

# Provider 'docker' can target local or remote Docker (via DOCKER_HOST/tls env)
provider "docker" {{
  host = var.docker_host
}}
"""

    variables_tf = f"""variable "project" {{
  type        = string
  description = "Project name (prefix for resources)"
  default     = "{project}"
}}

variable "environment" {{
  type        = string
  description = "Environment name (dev/staging/prod)"
  default     = "{environment}"
}}

variable "network_name" {{
  type        = string
  description = "Docker network name"
  default     = "{network_name}"
}}

variable "api_image" {{
  type        = string
  description = "API container image (registry/repo:tag)"
  default     = "{api_image}"
}}

variable "ws_image" {{
  type        = string
  description = "WS container image (registry/repo:tag)"
  default     = "{ws_image}"
}}

variable "ui_image" {{
  type        = string
  description = "UI container image (registry/repo:tag)"
  default     = "{ui_image}"
}}

variable "api_port" {{
  type        = number
  description = "Host port mapped to API container:8000"
  default     = {api_port}
}}

variable "ws_port" {{
  type        = number
  description = "Host port mapped to WS container:8766"
  default     = {ws_port}
}}

variable "ui_port" {{
  type        = number
  description = "Host port mapped to UI container:80"
  default     = {ui_port}
}}

variable "ws_internal_url" {{
  type        = string
  description = "WS URL as reachable from API container"
  default     = "{ws_internal_url}"
}}

variable "log_level" {{
  type        = string
  description = "Application log level"
  default     = "{log_level}"
}}

variable "docker_host" {{
  type        = string
  description = "Docker daemon host; null uses env DOCKER_HOST or local"
  default     = null
}}
"""

    main_tf = """# Network
resource "docker_network" "net" {
  name = var.network_name
}

# WS image and container
resource "docker_image" "ws" {
  name         = var.ws_image
  keep_locally = true
}

resource "docker_container" "ws" {
  name  = "${var.project}-${var.environment}-ws"
  image = docker_image.ws.image_id

  networks_advanced {
    name = docker_network.net.name
  }

  env = [
    "LOG_LEVEL=${var.log_level}"
  ]

  ports {
    internal = 8766
    external = var.ws_port
  }

  healthcheck {
    test     = ["CMD-SHELL", "nc -z 127.0.0.1 8766 || exit 1"]
    interval = "10s"
    timeout  = "3s"
    retries  = 10
  }

  restart = "unless-stopped"
}

# API image and container
resource "docker_image" "api" {
  name         = var.api_image
  keep_locally = true
}

resource "docker_container" "api" {
  name  = "${var.project}-${var.environment}-api"
  image = docker_image.api.image_id

  networks_advanced {
    name = docker_network.net.name
  }

  env = [
    "WS_URL=${var.ws_internal_url}",
    "PYTHONPATH=/app",
    "LOG_LEVEL=${var.log_level}"
  ]

  ports {
    internal = 8000
    external = var.api_port
  }

  depends_on = [docker_container.ws]

  healthcheck {
    test     = ["CMD", "wget", "-qO-", "http://127.0.0.1:8000/readyz"]
    interval = "10s"
    timeout  = "3s"
    retries  = 12
  }

  restart = "unless-stopped"
}

# UI image and container
resource "docker_image" "ui" {
  name         = var.ui_image
  keep_locally = true
}

resource "docker_container" "ui" {
  name  = "${var.project}-${var.environment}-ui"
  image = docker_image.ui.image_id

  networks_advanced {
    name = docker_network.net.name
  }

  ports {
    internal = 80
    external = var.ui_port
  }

  restart = "unless-stopped"
}
"""

    outputs_tf = """output "api_url" {
  description = "API base URL"
  value       = "http://localhost:${var.api_port}"
}

output "ws_url" {
  description = "WS URL"
  value       = "ws://localhost:${var.ws_port}"
}

output "ui_url" {
  description = "UI base URL"
  value       = "http://localhost:${var.ui_port}"
}
"""

    return {
        "infra/terraform/provider.tf": provider_tf.encode("utf-8"),
        "infra/terraform/variables.tf": variables_tf.encode("utf-8"),
        "infra/terraform/main.tf": main_tf.encode("utf-8"),
        "infra/terraform/outputs.tf": outputs_tf.encode("utf-8"),
    }
