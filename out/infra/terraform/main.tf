# Network
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
