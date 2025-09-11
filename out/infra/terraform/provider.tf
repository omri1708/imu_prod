terraform {
  required_version = ">= 1.5.0"
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

# Provider 'docker' can target local or remote Docker (via DOCKER_HOST/tls env)
provider "docker" {
  host = var.docker_host
}
