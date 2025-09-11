variable "project" {
  type        = string
  description = "Project name (prefix for resources)"
  default     = "imu"
}

variable "environment" {
  type        = string
  description = "Environment name (dev/staging/prod)"
  default     = "dev"
}

variable "network_name" {
  type        = string
  description = "Docker network name"
  default     = "imu_net"
}

variable "api_image" {
  type        = string
  description = "API container image (registry/repo:tag)"
  default     = "imu/api:latest"
}

variable "ws_image" {
  type        = string
  description = "WS container image (registry/repo:tag)"
  default     = "imu/ws:latest"
}

variable "ui_image" {
  type        = string
  description = "UI container image (registry/repo:tag)"
  default     = "imu/ui:latest"
}

variable "api_port" {
  type        = number
  description = "Host port mapped to API container:8000"
  default     = 8000
}

variable "ws_port" {
  type        = number
  description = "Host port mapped to WS container:8766"
  default     = 8766
}

variable "ui_port" {
  type        = number
  description = "Host port mapped to UI container:80"
  default     = 8080
}

variable "ws_internal_url" {
  type        = string
  description = "WS URL as reachable from API container"
  default     = "ws://ws:8766"
}

variable "log_level" {
  type        = string
  description = "Application log level"
  default     = "INFO"
}

variable "docker_host" {
  type        = string
  description = "Docker daemon host; null uses env DOCKER_HOST or local"
  default     = null
}
