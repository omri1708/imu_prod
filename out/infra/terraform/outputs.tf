output "api_url" {
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
