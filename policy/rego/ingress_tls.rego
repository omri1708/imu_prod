package k8s.ingress

# deny ingresses without TLS when gating requires TLS (chart renders annotation imu/gating-require-tls=true on metadata)
deny[msg] {
  input.kind == "Ingress"
  input.metadata.annotations["imu/gating-require-tls"] == "true"
  not input.spec.tls
  msg := sprintf("ingress %s: TLS required but not configured", [input.metadata.name])
}