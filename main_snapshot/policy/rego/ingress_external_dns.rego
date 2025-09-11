package k8s.ingress_externaldns

deny[msg] {
  input.kind == "Ingress"
  not input.spec.tls
  msg := sprintf("ingress %s: TLS is required by policy", [input.metadata.name])
}

deny[msg] {
  input.kind == "Ingress"
  not input.metadata.annotations["external-dns.alpha.kubernetes.io/hostname"]
  msg := sprintf("ingress %s: missing external-dns hostname annotation", [input.metadata.name])
}

deny[msg] {
  input.kind == "Ingress"
  h := input.metadata.annotations["external-dns.alpha.kubernetes.io/hostname"]
  h != ""
  not endswith(h, ".yourcompany.com")
  msg := sprintf("ingress %s: hostname %s not in allowed zone yourcompany.com", [input.metadata.name, h])
}