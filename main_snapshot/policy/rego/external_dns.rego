package k8s.externaldns

deny[msg] {
  input.kind == "Deployment"
  input.metadata.labels["app.kubernetes.io/name"] == "external-dns"
  # must specify domainFilters and allowed zone
  not input.spec.template.spec.containers[_].args[_] == "--domain-filter=yourcompany.com"
  msg := "external-dns must restrict to domain-filter=yourcompany.com"
}