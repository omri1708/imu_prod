package k8s.ingressclass

# ingress class must be allowed (imu/allowed-ingress-classes label supplied by gate template)
deny[msg] {
  input.kind == "Ingress"
  allowed := split(input.metadata.annotations["imu/allowed-ingress-classes"], ",")
  has_cls := input.spec.ingressClassName
  has_cls
  not allowed[_] == input.spec.ingressClassName
  msg := sprintf("ingress %s: class %s not allowed", [input.metadata.name, input.spec.ingressClassName])
}