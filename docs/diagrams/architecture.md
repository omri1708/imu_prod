**`docs/diagrams/architecture.md`**
```markdown
# Architecture

```mermaid
flowchart LR
  Dev((Dev))-->CI[CI Pipelines]
  CI-->ArgoCD
  ArgoCD-->Umbrella[Umbrella Helm]
  Umbrella-->CP[Control Plane Chart]
  CP-->API((API))
  CP-->WS((WFQ-WS))
  CP-->UI((Static UI))
  CP-->Prometheus
  CP-->Gatekeeper
  CP-->Grafana