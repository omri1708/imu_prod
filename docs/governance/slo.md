**`docs/governance/slo.md`**
```markdown
# SLO & Governance

- **p95 API:** Grafana → IMU SLO.
- **Gatekeeper:** Denies & Violations per constraint.
- **Kind-Smoke:** imu_kind_smoke_pass / duration_seconds metrics.
CI ל-Docs — .github/workflows/docs.yml

name: docs
on:
  push:
    branches: [ "main" ]
    paths: [ "docs/**", "mkdocs.yml" ]
  workflow_dispatch: {}

jobs:
  build-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install mkdocs mkdocs-material
      - run: mkdocs build --strict
      - uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site