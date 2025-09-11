**`docs/runbooks/emergency.md`**
```markdown
# Emergency Rollback (“Big Red Button”)

במצבי כשל קריטיים:
1. פתח את **/ui/emergency.html** ולחץ “Rollback”.
2. או API:
```bash
curl -s -X POST http://API/controlplane/emergency/rollback \
  -H 'content-type: application/json' \
  -d '{"target":"umbrella","release":"umbrella","namespace":"prod","revision":1}'